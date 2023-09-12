import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread, imsave
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import wandb
import yaml
import shutil
import mcubes
import trimesh
from glob import glob

from copy import deepcopy

from .delta import DELTA
from .model.pose import PoseModel
from .dataset import build_dataset
from .utils.log_util import WandbLogger
from .utils import util, lossfunc, rotation_converter
from .render.mesh_helper import render_shape
from .utils.metric_util import Evaluator
from .trainer import Trainer

class Evaluator(Trainer):
    def __init__(self, config=None):
        super(Evaluator, self).__init__(config=config)

    def optimize(self, args):
        '''
        Given trained models, initial pose,
        optimize (refine) the pose parameters to fit the input image
        '''
        savefolder = os.path.join(self.cfg.savedir, 'evaluation', 'optimize')
        os.makedirs(savefolder, exist_ok=True)
        
        # load test data
        test_dataset = build_dataset.build_train(self.cfg.dataset, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        for batch in tqdm(test_dataloader):
            # load data
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}.jpg')
            
            if os.path.exists(savepath):
                continue
            # setup optimizer
            batch['init_full_pose'] = batch['full_pose'].clone()
            pose = batch['full_pose']
            init_pose = rotation_converter.batch_matrix2axis(pose[0])[None,...] + 1e-8
            pose = torch.nn.Parameter(init_pose.detach())
            cam = torch.nn.Parameter(batch['cam'].detach())
            exp = torch.nn.Parameter(batch['exp'].detach())
            init_lights = torch.rand((1, 6, 6)).float().to(self.device)
            lights = torch.nn.Parameter(init_lights)
            parameters = [
                    {'params': [cam], 'lr': 1e-4},
                    {'params': [pose], 'lr': 1e-4},
                    {'params': [exp], 'lr': 1e-4},
                    {'params': [lights], 'lr': 2e-3},
                    ]
            pose_optimizer = torch.optim.Adam(params=parameters)   
            
            # run optimization
            logger.info(f'Optimize frame {frame_id}')
            n_iters = 500
            for i in tqdm(range(n_iters)):
                batch_pose = rotation_converter.batch_axis2matrix(pose.reshape(-1, 3)).reshape(1, 55, 3, 3)
                batch['full_pose'] = batch_pose
                batch['cam'] = cam
                batch['exp'] = exp
                batch['light'] = lights
                util.move_dict_to_device(batch, self.device)
                opdict = self.model(batch)
                #-- loss
                #### ----------------------- Losses
                losses = {}
                if self.cfg.use_mesh:
                    mesh_losses = self._compute_mesh_loss(batch, opdict)
                    losses = {**losses, **mesh_losses}
                
                if self.cfg.use_nerf:
                    nerf_losses = self._compute_nerf_loss(batch, opdict)
                    losses = {**losses, **nerf_losses}
                    
                #########################################################d
                all_loss = 0.
                losses_key = losses.keys()
                for key in losses_key:
                    all_loss = all_loss + losses[key]
                losses['all_loss'] = all_loss
    
                ## backward
                pose_optimizer.zero_grad()
                all_loss = losses['all_loss']
                all_loss.backward()
                pose_optimizer.step()
                
                ### visualization
                # if i % 50 == 0:
                #     opdict = self.model(batch, render_image=True, render_normal=self.cfg.nerf.render_normal)
                #     visdict = {
                #         'image': batch['image'],
                #         'render': opdict['nerf_image']}
                #     os.makedirs(os.path.join(savefolder,  f'opt_{self.cfg.exp_name}_f{frame_id}'), exist_ok=True)
                #     savepath = os.path.join(savefolder, f'opt_{self.cfg.exp_name}_f{frame_id}', f'{self.cfg.exp_name}_f{frame_id}_{i:06}.jpg')
                #     grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                #     print(savepath)
                    
                #     ## test number gt = batch['image']
                #     pred = opdict['nerf_image']
                #     gt = batch['image'] 
                #     val_metrics = self.evaluator(pred, gt)
                #     val_info = f'step {i}'
                #     for k, v in val_metrics.items():
                #         val_info = val_info + f'{k}: {v:.6f}, '
                #     print(val_info)
            
            opdict = self.model.forward_vis(batch)
            visdict = {
                        'image': batch['image'],
                        'render': opdict['render'],
                        'render_hybrid': opdict['render_hybrid']}
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}'), exist_ok=True)
            for key in visdict.keys():
                image = visdict[key]
                cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))       
            print(savepath)
            logger.info(f'Optimize frame {frame_id} done')
            pred = opdict['render']
            gt = batch['image'] 
            val_metrics = self.evaluator(pred, gt)
            val_info = f'step {i}'
            for k, v in val_metrics.items():
                val_info = val_info + f'{k}: {v:.6f}, '
            print(val_info)
                
    def run(self, args, method='delta', region='face_neck_shoulder'):
        ''' run evaluation
        metrics: PSNR, SSIM, LPIPS
        '''
        self.device = 'cpu'
        self.evaluator.to(self.device)
        
        savefolder = os.path.join(self.cfg.savedir, 'evaluation', 'comparison', f'{region}', f'{method}')
        os.makedirs(savefolder, exist_ok=True)
        
        test_l1 = []
        test_psnr = []
        test_ssim = []
        test_lpips = []
        
        # load test data
        test_dataset = build_dataset.build_train(self.cfg.dataset, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        subject = test_dataset.subject
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            frame_id = int(sample['frame_id'])
            image = sample['image']
            mask = sample['mask']
            orig_imagepath = sample['imagepath']
            orig_image = imread(orig_imagepath)[:,:,:3]/255.
            orig_image = torch.from_numpy(orig_image).permute(2,0,1).float().to(image.device)
            
            # load pred data
            if method == 'nha':
                if subject == 'MVI_1810':
                    pred_folder = '/is/cluster/yfeng/other_github/neural-head-avatars/experiments/MVI_1810/lightning_logs/version_1/test'
                elif subject == 'person_0000':
                    pred_folder = '/is/cluster/yfeng/other_github/neural-head-avatars/experiments/person_0000/lightning_logs/version_0/test'
                elif subject == 'person_0004':
                    pred_folder = '/is/cluster/yfeng/other_github/neural-head-avatars/experiments/person_0004/lightning_logs/version_0/test'
                elif subject == 'b0_0':
                    pred_folder = '/is/cluster/yfeng/other_github/neural-head-avatars/experiments/b0_0/lightning_logs/version_0/test'
                imagepath = os.path.join(pred_folder, f'rgb_{frame_id:06d}.png')
            elif method == 'IMavatar':
                if subject == 'MVI_1810':
                    pred_folder = '/home/yfeng/other_github/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/epoch_1000/rgb'
                elif subject == 'person_0000':
                    pred_folder = '/home/yfeng/other_github/IMavatar/data/experiments/nha/IMavatar/person_0000/eval/person_0000/epoch_466/rgb'
                elif subject == 'person_0004':
                    pred_folder = '/home/yfeng/other_github/IMavatar/data/experiments/nha_person_0004/IMavatar/person_0004/eval/person_0004/epoch_752/rgb'
                elif subject == 'b0_0':
                    pred_folder = '/home/yfeng/other_github/IMavatar/data/experiments/yao/IMavatar/b0_0/eval/b0_0/epoch_1000/rgb'
                imagepath = os.path.join(pred_folder, f'{frame_id}.png')
            elif method == 'delta':
                exp_folder = self.cfg.savedir 
                imagepath = os.path.join(exp_folder, 'evaluation', 'optimize', f'{subject}_f{frame_id:06d}', f'{subject}_f{frame_id:06d}_render.jpg')
                if not os.path.exists(imagepath):
                    continue 
            
            if method == 'gt':
                pred_image = orig_image*mask + torch.ones_like(orig_image)*(1-mask)
            else:
                pred_image = imread(imagepath)/255.
                pred_image = torch.from_numpy(pred_image).permute(2,0,1).float().to(image.device)
                
            if region == 'face_neck_shoulder':
                eval_image = orig_image*mask + torch.ones_like(orig_image)*(1-mask)
            elif region == 'face':
                mask = sample['face_mask']
                eval_image = orig_image*mask + torch.ones_like(orig_image)*(1-mask)
                pred_image = pred_image*mask + torch.ones_like(pred_image)*(1-mask)
                # pred_image = pred_image*mask + orig_image*(1-mask)
                # eval_image = orig_image
            elif region == 'face_neck':
                mask = sample['face_neck_mask']
                eval_image = orig_image*mask + torch.ones_like(orig_image)*(1-mask)
                pred_image = pred_image*mask + torch.ones_like(pred_image)*(1-mask)
                
            eval_image = eval_image[None,...]
            pred_image = pred_image[None,...]
            
            test_metrics = self.evaluator(eval_image.to(self.device), pred_image.to(self.device), mask=mask[None,...].to(self.device))
            test_l1.append(test_metrics['l1'].item())
            test_psnr.append(test_metrics['psnr'].item())
            test_ssim.append(test_metrics['ssim'].item())
            test_lpips.append(test_metrics['lpips'].item())

            ## -- check eval results
            savepath = os.path.join(savefolder, f'{method}_{frame_id:06}.jpg')
            visdict = {'image': eval_image, 'pred_image': pred_image, 'off': (eval_image-pred_image).abs()}
            util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
        
        print(f"{subject} - {region} - {method} -    L1: {np.mean(test_l1)}")
        print(f"{subject} - {region} - {method} -  PSNR: {np.mean(test_psnr)}")
        print(f"{subject} - {region} - {method} -  SSIM: {np.mean(test_ssim)}")
        print(f"{subject} - {region} - {method} - LPIPS: {np.mean(test_lpips)}")
                
        # if args.model == 'delta':
        #     txtpath = os.path.join(savepath, f'{args.model}_{args.version}.txt')
        # else:
        txtpath = os.path.join(self.cfg.savedir, 'evaluation', 'comparison', f'{region}%_{method}.txt')
        with open(txtpath, 'w') as f:
            f.write(f"{subject} - {region} -    L1: {np.mean(test_l1)} - {method}\n")
            f.write(f"{subject} - {region} -  PSNR: {np.mean(test_psnr)} - {method} \n")
            f.write(f"{subject} - {region} -  SSIM: {np.mean(test_ssim)} - {method} \n")
            f.write(f"{subject} - {region} - LPIPS: {np.mean(test_lpips)} - {method}\n")
                    
    
    def combine_images(self, method_list='nha, IMavatar, delta'):
        rootdir = os.path.join(self.cfg.savedir, 'evaluation', 'comparison', 'face_neck_shoulder', 'gt')
        savefolder = os.path.join(self.cfg.savedir, 'evaluation', 'comparison', 'combined')
        os.makedirs(savefolder, exist_ok=True)
        
        imagepath_list = glob(os.path.join(rootdir, f'*_*.jpg'))
        imagepath_list = sorted(imagepath_list)
        for imagepath in imagepath_list:
            h = 512
            image0 = imread(imagepath)[:, h:2*h, :]
            image1 = imread(imagepath.replace('gt', 'nha'))[:, h:2*h, :]
            image2 = imread(imagepath.replace('gt', 'IMavatar'))[:, h:2*h, :]
            image3 = imread(imagepath.replace('gt', 'delta'))[:, h:2*h, :]            
            image = np.concatenate([image0, image1, image2, image3], axis=1)
            savepath = os.path.join(savefolder, os.path.basename(imagepath))
            imsave(savepath, image)
            
    def convert2pdf(self, method_list='nha, IMavatar, delta'):
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw 
        rootdir = os.path.join(self.cfg.savedir, 'evaluation', 'comparison', 'face_neck_shoulder', 'gt')
        imagepath_list = glob(os.path.join(rootdir, f'*_*.jpg'))
        imagepath_list = sorted(imagepath_list)
        for imagepath in imagepath_list:
            image_1 = Image.open(imagepath_list[0])
            im_1 = image_1.convert('RGB')
        
        image_list = []
        for imagepath in tqdm(imagepath_list[1:]):
            image = Image.open(imagepath).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Microsoft-Sans-Serif.ttf", 30)
            draw.text((10, 0), imagepath, (0,0,255),font=font)
            # image.save('test1.jpg')
            image_list.append(image)

        im_1.save(f'{savefolder}/{vis_type}.pdf', save_all=True, append_images=image_list)
        print(f'{savefolder}/{vis_type}.pdf')