# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import wandb
from pytorch3d.structures import Meshes

from .delta import DELTA
from .model.pose import PoseModel
from .dataset import build_dataset
from .utils.log_util import WandbLogger
from .utils import util, lossfunc, rotation_converter
from .render.mesh_helper import render_shape
from .utils.metric_util import Evaluator
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

class Trainer(torch.nn.Module):
    def __init__(self, config=None, rank=0):
        super(Trainer, self).__init__()
        self.cfg = config
        
        device = torch.device(rank)
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.batch_size = self.cfg.train.batch_size

        # -- logger
        self.logger = WandbLogger(
                 dir=self.cfg.savedir, 
                 use_wandb=self.cfg.use_wandb, 
                 project=self.cfg.project, 
                 group=self.cfg.group, 
                 name=self.cfg.exp_name, 
                 config=self.cfg, 
                 resume=self.cfg.resume)

        # -- prepare dataset
        # need to be done before building model, since we need to know the init beta value (body shape)
        self.prepare_data(data_cfg=self.cfg.dataset)
        
        # -- prepare canonical model
        init_beta = self.train_dataset[0]['beta']
        self.model = DELTA(self.cfg, init_beta=init_beta).to(self.device)
        
        # -- posemodel to optimize per frame parameters
        if self.cfg.use_posemodel:
            self.posemodel = PoseModel(dataset=self.train_dataset, **self.cfg.posemodel).to(self.device)
        # -- optimizer
        self.configure_optimizers()
        # -- load checkpoint
        self.load_checkpoint() 
        # -- init class for loss
        if self.cfg.loss.mesh_w_mrf > 0. or self.cfg.loss.w_patch_mrf:
            self.mrf_loss = lossfunc.IDMRFLoss()
        if self.cfg.loss.mesh_w_perceptual > 0. or self.cfg.loss.w_patch_perceptual:
            self.perceptual_loss = lossfunc.VGGPerceptualLoss().to(self.device)
        if self.cfg.loss.mesh_w_reg_edge > 0.:
            reg_verts = self.model.base.verts.cpu().numpy().squeeze()
            reg_faces = self.model.base.faces.cpu().numpy().squeeze()
            verts_per_edge = lossfunc.get_vertices_per_edge(len(reg_verts), reg_faces)
            self.verts_per_edge = torch.from_numpy(verts_per_edge).float().to(self.device).long()
        self.evaluator = Evaluator().to(self.device)
        
    def configure_optimizers(self):
        parameters = []
        if self.cfg.train.beta_lr > 0:
            parameters = [{'params': [self.model.beta], 'lr': self.cfg.train.beta_lr}]
        if self.cfg.use_nerf and self.cfg.train.nerf_lr > 0:
            parameters.append({'params': self.model.nerf.parameters(), 'lr': self.cfg.train.nerf_lr})
        if self.cfg.use_mesh and self.cfg.opt_mesh:
            if self.cfg.train.mesh_geo_lr > 0:
                parameters.append({'params': self.model.mesh.geo_model.parameters(), 'lr': self.cfg.train.mesh_geo_lr})
            if self.cfg.train.mesh_color_lr > 0:
                parameters.append({'params': self.model.mesh.color_model.parameters(), 'lr': self.cfg.train.mesh_color_lr})
        if self.cfg.use_posemodel:
            if self.cfg.train.pose_lr > 0:
                pose_params = [p for n, p in self.posemodel.named_parameters() if 'pose_' in n]
                parameters.append({'params': pose_params, 'lr': self.cfg.train.pose_lr})
            if self.cfg.train.cam_lr > 0:
                cam_params = [p for n, p in self.posemodel.named_parameters() if 'cam_' in n]
                parameters.append({'params': cam_params, 'lr': self.cfg.train.cam_lr})
            if self.cfg.train.exp_lr > 0:
                exp_params = [p for n, p in self.posemodel.named_parameters() if 'exp' in n]
                parameters.append({'params': exp_params, 'lr': self.cfg.train.exp_lr})
            if self.cfg.train.light_lr > 0:
                light_params = [p for n, p in self.posemodel.named_parameters() if 'light' in n]
                parameters.append({'params': light_params, 'lr': self.cfg.train.light_lr})
            
        self.optimizer = torch.optim.Adam(params=parameters, eps=1e-15)   
        max_steps = self.cfg.train.max_steps
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 5 // 6,
                max_steps * 9 // 10,
            ],
            gamma=0.33)

    def model_dict(self):
        current_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step
        }
        if self.cfg.use_posemodel:
            current_dict['posemodel'] = self.posemodel.state_dict()
        return current_dict

    def load_checkpoint(self):
        self.global_step = 0
        model_dict = self.model_dict()
        
        # resume training, including model weight, opt, steps
        if self.cfg.resume and os.path.exists(os.path.join(self.cfg.savedir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.savedir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    if isinstance(checkpoint[key], dict):
                        util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.savedir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        else:
            if os.path.exists(self.cfg.ckpt_path):
                checkpoint = torch.load(self.cfg.ckpt_path)
                for key in ['model', 'posemodel']:
                    if key in checkpoint.keys() and key in model_dict.keys():
                        if isinstance(checkpoint[key], dict):
                            util.copy_state_dict(model_dict[key], checkpoint[key])
                logger.info(f"load pretrained model from {self.cfg.ckpt_path}")
            if os.path.exists(self.cfg.nerf_ckpt_path):
                checkpoint = torch.load(self.cfg.nerf_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'nerf' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
                logger.info(f"load pretrained nerf model from {self.cfg.nerf_ckpt_path}")
            if os.path.exists(self.cfg.mesh_ckpt_path):
                checkpoint = torch.load(self.cfg.mesh_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'mesh' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
                logger.info(f"load pretrained mesh model from {self.cfg.mesh_ckpt_path}")
            if os.path.exists(self.cfg.pose_ckpt_path):
                checkpoint = torch.load(self.cfg.pose_ckpt_path)
                util.copy_state_dict(model_dict['posemodel'], checkpoint['posemodel'])
                logger.info(f"load pretrained pose model from {self.cfg.pose_ckpt_path}")
            
            ### DEBUG, load old checkpoint for nerf part
            if self.cfg.nerf.nerf_net=='mlp':
                # old path:
                # ckpt_path = '/is/cluster/yfeng/github/SCARF/exps/mpiis/DSC_7157/model.tar'
                ckpt_path = '/is/cluster/yfeng/github/SCARF/exps/snapshot/male-3-casual/model.tar'
                checkpoint = torch.load(ckpt_path)
                nerf_checkpoint = checkpoint['model']
                nerf_dict = model_dict['model']
                # nerf_checkpoint = {k.replace('nerf_fine.', ''): v for k, v in nerf_checkpoint.items() if 'nerf_fine' in k}
                # nerf_dict = {k.replace('nerf.nerf.', ''): v for k, v in nerf_dict.items() if 'nerf.nerf' in k}
                # nerf_dict.update(nerf_checkpoint)
                for param_name in nerf_dict.keys():
                    if 'nerf.nerf.' in param_name:
                        old_name = param_name.replace('nerf.nerf.', 'nerf_fine.')
                        model_dict['model'][param_name].copy_(nerf_checkpoint[old_name])
                    if 'nerf.nerf_coarse.' in param_name:
                        old_name = param_name.replace('nerf.nerf_coarse.', 'nerf.')
                        model_dict['model'][param_name].copy_(nerf_checkpoint[old_name])

    @torch.cuda.amp.autocast(enabled=False)
    def _compute_mesh_loss(self, batch, opdict):
        losses = {}
        # target_mask = batch.get('target_mask', None)
        
        #-- process gt
        if self.cfg.loss.mesh_rgb_region == 'all':
            rgb_mask = batch['mask']
        elif self.cfg.loss.mesh_rgb_region == 'skin':
            rgb_mask = batch['skin_mask']
            
        #-- photometric 
        # note: even if we use the same nerf model, we still need to compute the loss separately
        # this loss helps to backprop the gradient to the mesh vertices
        if self.cfg.loss.mesh_w_rgb > 0:
            # losses['mesh_rgb'] = lossfunc.image_loss(opdict['mesh_image'], batch['image'], 
            #                                         mask=rgb_mask, losstype=F.smooth_l1_loss)*self.cfg.loss.mesh_w_rgb
            losses['mesh_rgb'] = lossfunc.huber(rgb_mask*opdict['mesh_image'], rgb_mask*batch['image'])*self.cfg.loss.mesh_w_rgb  
        # skin consistency: inside body is occluded by the clothing, so we use the color from hands to supervise the skin
        if self.cfg.loss.mesh_w_rgb_skin > 0:
            # skin region 
            # if 'face_mask' in batch.keys():
            #     target_rgb = torch.mean(batch['image']*batch['face_mask'], 
            part_idx_dict = self.model.mesh.part_idx_dict
            target_rgb = opdict['mesh_colors'][:, part_idx_dict['face'], :].mean(dim=1).detach()
            if self.cfg.loss.skin_consistency_type == 'white':
                target_rgb = torch.ones_like(target_rgb)
            losses['mesh_skin_consistency'] = (batch['nonskin_mask']*(opdict['mesh_image'] - target_rgb[:,:,None,None])).abs().mean()*self.cfg.loss.mesh_w_rgb_skin
        # image mrf loss for details
        if self.cfg.loss.mesh_w_mrf > 0. and self.global_step > 500:
            losses['mesh_mrf'] = self.mrf_loss(opdict['mesh_image']*rgb_mask, batch['image']*rgb_mask)*self.cfg.loss.mesh_w_mrf
        # start perceptual loss after half of training, when rgb loss is small enough
        if self.cfg.loss.mesh_w_perceptual > 0. and self.global_step > 500:
            losses['mesh_perceptual'] = self.perceptual_loss(opdict['mesh_image']*rgb_mask, batch['image']*rgb_mask)*self.cfg.loss.mesh_w_perceptual
        
        # l2 mask loss
        # note: set mask region as all for body video, but with small weight
        if self.cfg.loss.mesh_w_alpha > 0.:
            if self.cfg.loss.mesh_alpha_region == 'all':
                pred_mask = opdict['mesh_mask']
            elif self.cfg.loss.mesh_alpha_region == 'skin':
                # add non skin region to pred mask
                mesh_mask = opdict['mesh_mask']
                non_skin_mask = 1-mesh_mask.detach()
                nonskin_only = non_skin_mask*batch['nonskin_mask']
                pred_mask = mesh_mask + nonskin_only
            # losses['mesh_alpha'] = lossfunc.image_loss(pred_mask, batch['mask'], 
            #                                         mask=None, losstype=F.smooth_l1_loss)*self.cfg.loss.mesh_w_alpha
            losses['mesh_mask'] = lossfunc.huber(pred_mask, batch['mask'])*self.cfg.loss.mesh_w_alpha
        # for body: let mesh close to the full mask, should be small 
        if self.cfg.loss.mesh_w_alpha_all > 0.:
            losses['mesh_alpha_all'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.cfg.loss.mesh_w_alpha_all
        if self.cfg.loss.mesh_w_alpha_skin > 0.:
            losses['mesh_alpha_skin'] =lossfunc.huber(batch['skin_mask']*opdict['mesh_mask'], batch['skin_mask'])*self.cfg.loss.mesh_w_alpha_skin
        # mesh inside mask loss
        if self.cfg.loss.mesh_w_alpha_inside > 0.:
            losses['mesh_alpha_inside'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.cfg.loss.mesh_w_alpha_inside
            
        # mesh 
        #-- geometry regularizations
        if self.cfg.loss.mesh_w_reg_offset > 0.:
            losses["reg_offset"] = (opdict['mesh_offset']**2).sum(-1).mean()*self.cfg.loss.mesh_w_reg_offset
            ## for face and hand
            face_idx = self.model.mesh.part_idx_dict['face']
            hand_idx = self.model.mesh.part_idx_dict['hand']
            losses["reg_offset_face"] = (opdict['mesh_offset'][:,face_idx]**2).sum(-1).mean()*self.cfg.loss.mesh_w_reg_offset
            losses["reg_offset_hand"] = (opdict['mesh_offset'][:,hand_idx]**2).sum(-1).mean()*self.cfg.loss.mesh_w_reg_offset*10

        if self.cfg.loss.mesh_w_reg_edge > 0.:
            offset = opdict['mesh_offset']
            base_verts = self.model.base.verts[None].expand(offset.shape[0], -1, -1)
            new_verts = base_verts + offset
            losses["reg_edge"] = lossfunc.relative_edge_loss(new_verts, 
                                                             base_verts,
                                                             vertices_per_edge=self.verts_per_edge)*self.cfg.loss.mesh_w_reg_edge
        if self.cfg.loss.mesh_w_reg_laplacian > 0.:
            offset = opdict['mesh_offset']
            base_verts = self.model.base.verts[None].expand(offset.shape[0], -1, -1)
            new_verts = base_verts + offset
            mesh = Meshes(verts=new_verts, faces=opdict['mesh_faces'])
            losses["reg_laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")*self.cfg.loss.mesh_w_reg_laplacian

        # offset loss
        return losses
    
    def _compute_nerf_loss(self, batch, opdict):
        losses = {}
        #-- sample corresponding gts from coords
        coords = opdict['coords']
        gt_rgbs = batch['image'].permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
        if self.cfg.loss.alpha_region == 'all':
            mask = batch['mask']
        elif self.cfg.loss.alpha_region == 'nonskin':
            mask = batch['nonskin_mask']
        gt_alphas = mask.permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
        object_alphas = batch['mask'].permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
        #-- compute nerf loss
        # l1 image loss
        # losses['nerf_rgb'] = F.smooth_l1_loss(opdict['rgbs'], gt_rgbs)*self.cfg.loss.w_rgb
        losses['nerf_rgb'] = lossfunc.huber(opdict['rgbs']*object_alphas, gt_rgbs*object_alphas)*self.cfg.loss.w_rgb
        if 'rgbs_coarse' in opdict:
            # losses['nerf_rgb_coarse'] = F.smooth_l1_loss(opdict['rgbs_coarse'], gt_rgbs)*self.cfg.loss.w_rgb
            losses['nerf_rgb_coarse'] = lossfunc.huber(opdict['rgbs_coarse']*object_alphas, gt_rgbs*object_alphas)*self.cfg.loss.w_rgb
        # if self.cfg.loss.w_rgb_nerf > 0. and self.cfg.loss.w_alpha > 0.:
        #     losses['nerf_rgb_nerfregion'] = F.smooth_l1_loss(opdict['rgbs']*gt_alphas, gt_rgbs*gt_alphas)*self.cfg.loss.w_rgb_nerf
        # percetual loss
        if self.cfg.loss.w_patch_perceptual > 0.:
            assert self.cfg.nerf.n_rays_patch_size > 0, "perceptual loss requires sampling rays in patches, specify patch size, suggested number:32,64,128"
            # resize to patch
            ps = self.cfg.nerf.n_rays_patch_size
            gt_rgbs_patch = gt_rgbs[:,:ps**2].reshape(-1, self.cfg.nerf.n_rays_patch_size, self.cfg.nerf.n_rays_patch_size, 3).permute(0,3,1,2)
            pred_rgbs_patch = opdict['rgbs'][:,:ps**2].reshape(-1, self.cfg.nerf.n_rays_patch_size, self.cfg.nerf.n_rays_patch_size, 3).permute(0,3,1,2)
            losses['nerf_patch_perceptual'] = self.perceptual_loss(pred_rgbs_patch, gt_rgbs_patch)*self.cfg.loss.w_patch_perceptual
        # mask loss
        if self.cfg.loss.w_alpha > 0.:
            # losses['nerf_alpha'] = F.smooth_l1_loss(opdict['alphas'], gt_alphas)*self.cfg.loss.w_alpha
            losses['nerf_alpha'] = lossfunc.huber(opdict['alphas'], gt_alphas)*self.cfg.loss.w_alpha
            if 'alphas_coarse' in opdict:
                # losses['nerf_alpha_coarse'] = F.smooth_l1_loss(opdict['alphas_coarse'], gt_alphas)*self.cfg.loss.w_alpha
                losses['nerf_alpha_coarse'] = lossfunc.huber(opdict['alphas_coarse'], gt_alphas)*self.cfg.loss.w_alpha
            # gt_skin_alphas = batch['skin_mask'].permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
            # losses['nerf_alpha_black'] = (opdict['alphas']*gt_skin_alphas).abs().mean()*self.cfg.loss.w_alpha
        # regularization on normal
        if self.cfg.loss.w_reg_normal > 0.:
            points_normal, points_neighbs_normal = self.model.nerf.canonical_normal()
            losses['nerf_reg_normal'] = F.mse_loss(points_normal, points_neighbs_normal)*self.cfg.loss.w_reg_normal
            if self.model.nerf.nerf_coarse is not None:
                points_normal, points_neighbs_normal = self.model.nerf.canonical_normal(use_coarse=True)
                losses['nerf_reg_normal_coarse'] = F.mse_loss(points_normal, points_neighbs_normal)*self.cfg.loss.w_reg_normal
        if self.cfg.loss.w_hard > 0.:
            weights = opdict['weights']
            losses['nerf_hard'] = lossfunc.hard_loss_func(weights, scale=self.cfg.loss.w_hard_scale)*self.cfg.loss.w_hard
        if self.cfg.nerf.deform_cond == 'posed_verts' and self.cfg.loss.w_reg_correction > 0.:
            correction = opdict['correction']
            losses['nerf_reg_correction'] = correction.abs().mean()*self.cfg.loss.w_reg_correction
        return losses
        
    def training_step(self, batch):
        self.model.train()
        # move data to device
        util.move_dict_to_device(batch, self.device)
        
        #-- update per frame parameters (cam, pose, exp, etc) if using posemodel
        if self.cfg.use_posemodel:
            batch = self.posemodel(batch)
        #-- forward DELTA model, get output to compute losses
        opdict = self.model(batch)
        
        #-- loss
        #### ----------------------- Losses
        losses = {}
        if self.cfg.use_mesh and self.cfg.opt_mesh:
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
        return losses, opdict

    @torch.no_grad()
    def prepare_data(self, data_cfg):
        logger.info('Preparing data...')
        self.train_dataset = build_dataset.build_train(data_cfg, mode='train')
        self.val_dataset = build_dataset.build_train(data_cfg, mode='val')
        logger.info(f"load data from {self.train_dataset.dataset_path}, \
                    training frame numbers: {len(self.train_dataset)}, \
                    validation frame numbers: {len(self.val_dataset)}")
        self.train_dataloader = DataLoader(self.train_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=True,
                            # num_workers=min(self.batch_size, 8),
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)

    @torch.no_grad()
    def validation_step(self, data='train', batch=None, returnVis=False, novel_view=False):
        self.model.eval()
        if batch is None:
            if data == 'val':
                val_iter = iter(self.val_dataloader)
                batch = next(val_iter)
            else:
                val_iter = iter(self.train_dataloader)
                batch = next(val_iter)
            util.move_dict_to_device(batch, self.device)
            # for training data, update pose
            # if data == 'train' and self.cfg.use_posemodel:
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            if novel_view and data == 'val':
                data = 'val_novel_view'
                yaw_list = [-70, -30, 30, 70]
                for key in batch.keys():
                    if not torch.is_tensor(batch[key]):
                        continue
                    if key == 'full_pose':
                        for i in range(1, batch[key].shape[0]):
                            euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                            euler_pose[:,1] = yaw_list[i]
                            global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                            pose = batch['full_pose'][0:1].clone()
                            if 'face' in self.cfg.dataset.type:
                                pose[:,12,:,:] = torch.matmul(pose[:,12,:,:], global_pose)
                            elif 'body' in self.cfg.dataset.type:
                                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                            batch['full_pose'][i:i+1] = pose
                    else:
                        for i in range(1, batch[key].shape[0]):
                            batch[key][i] = batch[key][0]
                                    
        batch['global_step'] = self.global_step
        opdict = self.model(batch, render_image=True, render_normal=self.cfg.nerf.render_normal)
        visdict = {}
        datadict = {**batch, **opdict}
        for key in datadict.keys():
            if 'path' in key or 'depth' in key or 'vis' in key or 'sampled' in key:
                continue
            if 'image' in key:
                if 'normal' in key:
                    visdict[key] = datadict[key]*0.5 + 0.5
                else:
                    visdict[key] = datadict[key]
            if data=='train' and 'mask' in key:
                visdict[key] = datadict[key].expand(-1, 3, -1, -1)        
        # render shape for visualization
        # if 'mesh_trans_verts' and 'mesh_faces' in opdict:
        #     shape_image = render_shape(opdict['mesh_trans_verts'], 
        #                                opdict['mesh_faces'], 
        #                                image_size=self.image_size, 
        #                                background=batch.get('image', None))
        #     visdict['shape_image'] = shape_image
        
        # if use pose model, show original pose
        # if self.cfg.use_posemodel: # and data=='train': # and self.cfg.use_mesh:
        batch_size = batch['full_pose'].shape[0]
        rendering = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
                                batch['full_pose'], 
                                batch['cam'], 
                                batch.get('exp', None),
                                renderShape=True,
                                background=batch.get('image', None))
        visdict['refined_pose_image'] = rendering['shape_image']
        if 'init_full_pose' in batch:
            rendering = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
                                    batch['init_full_pose'], 
                                    batch['init_cam'], 
                                    batch.get('init_exp', None),
                                    renderShape=True,
                                    background=batch.get('image', None))
            visdict['init_pose_image'] = rendering['shape_image']
        self.logger.log_image(visdict, self.global_step, mode=data)
        
        # run metrics for val data
        if data == 'val':
            gt = batch['image']
            if self.cfg.use_nerf:
                pred = opdict['nerf_image']
            else:
                pred = opdict['mesh_image']
            val_metrics = self.evaluator(pred, gt)
            self.logger.log_metrics(val_metrics, self.global_step)
        
    def fit(self):
        if self.cfg.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        # iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        # start_epoch = self.global_step//iters_every_epoch
        # for epoch in tqdm(range(start_epoch, self.cfg.train.max_epochs)):
        # for step in range(self.cfg.train.max_steps):
        for step in tqdm(range(self.cfg.train.max_steps), desc=f"training steps"):
            # for step in range(iters_every_epoch):
            if step < self.global_step:
                continue
            #-- validation
            if self.global_step % self.cfg.train.val_steps == 0:
                with torch.cuda.amp.autocast():
                    self.validation_step(data='train')
                    self.validation_step(data='val')
                    self.validation_step(data='val', novel_view=True)
                    
            #-- training
            self.optimizer.zero_grad()
            try:
                batch = next(self.train_iter)
            except:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)
            batch['global_step'] = self.global_step
            
            if self.cfg.mixed_precision:
                with torch.cuda.amp.autocast():
                    losses, opdict = self.training_step(batch)
            else:
                losses, opdict = self.training_step(batch)
            all_loss = losses['all_loss']

            if self.cfg.mixed_precision:
                self.scaler.scale(all_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                all_loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            
            ### logger
            if self.global_step % self.cfg.train.log_steps == 0:
                self.logger.log_loss(losses, self.global_step)    
                
            if self.global_step % self.cfg.train.checkpoint_steps == 0:
                torch.save(self.model_dict(), os.path.join(self.cfg.savedir, 'model.tar'))   
                print(f"save model at {self.global_step} steps")
            
            self.global_step += 1
            if self.global_step > self.cfg.train.max_steps:
                logger.info(f"Reach max steps {self.cfg.train.max_steps}, stop training")
                if self.cfg.mixed_precision:
                    with torch.cuda.amp.autocast():
                        self.validation_step(data='train')
                        self.validation_step(data='val')
                        self.validation_step(data='val', novel_view=True)
                else:
                    self.validation_step(data='train')
                    self.validation_step(data='val')
                    self.validation_step(data='val', novel_view=True)
                exit()
            