import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
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

def mcubes_to_world(vertices, N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    vertices_ = vertices / N
    x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    return vertices_

def create_grid(N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    grid = np.stack(np.meshgrid(x, y, z), -1)
    return grid

def load_pixie_smplx(actions_file='/home/yfeng/github/SCARF/data/pixie_radioactive.pkl',
                actions_folder='/home/yfeng/Data/Projects-data/DHA/datasets/processed_data/dha_face/person_0004/smplx_all'):
    # load pixie animation poses
    if actions_file is None or not os.path.exists(actions_file) and os.path.isdir(actions_folder):
        actions_file = convert2pkl(actions_file, actions_folder)
    
    with open(actions_file, 'rb') as f:
        codedict = pickle.load(f)
    full_pose = torch.from_numpy(codedict['full_pose'])
    cam = torch.from_numpy(codedict['cam'])
    # cam[:,0] = cam[:,0] * 1.15
    # cam[:,2] = cam[:,2] + 0.3
    exp = torch.from_numpy(codedict['exp'])
    return full_pose, exp, cam

def convert2pkl(actions_file='/home/yfeng/github/SCARF/data/pixie_radioactive.pkl',
                actions_folder='/home/yfeng/Data/Projects-data/DHA/datasets/processed_data/dha_face/person_0004/smplx_all',
                # valid_range=[96, 136]):
                valid_range=[100, 150]):
    # convert to pkl
    folder = actions_folder
    action_list = glob(os.path.join(folder, '*.pkl'))
    action_list = sorted(action_list)[valid_range[0]:valid_range[1]]
    full_pose_list = []
    cam_list = []
    exp_list = []
    for action_file in action_list:
        with open(action_file, 'rb') as f:
            codedict = pickle.load(f)
        full_pose = torch.from_numpy(codedict['full_pose'])
        cam = torch.from_numpy(codedict['cam'])
        exp = torch.from_numpy(codedict['exp'])
        full_pose_list.append(full_pose)
        cam_list.append(cam)
        exp_list.append(exp)
    
    full_pose = torch.stack(full_pose_list, 0)
    cam = torch.stack(cam_list, 0)
    exp = torch.stack(exp_list, 0)
    codedict = {'full_pose': full_pose.cpu().numpy(), 'cam': cam.cpu().numpy(), 
                'exp': exp.cpu().numpy()}
    # import ipdb; ipdb.set_trace()
    if actions_file is None:
        actions_file = os.path.join('animation_actions.pkl')
    with open(actions_file, 'wb') as f:
        pickle.dump(codedict, f)
    return actions_file
class Visualizer(Trainer):
    def __init__(self, config=None):
        super(Visualizer, self).__init__(config=config)

    def save_video(self, savepath, image_list, fps=10):
        video_type = savepath.split('.')[-1]
        if video_type == 'mp4' or video_type == 'gif':
            import imageio
            if video_type == 'mp4':
                writer = imageio.get_writer(savepath, mode='I', fps=fps)
            elif video_type == 'gif':
                writer = imageio.get_writer(savepath, mode='I', duration=1/fps)
            for image in image_list:
                writer.append_data(image[:,:,::-1])
            writer.close()
            logger.info(f'{video_type} saving to {savepath}')
        
    @torch.no_grad()
    def capture(self, savefolder, saveImages=False, video_type='mp4', fps=10):
        """ show color and hybrid rendering of training frames

        Args:
            savefolder (_type_): _description_
        """
        # load data
        self.cfg.dataset.train.frame_step *= 2
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, self.device)
            frame_id = batch['frame_id'][0]
            visdict = {'image': batch['image']}
            # visdict = {}
            # run model
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            opdict = self.model.forward_vis(batch)
            visdict.update(opdict)
            # visualization
            # visdict['render'] = opdict['nerf_image']
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            image_list.append(grid_image)
            print(f'saving to {savepath}')
            if saveImages:
                os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}'), exist_ok=True)
                for key in visdict.keys():
                    image = visdict[key]
                    cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
        videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_capture.{video_type}')
        self.save_video(videopath, image_list, fps=fps)
            
    @torch.no_grad()
    def novel_view(self, savefolder, frame_id=0, saveImages=False, video_type='mp4', fps=10, max_yaw=90):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        # self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = self.cfg.dataset.train.frame_start + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            # batch['cam'][:,0] *= 0.5
            # batch['cam'][:,1] 
            # change the global pose (pelvis) for novel view
            # yaws = np.arange(0, 361, 10)
            yaws = list(np.arange(0, -max_yaw-1, -10)) + list(np.arange(-max_yaw, max_yaw+1, 10)) + list(np.arange(max_yaw, 0, -10)) 
            init_pose = batch['full_pose']
            for yaw in tqdm(yaws):
                euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                euler_pose[:,1] = yaw
                global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                pose = init_pose.clone()
                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                batch['full_pose'] = pose

                opdict = self.model.forward_vis(batch)
                # visualization
                visdict = opdict
                # visdict['render'] = opdict['nerf_fine_image']
                # visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view.{video_type}')
            self.save_video(videopath, image_list, fps=fps)
    
    @torch.no_grad()
    def change_shape(self, savefolder, frame_id=0, saveImages=False, video_type='mp4', fps=10):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        # self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = self.cfg.dataset.train.frame_start + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            # batch['cam'][:,0] *= 0.5
            # batch['cam'][:,1] 
            # change the shape scale
            shape_scales = list(np.arange(1.0, 0.1, -0.1)) + list(np.arange(0.1, 1.3, 0.1))  + list(np.arange(1.3, 1, -0.1))
            init_beta = batch['beta'].clone()
            model_dict = self.model_dict()
            for shape_scale in tqdm(shape_scales):
                new_beta = init_beta * shape_scale * shape_scale
                model_dict['model']['beta'].copy_(new_beta)
                opdict = self.model.forward_vis(batch)
                # visualization
                visdict = opdict
                # visdict['render'] = opdict['nerf_fine_image']
                # visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{shape_scale:.1}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{shape_scale:.1}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{shape_scale:.1}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_change_shape.{video_type}')
            self.save_video(videopath, image_list, fps=fps)
    
    @torch.no_grad()
    def novel_view_with_hair(self, savefolder, hair_visualizer=None, 
                             frame_id=0, saveImages=False, video_type='mp4', fps=10, body_mesh=None, 
                             use_halfcam=True, use_bodypose=True):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = frame_id + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        #### 
        if body_mesh is not None:
            from .utils.deca_render import SRenderY
            self.deca_render = SRenderY(image_size=512, 
                                   obj_filename=self.cfg.model.topology_smplxtex_path, 
                                   uv_size=self.cfg.model.uv_size, rasterizer_type='pytorch3d', use_uv=True, raster_settings=None)
            self.deca_render.to(self.device)
            # import ipdb; ipdb.set_trace()
            
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            # if self.cfg.use_posemodel:
            #     batch = self.posemodel(batch)
                
            # change the global pose (pelvis) for novel view
            yaws = np.arange(0, 361, 20)
            if use_bodypose:
                init_pose = body_mesh['full_pose']
                # spin pose
                # init_pose[:,[3,6,9]] = batch['full_pose'][:,[3,6,9]]
                init_pose[:,[1,2,3,6,9]] = batch['full_pose'][:,[1,2,3,6,9]]
            else:
                init_pose = batch['full_pose']
            if use_halfcam:
                batch['cam'][:,0] = batch['cam'][:,0]*1.5
                batch['cam'][:,2] = batch['cam'][:,2]+0.72
            batch['exp'] = body_mesh['exp']
            
            for yaw in tqdm(yaws):
                euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                euler_pose[:,1] = yaw
                global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                pose = init_pose.clone()
                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                batch['full_pose'] = pose
                opdict = self.model.forward_vis(batch, returnMask=True)
                # visualization
                # visdict['render_clothing'] = opdict['render']
                # visdict['render_clothing_hybrid'] = opdict['render_hybrid']
                clothing_mask = opdict['nerf_mask']
                
                ### hair visualization
                hair_opdict = hair_visualizer.model.forward_vis(batch, returnMask=True)
                # visdict['render_hair'] = hair_opdict['render']
                # visdict['render_hair_hybrid'] = hair_opdict['render_hybrid']
                hair_mask = hair_opdict['nerf_mask']
                
                ### combine, copy clothing to face/hair rendering
                #render body
                # import ipdb; ipdb.set_trace()
                mesh_rendering = self.model.mesh(self.model.beta, 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderShape=True
                                    )
                trans_verts = mesh_rendering['mesh_trans_verts']
                trans_verts[...,-1] = trans_verts[...,-1]+20
                rendering = self.deca_render(mesh_rendering['mesh_posed_verts'], trans_verts, albedos=body_mesh['albedo'])
                albedo_images = rendering['albedo_images']*rendering['alpha_images'] + 1-rendering['alpha_images']
                # import ipdb; ipdb.set_trace()
                visdict['body'] = albedo_images
                visdict['render_clothing'] = visdict['body']*(1-clothing_mask) + opdict['render']*clothing_mask 
                visdict['render_combine'] = visdict['render_clothing']*(1-hair_mask) + hair_opdict['render_hybrid']*hair_mask
                # visdict['hair_mask'] = hair_mask.expand(-1,3,-1,-1)
                visdict['shape_combine'] = opdict['render_hybrid']*(1-hair_mask) + hair_opdict['render']*hair_mask

                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                
                # exit()
                
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view_combine.{video_type}')
            self.save_video(videopath, image_list, fps=fps)

    @torch.no_grad()
    def capture_with_hair(self, savefolder, hair_visualizer=None, 
                             frame_id=0, saveImages=False, video_type='mp4', fps=10, body_mesh=None):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        self.cfg.dataset.train.frame_step = 2
        self.train_dataset = build_dataset.build_train(hair_visualizer.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        #### 
        if body_mesh is not None:
            from .utils.deca_render import SRenderY
            self.deca_render = SRenderY(image_size=512, 
                                   obj_filename=self.cfg.model.topology_smplxtex_path, 
                                   uv_size=self.cfg.model.uv_size, rasterizer_type='pytorch3d', use_uv=True, raster_settings=None)
            self.deca_render.to(self.device)
            # import ipdb; ipdb.set_trace()
            
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if hair_visualizer.cfg.use_posemodel:
                batch = hair_visualizer.posemodel(batch)
            batch['exp'] = body_mesh['exp']
            batch['full_pose'][:,22] = body_mesh['full_pose'][:,22]
            opdict = self.model.forward_vis(batch, returnMask=True)
                # visualization
                # visdict['render_clothing'] = opdict['render']
                # visdict['render_clothing_hybrid'] = opdict['render_hybrid']
            clothing_mask = opdict['nerf_mask']
            
            ### hair visualization
            hair_opdict = hair_visualizer.model.forward_vis(batch, returnMask=True)
            # visdict['render_hair'] = hair_opdict['render']
            # visdict['render_hair_hybrid'] = hair_opdict['render_hybrid']
            hair_mask = hair_opdict['nerf_mask']
            
            ### combine, copy clothing to face/hair rendering
            #render body
            # import ipdb; ipdb.set_trace()
            mesh_rendering = self.model.mesh(self.model.beta, 
                                batch['full_pose'], 
                                batch['cam'], 
                                batch.get('exp', None),
                                renderShape=True
                                )
            trans_verts = mesh_rendering['mesh_trans_verts']
            trans_verts[...,-1] = trans_verts[...,-1]+20
            rendering = self.deca_render(mesh_rendering['mesh_posed_verts'], trans_verts, albedos=body_mesh['albedo'])
            albedo_images = rendering['albedo_images']*rendering['alpha_images'] + 1-rendering['alpha_images']
            # import ipdb; ipdb.set_trace()
            visdict['body'] = albedo_images
            visdict['render_clothing'] = visdict['body']*(1-clothing_mask) + opdict['render']*clothing_mask 
            visdict['render_combine'] = visdict['render_clothing']*(1-hair_mask) + hair_opdict['render']*hair_mask
            # visdict['hair_mask'] = hair_mask.expand(-1,3,-1,-1)
            visdict['shape_combine'] = opdict['render_hybrid']*(1-hair_mask) + hair_opdict['render']*hair_mask

            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            image_list.append(grid_image)
            print(f'saving to {savepath}')
            
            # exit()
            if saveImages:
                os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}'), exist_ok=True)
                for key in visdict.keys():
                    image = visdict[key]
                    cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
        videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view_combine.{video_type}')
        self.save_video(videopath, image_list, fps=fps)

      
    @torch.no_grad()
    def extract_mesh(self, savefolder, frame_id=0):
        logger.info(f'extracting mesh from frame {frame_id}')
        # load data
        self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = frame_id + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            visdict = {'image': batch['image']}
            
            # extract mesh
            if self.cfg.use_nerf:
                N_grid = 256
                sigma_threshold = 10.
                near = self.model.nerf.render.near
                far = self.model.nerf.render.far
                ## points from rays, rays, xyz_fine, viewdir, z_vals
                x_range = [-1, 1]
                y_range = [-1, 1]
                z_range = [near, far]
                grid = create_grid(N_grid, x_range, y_range, z_range)
                xyz = torch.from_numpy(grid.reshape(-1, 3)).unsqueeze(0).float().to(self.device) # (1, N*N*N, 3)

                sigmas = []
                chunk = 32*32*4*2
                mesh_out = self.model(batch, run_mesh=True, run_nerf=False)
                batch.update(mesh_out)
                for i in tqdm(range(0, xyz.shape[1], chunk)):
                    xyz_chunk = xyz[:, i:i+chunk, :]
                    dir_chunk = torch.zeros_like(xyz_chunk)
                    ## run nerf model
                    _, sigma_chunk = self.model.nerf(xyz_chunk, batch)
                    # import ipdb; ipdb.set_trace()
                    # xyz_chunk, dir_chunk, _ = self.model.backward_skinning(batch, xyz_chunk, dir_chunk)
                    # ## query canonical model
                    # _, sigma_chunk = self.model.query_canonical_space(xyz_chunk, dir_chunk, use_fine=True)
                    sigmas.append(torch.relu(sigma_chunk))
                sigmas = torch.cat(sigmas, 1)

                sigmas = sigmas.cpu().numpy()
                sigmas = np.maximum(sigmas, 0).reshape(N_grid, N_grid, N_grid)
                sigmas = sigmas - sigma_threshold
                # smooth
                sigmas = mcubes.smooth(sigmas)
                # sigmas = mcubes.smooth(sigmas)
                vertices, faces = mcubes.marching_cubes(-sigmas, 0.)
                vertices = mcubes_to_world(vertices, N_grid, x_range, y_range, z_range)
                cloth_verts = vertices
                cloth_faces = faces
                
            ### add body shape
            if self.cfg.use_mesh:
                mesh_out = self.model(batch, run_mesh=True, run_nerf=False)
                body_verts = mesh_out['mesh_trans_verts'].cpu().numpy().squeeze()
                body_faces = mesh_out['mesh_faces'].cpu().numpy().squeeze()
                if self.cfg.use_nerf:
                    # combine two mesh
                    faces = np.concatenate([faces, body_faces+vertices.shape[0]]).astype(np.int32)
                    vertices = np.concatenate([vertices, body_verts], axis=0).astype(np.float32)
                else:
                    faces = body_faces
                    vertices = body_verts
                    
            ### visualize
            batch_size = 1
            faces = torch.from_numpy(faces.astype(np.int32)).long().to(self.device)[None,...]
            vertices = torch.from_numpy(vertices).float().to(self.device)[None,...]
            if self.cfg.use_mesh and self.cfg.use_nerf:
                colors = torch.ones_like(vertices)*180/255.
                colors[:,:cloth_verts.shape[0], [0,2]] = 180/255.
                colors[:,:cloth_verts.shape[0], 1] = 220/255.
            else:
                colors = torch.ones_like(vertices)*180/255.
            # import ipdb; ipdb.set_trace()
            shape_image = render_shape(vertices = vertices, faces = faces, 
                                    image_size=512, blur_radius=1e-8,
                                    colors=colors)
                                    # background=batch['image'])
            visdict['shape_image'] = shape_image
                
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_vis.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            # save obj
            util.write_obj(os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_all.obj'), 
                           vertices[0].cpu().numpy(), 
                           faces[0].cpu().numpy()[:,[0,2,1]],
                           colors=colors[0].cpu().numpy())
            if self.cfg.use_mesh:
                util.write_obj(os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_body.obj'), 
                            body_verts, 
                            body_faces[:, [0,2,1]],
                            colors=colors[0, cloth_verts.shape[0]:].cpu().numpy())
            if self.cfg.use_nerf:
                util.write_obj(os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_cloth.obj'), 
                            cloth_verts, 
                            cloth_faces[:, [0,2,1]],
                            colors=colors[0, :cloth_verts.shape[0]].cpu().numpy())
        logger.info(f'Visualize results saved to {savefolder}')
    
    
    @torch.no_grad()
    def animate(self, savefolder, animation_file=None, 
                animation_folder=None,
                saveImages=False, video_type='mp4', fps=10):
        # load animation poses        
        full_pose, exp, cam = load_pixie_smplx(
            # actions_file='/is/cluster/yfeng/Data/Projects-data/DELTA/datasets/face/person_0000/animation.pkl',
            actions_file=None,
            actions_folder='/is/cluster/yfeng/Data/Projects-data/DELTA/datasets/face/person_0004/smplx_all')
        cam = cam.to(self.device)
        full_pose = full_pose.to(self.device)
        exp = exp.to(self.device)
        image_list = []
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            curr_cam = batch['cam']
            break
        
        for i in tqdm(range(full_pose.shape[0])):
            batch = {'full_pose': full_pose[i:i+1], 
                     'exp': exp[i:i+1], 
                     'cam': cam[i:i+1]}
            # batch['cam'] = curr_cam.to(self.device)
            util.move_dict_to_device(batch, self.device)
            visdict = {}
            # run model
            opdict = self.model.forward_vis(batch)
            visdict.update(opdict)
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_animation_{i:03}.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            image_list.append(grid_image)
            print(savepath)
        video_path = os.path.join(savefolder, f'{self.cfg.exp_name}_animation.{video_type}')
        self.save_video(video_path, image_list, fps=fps)
        
    @torch.no_grad()
    def run(self, vistype, args=None):
        # check if body or clothing model are specified
        model_dict = self.model_dict()
        if os.path.exists(args.body_model_path):
            body_name = args.body_model_path.split('/')[-2]
            savefolder = os.path.join(self.cfg.savedir, f'visualization_body_{body_name}', vistype)            
            if os.path.exists(args.body_model_path):
                checkpoint = torch.load(args.body_model_path)
                for param_name in model_dict['model'].keys():
                    if 'mesh' in param_name or 'beta' in param_name:
                        if param_name in checkpoint['model']:
                            model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
        elif os.path.exists(args.clothing_model_path):
            clothing_name = args.clothing_model_path.split('/')[-2]
            savefolder = os.path.join(self.cfg.savedir, f'visualization_clothing_{clothing_name}', vistype)
            if os.path.exists(args.clothing_model_path):
                checkpoint = torch.load(args.clothing_model_path)
                for param_name in model_dict['model'].keys():
                    if 'nerf' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
        else:
            savefolder = os.path.join(self.cfg.savedir, 'visualization', vistype)
        
        if args.shape_scale != 0.:
            current_beta = model_dict['model']['beta'].clone()
            new_beta = current_beta * args.shape_scale
            model_dict['model']['beta'].copy_(new_beta)
            
        os.makedirs(savefolder, exist_ok=True)
        if vistype == 'capture':
            self.capture(savefolder, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'novel_view':
            self.novel_view(savefolder, frame_id=args.frame_id, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps, max_yaw=args.max_yaw)
        elif vistype == 'extract_mesh':
            self.extract_mesh(savefolder, frame_id=args.frame_id)
        elif vistype == 'animate':
            self.animate(savefolder, animation_file=args.animation_file, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'change_shape':
            self.change_shape(savefolder, frame_id=args.frame_id, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'inpaint':
            self.inpaint(savefolder, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'pipeline':
            self.pipeline(savefolder, frame_id=args.frame_id)
        elif vistype == 'relight':
            self.relight(savefolder, frame_id=args.frame_id)
            
    @torch.no_grad()
    def relight(self, savefolder, frame_id=0, saveImages=False, video_type='mp4', fps=10):
        """ show relighting results of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        # self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = self.cfg.dataset.train.frame_start + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            # batch['cam'][:,0] *= 0.5
            # batch['cam'][:,1] 
            # change the global pose (pelvis) for novel view
            yaws = np.arange(0, 361, 10)
            init_pose = batch['full_pose']
            for yaw in tqdm(yaws):
                euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                euler_pose[:,1] = yaw
                global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                pose = init_pose.clone()
                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                batch['full_pose'] = pose

                full_opdict = self.model.forward(batch, run_mesh=True, run_nerf=True, 
                                        render_image=True, render_normal=False, render_shape=True)
                # import ipdb; ipdb.set_trace()
                nerf_mask = full_opdict['nerf_mask']
                nerf_image = full_opdict['nerf_image']
                mesh_mask = full_opdict['mesh_vis_image']
                mesh_image = full_opdict['mesh_image']
                mesh_albedo_image = full_opdict['mesh_albedo_image']
                shape_image = full_opdict['shape_image']

                visdict = {
                    'render': nerf_image,
                    'render_albedo': nerf_image * nerf_mask + mesh_albedo_image * (1 - nerf_mask),
                    'render_hybrid': nerf_image * nerf_mask + shape_image * (1 - nerf_mask)
                }
                # visdict['render'] = full_opdict['nerf_fine_image']
                
                # opdict = self.model.forward_vis(batch)
                # # visualization
                # visdict = opdict
                # visdict['render'] = opdict['nerf_fine_image']
                # visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view.{video_type}')
            self.save_video(videopath, image_list, fps=fps)
    
    @torch.no_grad()
    def inpaint(self, savefolder, frame_id=0, saveImages=False, video_type='mp4', fps=10):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        ### load diffusion models
        sys.path.insert(0, '/home/yfeng/other_github/TEXTurePaper')
        from src.stable_diffusion_depth import StableDiffusion
        diffusion_model = StableDiffusion(self.device, model_name='stabilityai/stable-diffusion-2-depth',
                                          concept_name=None,
                                          concept_path=None,
                                          latent_mode=False,
                                          min_timestep=0.02,
                                          max_timestep=0.98,
                                          no_noise=False,
                                          use_inpaint=True)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        self.diffusion = diffusion_model.to(self.device)
        
        # load data
        # self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = self.cfg.dataset.train.frame_start + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if self.cfg.use_posemodel:
                batch = self.posemodel(batch)
            # batch['cam'][:,0] *= 0.5
            # batch['cam'][:,1] 
            # change the global pose (pelvis) for novel view
            yaws = np.arange(0, 361, 10)
            init_pose = batch['full_pose']
            for yaw in tqdm(yaws):
                euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                euler_pose[:,1] = yaw
                global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                pose = init_pose.clone()
                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                batch['full_pose'] = pose

                opdict = self.model.forward_vis(batch)
                # visualization
                visdict = opdict
                ## depth
                mesh_rendering = self.model.mesh(self.model.beta.repeat(1, 1), 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderDepth=True
                                    )
                mesh_depth_image = -mesh_rendering['mesh_depth_image']
                mesh_depth_image = mesh_depth_image - mesh_depth_image.min()
                mesh_depth_image = mesh_depth_image / mesh_depth_image.max()
                mesh_vis_mask = mesh_rendering['mesh_vis_image'][:,None,:,:]
                visdict['mesh_depth_image'] = mesh_depth_image[:,None,:,:].repeat(1,3,1,1)
                depth_mask = mesh_vis_mask
                #### inpainting
                # text embedding
                text = "a photo of a bald caucasian man, front view"
                text_z = self.diffusion.get_text_embeds([text], negative_prompt=None).to(self.device)
                # image input from mesh -> croppend_rgb_render, cropped_depth_render, update mask
                
                rgb_render = opdict['render']
                keep_mask = batch['skin_mask']
                nonskin_mask = batch['nonskin_mask']
                ## erode keep mask
                keep_mask = torch.from_numpy(
                    cv2.erode(keep_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
                    keep_mask.device).unsqueeze(0).unsqueeze(0)

                cropped_rgb_render = (rgb_render*keep_mask)
                cropped_depth_render = mesh_depth_image[:,None,:,:]*depth_mask
                # cropped_update_mask = depth_mask - keep_mask #1 - keep_mask
                cropped_update_mask = 1 - keep_mask #1 - keep_mask
                visdict['depth_mask'] = depth_mask.repeat(1,3,1,1)
                visdict['inpaint_in'] = cropped_rgb_render
                visdict['inpaint_update_mask'] = cropped_update_mask.repeat(1,3,1,1)
                # self.diffusion.use_inpaint = False
                # cropped_update_mask = depth_mask
                
                cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(),
                                                                        cropped_depth_render.detach(),
                                                                        guidance_scale=7.5,
                                                                        strength=1.0, update_mask=cropped_update_mask,
                                                                        fixed_seed=0,
                                                                        check_mask=None,
                                                                        intermediate_vis=False)
                visdict['inpaint_out'] = cropped_rgb_output*depth_mask
                # visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                print(savepath)
                exit()
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            
            
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view.{video_type}')
            self.save_video(videopath, image_list, fps=fps)
    
    @torch.no_grad()
    def pipeline(self, savefolder, frame_id=100):
        # model_dict = self.model_dict()
        # checkpoint = torch.load(self.cfg.ckpt_path)
        # util.copy_state_dict(model_dict['model'], checkpoint['model'])
        # cfg = self.cfg 
        # import ipdb; ipdb.set_trace()
        frame_id = 200
        self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = frame_id + 1
        self.train_dataset = build_dataset.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        data_iter = iter(self.train_dataloader)
        batch = next(data_iter)
        util.move_dict_to_device(batch, device=self.device)

        frame_id = batch['frame_id'][0]
        visdict = {'image': batch['image']}

        ##### 
        visdict = {}
        # faces = self.model.faces
        # 1. canonical SMPL-X (frontal, ideally also 45 degrees rotated),
        # verts = self.model.verts
        ## set caninical cam
        canonical_cam = torch.tensor([6.0, 0., 1.]).to(self.device)
        # batch['cam'][:1] = self.model.canonical_cam    
        # import ipdb; ipdb.set_trace()
        batch_size = 1
        # import ipdb; ipdb.set_trace()
        # batch['full_pose'][:, 15] = self.model.canonical_pose[15]
        orig_pose = batch['full_pose'].clone()
        # batch['full_pose'][:, 12:] = self.model.canonical_pose[12:]
        # import ipdb; ipdb.set_trace()
        self.model.canonical_pose[0, 1, 1] = -1.
        self.model.canonical_pose[0, 2, 2] = -1.
        batch['full_pose'][:, :] = self.model.canonical_pose[:]
        batch['exp'][:] = 0.
        
        mesh_rendering = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderShape=True,
                                    clean_offset=True
                                    ) 
        visdict['1_canonical_smplx'] = mesh_rendering['shape_image']
        # batch['full_pose'][:1] = self.model.canonical_pose
        # visdict['1_canonical_smplx'] = self.model.forward_mesh(batch, renderShape=True, clean_offset=True)['shape_image']
        
        # 2. canonical SMPL-X with offset (frontal, ideally also 45 degrees rotated),
        mesh_rendering = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderShape=True,
                                    renderImage=True,
                                    clean_offset=False
                                    ) 
        visdict['2_canonical_smplx_offset_shape'] = mesh_rendering['shape_image']
        opdict = self.model.forward_vis(batch, returnMask=True)
        visdict['2_canonical_smplx_offset_color'] = mesh_rendering['mesh_image']*opdict['mesh_mask'] + torch.ones_like(mesh_rendering['mesh_image'])*(1-opdict['mesh_mask'])
        visdict['2_canonical_nerf'] = opdict['render_hybrid']*opdict['nerf_mask'] + torch.ones_like(opdict['render_hybrid'])*(1-opdict['nerf_mask'])

        # 3. in posed space
        batch['full_pose'] = orig_pose
        mesh_rendering = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderShape=True,
                                    renderImage=True,
                                    clean_offset=False
                                    ) 
        visdict['3_posed_smplx_offset_shape'] = mesh_rendering['shape_image']
        # import ipdb; ipdb.set_trace()
        opdict = self.model.forward_vis(batch, returnMask=True)
        visdict['3_posed_smplx_offset_color'] = mesh_rendering['mesh_image']*opdict['mesh_mask'] + torch.ones_like(mesh_rendering['mesh_image'])*(1-opdict['mesh_mask'])
        visdict['3_posed_nerf'] = opdict['render_hybrid']*opdict['nerf_mask'] + torch.ones_like(opdict['render_hybrid'])*(1-opdict['nerf_mask'])
        visdict['3_posed_hybrid'] = opdict['render_hybrid']
        visdict['3_posed_render'] = opdict['render']
        
        # 4. gt image
        visdict['4_gt'] = batch['image'][:1]
        visdict['4_gt_mask'] = batch['nonskin_mask'][:1].repeat(1,3,1,1)
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        # batch['full_pose'] = orig_pose
        # visdict['3_posed_smplx_offset_shape'] = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
        #                             batch['full_pose'], 
        #                             batch['cam'], 
        #                             batch.get('exp', None),
        #                             renderShape=True,
        #                             renderImage=True,
        #                             clean_offset=False
        #                             )['shape_image']
        # batch['full_pose'] = orig_pose
        # visdict['3_canonical_smplx_offset_color'] = self.model.mesh(self.model.beta.repeat(batch_size, 1), 
        #                             batch['full_pose'], 
        #                             batch['cam'], 
        #                             batch.get('exp', None),
        #                             renderShape=True,
        #                             renderImage=True,
        #                             clean_offset=False
        #                             )['mesh_image']
        
        # # 4. posed colored SMPL-X with offset,
        # batch = self.posemodel(batch)
        # opdict = self.model.forward_mesh(batch)
        # visdict['4_posed_body'] = opdict['mesh_image']
        # visdict['4_posed_body_shape'] = self.model.forward_mesh(batch, renderShape=True)['shape_image']

        # import ipdb; ipdb.set_trace()

        # # 5. Canonical NeRF clothing (frontal, ideally also 45 degrees rotated)
        # batch['cam'][:1] = self.model.canonical_cam    
        # batch['full_pose'][:1] = self.model.canonical_pose
        # opdict = self.model(batch, train=False, render_cloth=True, render_shape=False)
        # visdict['5_canonical_cloth'] = opdict['nerf_fine_image']

        # # 6. Posed NeRF clothing
        # batch = self.posemodel(batch)
        # opdict = self.model(batch, train=False, render_cloth=True, render_shape=False)
        # visdict['6_posed_cloth'] = opdict['nerf_fine_image']

        # # 7. Our rendering
        # opdict = self.model(batch, train=False)
        # visdict['7_our_rendering'] = opdict['nerf_fine_image']

        # # 8. GT image
        # visdict['8_gt_image'] = batch['image']

        # # 9. GT segmentation
        # visdict['9_gt_segmentation'] = batch['cloth_mask'].expand(-1,3,-1,-1)

        savepath = os.path.join(savefolder, f'pipeline.jpg')
        grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=self.image_size)
        # image_list.append(grid_image[:,:,[2,1,0]])
        print(savepath)
        for key in visdict.keys():
            image = visdict[key]
            cv2.imwrite(os.path.join(savefolder, f'pipeline_{key}.jpg'),util.tensor2image(visdict[key][0]))
        print(savefolder)
        exit()