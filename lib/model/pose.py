import os
import torch
from torch import nn
from tqdm import tqdm
from ..utils import rotation_converter
from loguru import logger

def cache_data(dataset, update=False): #, cache_pose=True, cache_cam=True, cache_exp=True, cache_beta=True):
    '''read pose data from original path, save cache data for fast loading
    Args:
        update: if True, update the cache data. If False, load the cache data if exists.
    '''
    root_dir = os.path.join(dataset.dataset_path, 'cache')
    os.makedirs(root_dir, exist_ok=True)
    dataset.pose_cache_path = os.path.join(root_dir, 'pose.pt')
    dataset.cam_cache_path = os.path.join(root_dir, 'cam.pt')
    dataset.exp_cache_path = os.path.join(root_dir, 'exp.pt')
    dataset.beta_cache_path = os.path.join(root_dir, 'beta.pt')
    dataset.light_cache_path = os.path.join(root_dir, 'light.pt')
    if update or not os.path.exists(dataset.pose_cache_path):    
        n_frames = len(dataset)
        logger.info(f'caching data...')
        pose_dict = {}
        cam_dict = {}
        exp_dict = {}
        beta_dict = {}
        light_dict = {}
        init_beta = []
        for i in tqdm(range(n_frames)):
            sample = dataset[i]
            frame_id = sample['frame_id']
            name = sample['name']
            if 'cam_id' in sample.keys():
                cam_id = sample['cam_id']
                name = f'{name}_{cam_id}'

            exp_dict[f'{name}_exp_{frame_id}'] = sample['exp'][None,...]
            cam_dict[f'{name}_cam_{frame_id}'] = sample['cam'][None,...]
            init_beta.append(sample['beta'])
            
            pose_matrix = sample['full_pose']
            pose_axis = rotation_converter.batch_matrix2axis(pose_matrix) + 1e-8
            pose_dict[f'{name}_pose_{frame_id}'] = pose_axis.clone()[None,...]
            
            if 'light' in sample.keys():
                light = sample['light'].squeeze()
            else:
                light = torch.zeros(9, 3)
            light_dict[f'{name}_light_{frame_id}'] = light[None,...]
        init_beta = torch.stack(init_beta)
        init_beta = init_beta.mean(0)[None,...]
        beta_dict[f'{name}_beta'] = init_beta
        beta_dict['beta'] = init_beta

        torch.save(pose_dict, dataset.pose_cache_path)
        torch.save(cam_dict, dataset.cam_cache_path)
        torch.save(exp_dict, dataset.exp_cache_path)
        torch.save(beta_dict, dataset.beta_cache_path)
        torch.save(light_dict, dataset.light_cache_path)

class PoseModel(nn.Module):
    def __init__(self, dataset, use_perspective=False, 
                    appearance_dim=0, deformation_dim=0, fix_hand=True):
        super(PoseModel, self).__init__()
        self.subject = dataset.subject
        self.use_perspective = use_perspective
        self.fix_hand = fix_hand

        # first load cache data
        cache_data(dataset, update=False)
        # cam         
        cam_dict = torch.load(dataset.cam_cache_path)
        for key in cam_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(cam_dict[key]))
        self.cam_dict = cam_dict
        if use_perspective:
            self.register_parameter('focal', torch.nn.Parameter(self.cam_dict[key][:,0]))
        # pose
        pose_dict = torch.load(dataset.pose_cache_path)
        for key in pose_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(pose_dict[key]))
            # seperate pose: eyeball, jaw are for expressions
            pose_for_exp = pose_dict[key][:,[22,23,24]].clone()
            self.register_parameter(key.replace('_pose_', '_pexp_'), torch.nn.Parameter(pose_for_exp))
        self.pose_dict = pose_dict     
        
        # exp
        exp_dict = torch.load(dataset.exp_cache_path)
        for key in exp_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(exp_dict[key]))
        self.exp_dict = exp_dict
        # light if exists
        if os.path.exists(dataset.light_cache_path):
            light_dict = torch.load(dataset.light_cache_path)
            for key in light_dict.keys():
                self.register_parameter(key, torch.nn.Parameter(light_dict[key]))
            self.light_dict = light_dict
        
    def forward(self, batch):
        name = self.subject
        if 'cam_id' in batch.keys():
            cam_id = batch['cam_id']
            names = [f'{name}_{cam}' for cam in cam_id]
        else:
            names = [name]*len(batch['frame_id'])
        frame_ids = batch['frame_id']
        batch_size = len(frame_ids)
        # cam
        batch['init_cam'] = batch['cam'].clone()
        batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])
        # pose
        batch_pose = torch.cat([getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)])
        batch_pexp = torch.cat([getattr(self, f'{names[i]}_pexp_{frame_ids[i]}') for i in range(batch_size)])
        # replace pose with pexp
        batch_pose[:,[22,23,24]] = batch_pexp
        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        # if self.fix_hand:
        #     batch_pose[:, -30:] = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(30,1,1).to(batch_pose.device)
        batch['init_full_pose'] = batch['full_pose'].clone()
        batch['full_pose'] = batch_pose        
        # exp
        batch['init_exp'] = batch['exp'].clone()
        batch['exp'] = torch.cat([getattr(self, f'{names[i]}_exp_{frame_ids[i]}') for i in range(batch_size)])
        # light
        if 'light' in batch.keys():
            batch['light'] = torch.cat([getattr(self, f'{names[i]}_light_{frame_ids[i]}') for i in range(batch_size)])
        return batch