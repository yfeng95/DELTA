''' Convert data from IMavatar to SCARF data format
'''
import os
from pathlib import Path
from re import S
from socket import IP_DEFAULT_MULTICAST_LOOP
import sys
import os.path as osp
import numpy as np
from glob import glob
import pickle
import torch
import argparse
from skimage.io import imread, imsave
import cv2
from pytorch3d.io import load_obj
from tqdm import tqdm
import shutil
import imageio
import torch.nn.functional as F
import torch.nn as nn

import pytorch3d
from pytorch3d.renderer import (look_at_view_transform, FoVOrthographicCameras, PointsRasterizationSettings,
                                PointsRenderer, PulsarPointsRenderer, PointsRasterizer, AlphaCompositor,
                                NormWeightedCompositor)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.utils.deca_render import SRenderY
from lib.utils import camera_util
from lib.utils import util, lossfunc
from lib.model.smplx import SMPLX
from lib.model.FLAME import FLAMETex, texture_flame2smplx
from lib.core.config import cfg, update_cfg
from lib.utils import rotation_converter
from lib.render.mesh_helper import pytorch3d_rasterize, render_shape, render_texture
from lib.dataset.face_video import Dataset

model_cfg = cfg.model

def cache_data(dataset, update=True): #, cache_pose=True, cache_cam=True, cache_exp=True, cache_beta=True):
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
    init_full_pose = torch.zeros([55, 3]).float() + 0.00001
    init_full_pose[0, 0] = np.pi
    ## init shoulder
    init_full_pose[16,2] = -np.pi*60/180
    init_full_pose[17,2] = np.pi*60/180

    init_full_pose = rotation_converter.batch_euler2axis(init_full_pose)
    
    if update or not os.path.exists(dataset.pose_cache_path):    
        n_frames = len(dataset)
        print(f'caching data...')
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
            ## use init shoulder pose
            pose_axis[[16,17],:] = init_full_pose[[16,17],:]
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
    def __init__(self, dataset, optimize_cam=False, use_perspective=False, 
                    use_appearance=False, appearance_dim=0, use_deformation=False, 
                    use_light=False, n_lights=3, 
                    deformation_dim=0):
        super(PoseModel, self).__init__()
        self.subject = dataset.subject   
        # first load cache data
        cache_data(dataset, update=True)
        pose_dict = torch.load(dataset.pose_cache_path)
        for key in pose_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(pose_dict[key]))
        self.pose_dict = pose_dict
        
        self.optimize_cam = optimize_cam
        self.use_appearance = use_appearance
        self.use_deformation = use_deformation
        self.use_light = use_light

        if self.optimize_cam:
            # first load cache data
            cam_dict = torch.load(dataset.cam_cache_path)
            for key in cam_dict.keys():
                self.register_parameter(key, torch.nn.Parameter(cam_dict[key]))
            self.cam_dict = cam_dict
            if use_perspective:
                self.register_parameter('focal', torch.nn.Parameter(self.cam_dict[key][:,0]))
        if self.use_deformation:
            deformation_dict = torch.load(dataset.cam_cache_path)
            for key in deformation_dict.keys():
                self.register_parameter(key.replace('cam', 'deformation'), torch.nn.Parameter(torch.zeros([1,deformation_dim])))
            self.deformation_dict = deformation_dict
        if self.use_appearance:
            appearance_dict = torch.load(dataset.cam_cache_path)
            for key in appearance_dict.keys():
                self.register_parameter(key.replace('cam', 'appearance'), torch.nn.Parameter(torch.zeros([1,appearance_dim])))
            self.appearance_dict = appearance_dict
        if self.use_light:
            ## use SH light
            # if n_lights == 9:
            #     light_dict = torch.load(dataset.light_cache_path)
            #     for key in light_dict.keys():
            #         self.register_parameter(key.replace('cam', 'light'), torch.nn.Parameter(torch.zeros([1, 9, 3])))
            # else:    
            light_dict = torch.load(dataset.light_cache_path)
            for key in light_dict.keys():
                self.register_parameter(key, torch.nn.Parameter(light_dict[key]))
            self.light_dict = light_dict
            
        self.opt_exp = True
        if self.opt_exp:
            exp_dict = torch.load(dataset.exp_cache_path)
            for key in exp_dict.keys():
                self.register_parameter(key, torch.nn.Parameter(exp_dict[key]))
            self.exp_dict = exp_dict

        self.use_perspective = use_perspective

        ## init pose
        init_full_pose = torch.zeros([55, 3]).float().cuda() + 0.00001
        init_full_pose[0, 0] = np.pi
        ## init shoulder
        init_full_pose[16,2] = -np.pi*60/180
        init_full_pose[17,2] = np.pi*60/180

        init_full_pose = rotation_converter.batch_euler2axis(init_full_pose)[None, ...]
        self.init_full_pose = init_full_pose
        
    def forward(self, batch):
        # return poses of given frame_ids
        name = self.subject
        if 'cam_id' in batch.keys():
            cam_id = batch['cam_id']
            names = [f'{name}_{cam}' for cam in cam_id]
        else:
            names = [name]*len(batch['frame_id'])
        frame_ids = batch['frame_id']
        batch_size = len(frame_ids)
        batch_pose = torch.cat([getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)])
        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        batch['init_full_pose'] = batch['full_pose'].clone()
        batch['full_pose'] = batch_pose
        
        # do not optimize body pose
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        fix_idx = list(range(1, 12)) + [13, 14, 18, 19, 20, 21] + list(range(25, 55))
        batch['full_pose'][:, fix_idx] = batch['init_full_pose'][:,fix_idx]
        ## fix shoulder pose
        # init_full_pose = self.init_full_pose.expand(batch_size, -1, -1)
        # batch['init_full_pose'] = rotation_converter.batch_axis2matrix(init_full_pose.clone().reshape(
        #     -1, 3)).reshape(batch_size, 55, 3, 3)
        # batch['full_pose'][:, 16:18] = batch['init_full_pose'][:, 16:18]
        
        if self.optimize_cam:
            batch['init_cam'] = batch['cam'].clone()
            batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])
            if self.use_perspective:
                batch['cam'][:,0] = self.focal
        if self.use_deformation:
            batch['deformation_code'] = torch.cat([getattr(self, f'{names[i]}_deformation_{frame_ids[i]}') for i in range(batch_size)])
        if self.use_appearance:
            batch['appearance_code'] = torch.cat([getattr(self, f'{names[i]}_appearance_{frame_ids[i]}') for i in range(batch_size)])
        if self.opt_exp:
            batch['exp'] = torch.cat([getattr(self, f'{names[i]}_exp_{frame_ids[i]}') for i in range(batch_size)])
        if self.use_light:
            batch['light'] = torch.cat([getattr(self, f'{names[i]}_light_{frame_ids[i]}') for i in range(batch_size)])
        return batch
    
    def cont_poses(self, start=0, n_pose=4):
        ''' query continuous pose
        '''
        # random start
        keys = self.pose_dict.keys()
        names = list(keys)[start:start+n_pose]
        batch_pose = torch.cat([getattr(self, name) for name in names])
        return batch_pose
            
class SMPLX_optimizer(torch.nn.Module):

    def __init__(self, dataset=None, device='cuda:0', image_size=512, light_type='SH'):
        super(SMPLX_optimizer, self).__init__()
        self.cfg = cfg
        self.device = device
        self.image_size = image_size
        self.light_type = light_type
        # smplx_cfg.model.n_shape = 100
        # smplx_cfg.model.n_exp = 50
        self.dataset = dataset
        self._setup_model()
        self._setup_renderer()
        self._setup_loss_weight()
        self.configure_optimizers()
        # loss
        # self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=model_cfg.fr_model_path)

    def _setup_model(self):
        ## pose model
        self.posemodel = PoseModel(dataset=self.dataset,
                                   optimize_cam=True,
                                   use_perspective=False,
                                   use_appearance=False,
                                   use_deformation=False,
                                   use_light=True,
                                   n_lights=9).to(self.device)

        ## smplx model
        self.smplx = SMPLX(model_cfg).to(self.device)
        self.flametex = FLAMETex(model_cfg).to(self.device)
        self.verts = self.smplx.v_template
        self.faces = self.smplx.faces_tensor
        ## iris index
        self.idx_iris = [9503, 10049]  #right, left
        ## part index
        ##--- load vertex mask
        with open(self.cfg.model.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.cfg.model.flame_ids_path)
        with open(self.cfg.model.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        # verts = torch.nn.Parameter(self.v_template, requires_grad=True)
        exclude_idx = []
        exclude_idx += list(hand_idx['left_hand'])
        exclude_idx += list(hand_idx['right_hand'])
        exclude_idx += list(flame_vertex_mask['face'])
        exclude_idx += list(flame_vertex_mask['left_eyeball'])
        exclude_idx += list(flame_vertex_mask['right_eyeball'])
        exclude_idx += list(flame_vertex_mask['left_ear'])
        exclude_idx += list(flame_vertex_mask['right_ear'])
        all_idx = range(self.smplx.v_template.shape[1])
        face_idx = list(flame_vertex_mask['face'])
        body_idx = [i for i in all_idx if i not in face_idx]
        self.part_idx_dict = {
            'face': flame_vertex_mask['face'],
            'hand': list(hand_idx['left_hand']) + list(hand_idx['right_hand']),
            'exclude': exclude_idx,
            'body': body_idx
        }

    def configure_optimizers(self):
        ###--- optimizer for training nerf model.
        # whether to use apperace code
        # nerf_params = list(self.model.mlp_coarse.parameters()) + list(self.model.mlp_fine.parameters())
        init_beta = torch.zeros([1, model_cfg.n_shape]).float().to(self.device)
        self.register_parameter('beta', torch.nn.Parameter(init_beta))
        init_tex = torch.zeros([1, model_cfg.n_tex]).float().to(self.device)
        self.register_parameter('tex', torch.nn.Parameter(init_tex))
        parameters = [{'params': [self.beta, self.tex], 'lr': 1e-3}]

        # parameters.append(
        #     {'params': self.posemodel.parameters(), 'lr': 1e-3})
        cam_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' in name]
        pose_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' not in name]
        parameters.append({'params': pose_parameters, 'lr': 5e-4})
        parameters.append({'params': cam_parameters, 'lr': 2e-4})

        self.optimizer = torch.optim.Adam(params=parameters)

    def _setup_renderer(self):
        ## setup raterizer
        uv_size = 1024
        topology_path = self.cfg.model.topology_path
        # cache data for smplx texture
        self.smplx_texture = imread(self.cfg.model.smplx_tex_path) / 255.
        self.cached_data = np.load(self.cfg.model.flame2smplx_cached_path, allow_pickle=True, encoding='latin1').item()

        self.render = SRenderY(self.image_size,
                               obj_filename=topology_path,
                               uv_size=uv_size,
                               rasterizer_type='pytorch3d').to(self.device)
        mask = imread(self.cfg.model.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_eye_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)

        # face region mask in flame texture map
        mask = imread(self.cfg.model.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)

        ########### silhouette rendering
        ## camera
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros([1, 3])
        batch_size = 1
        self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(R=R.expand(batch_size, -1, -1),
                                                                         T=T.expand(batch_size, -1),
                                                                         znear=0.0).to(self.device)

        blend_params = BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = RasterizationSettings(image_size=self.image_size,
                                                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                                                faces_per_pixel=50,
                                                bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def forward_model(self, batch, returnMask=False, returnRendering=False, returnNormal=False):
        ''' forward SMPLX model
        Args:
            batch: dict, batch data
                'beta': [B, n_shape(200)], shape parameters
                'exp': [B, n_exp(100)], expression parameters
                'full_pose': [B, n_pose(55), 3, 3], pose parameters, in Rotatation matrix format
                'cam': [B, 3], camera parameters, [scale, tx, ty], use orthographic projection
            returnMask: bool, whether to return mask
            returnRendering: bool, whether to return rendering
            returnNormal: bool, whether to return normal
        Returns:
            opdict: dict, output dict
        '''
        opdict = {}
        verts, landmarks, joints = self.smplx(shape_params=batch['beta'],
                                              expression_params=batch['exp'],
                                              full_pose=batch['full_pose'])
        cam = batch['cam']
        trans_verts = util.batch_orth_proj(verts, cam)
        pred_lmk = util.batch_orth_proj(landmarks, cam)[:, :, :2]
        # convert smpl-x landmarks to flame landmarks (right order)
        pred_lmk = torch.cat([pred_lmk[:, -17:], pred_lmk[:, :-17]], dim=1)

        opdict['verts'] = verts
        opdict['trans_verts'] = trans_verts
        opdict['pred_lmk'] = pred_lmk

        # render mask for silhouette loss
        batch_size = verts.shape[0]
        faces = self.faces.unsqueeze(0).expand(batch_size,-1,-1)
        if returnMask:
            trans_verts_mask = trans_verts.clone()
            trans_verts_mask[:,:,:2] = -trans_verts_mask[:,:,:2]
            trans_verts_mask[:,:,-1] = -trans_verts_mask[:,:,-1] + 50
            mesh = Meshes(
                verts = trans_verts_mask,
                faces = faces
            )
            mesh_mask = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:,3:]
            opdict['mesh_mask'] = mesh_mask
        # render image for image loss
        if returnRendering:
            # calculate normal for shading
            normal_verts = trans_verts.clone()
            normal_verts[..., 0] = -normal_verts[..., 0]
            normals = util.vertex_normals(normal_verts, faces)                    
            trans_verts[..., -1] = trans_verts[..., -1] + 50
            albedo = self.flametex(batch['tex'])
            rendering_out = self.render(verts,
                                trans_verts,
                                albedo,
                                lights=batch['light'],
                                light_type=self.light_type,
                                given_normal=normals)
            opdict['image'] = rendering_out['images']
            opdict['albedo_image'] = rendering_out['albedo_images']
            opdict['shading_image'] = rendering_out['shading_images']
            
        return opdict

    def _setup_loss_weight(self):
        loss_cfg = cfg.loss
        loss_cfg.lmk = 1.
        loss_cfg.eyed = 2.
        loss_cfg.lipd = 0.5
        # mask
        loss_cfg.inside_mask = 1.
        loss_cfg.mesh_mask = 1.
        # image
        loss_cfg.image = 2.
        loss_cfg.albedo = 2.
        
        self.loss_cfg = loss_cfg

    def optimize(self, batch, iters=1, vis=False,
                    data_type='else',
                    lmk_only=True,
                    use_mask = False,
                    use_rendering = False,
                    use_normal = False):
        ''' optimize the pose and shape parameters of the model, using lmk loss only
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        '''
        for iter in range(iters):
            # first stage, only optimize global and neck pose
            image = batch['image']
            batch_size = image.shape[0]
            batch['beta'] = self.beta.expand(batch_size, -1)
            batch['tex'] = self.tex.expand(batch_size, -1)
            batch = self.posemodel(batch)
            opdict = self.forward_model(batch, returnMask=use_mask, returnRendering=use_rendering, returnNormal=use_normal)

            losses = {}
            if 'iris' in batch.keys():
                pred_iris = opdict['trans_verts'][:, self.idx_iris, :2]
                pred_lmk = torch.cat([opdict['pred_lmk'], pred_iris], dim=1)
                gt_lmk = torch.cat([batch['lmk'][:, :], batch['iris'][:, :]], dim=1)
            else:
                pred_lmk = opdict['pred_lmk']
                gt_lmk = batch['lmk']
            
            # lmk loss, use confidence, if confidence is 0, then ignore this points (e.g. especially for iris points)
            losses['lmk'] = lossfunc.batch_kp_2d_l1_loss(gt_lmk, pred_lmk)*self.loss_cfg.lmk
            # eye distance loss from DECA
            if self.cfg.loss.eyed > 0.:
                losses['eyed'] = lossfunc.eyed_loss(pred_lmk[:,:68,:2], gt_lmk[:,:68,:2])*self.cfg.loss.eyed
            
            if use_mask:
                losses['mesh_inside_mask'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.loss_cfg.inside_mask
                mesh_mask = opdict['mesh_mask']
                non_skin_mask = 1-mesh_mask.detach()
                hair_only = non_skin_mask*batch['nonskin_mask']
                mesh_mask = mesh_mask + hair_only
                opdict['mesh_mask'] = mesh_mask
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.loss_cfg.mesh_mask
            if use_rendering:
                losses['image'] = (batch['face_mask'] *(batch['image'] - opdict['image'])).abs().mean() * self.loss_cfg.image
                losses['image_albedo'] = (batch['face_mask'] *
                                        (batch['image'] - opdict['albedo_image'])).abs().mean() * self.loss_cfg.albedo
            
            losses['reg_shape'] = (torch.sum(batch['beta']**2) / 2).mean() * 1e-5
            losses['reg_exp'] = (torch.sum(batch['exp']**2) / 2).mean() * 1e-5
            losses['reg_tex'] = (torch.sum(batch['tex']**2) / 2).mean() * 5e-5
            
            # regs in init shoulder
            shoulder_pose = batch['full_pose'][:,16:18]            
            init_full_pose = self.posemodel.init_full_pose.expand(batch_size, -1, -1)
            batch['init_full_pose'] = rotation_converter.batch_axis2matrix(init_full_pose.clone().reshape(
                -1, 3)).reshape(batch_size, 55, 3, 3)
            # batch['full_pose'][:, 16:18] = batch['init_full_pose'][:, 16:18]
            losses['reg_shoulder'] = (batch['full_pose'][:, 16:18] - batch['init_full_pose'][:, 16:18]).mean().abs()*10
            # losses['reg_shoulder'] = (shoulder_pose_axis - torch.mean(shoulder_pose_axis, dim=0, keepdim=True)).mean().abs()*100
            
            ## pose consistency except for global pose
            # start_idx = int(batch['idx'][0])
            # cont_poses = self.posemodel.cont_poses(start=start_idx, n_pose=4) #[n_pose, 55, 3]
            # cont_poses = cont_poses[:,1:,:] # exclude global pose
            # losses['pose_consistency'] = (cont_poses - cont_poses.mean(dim=0, keepdim=True)).mean().abs()*1000000000
            
            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

        if vis:
            loss_info = f"Iter: {iter}/{iters}: "
            for k, v in losses.items():
                loss_info = loss_info + f'{k}: {v:.6f}, '
            print(loss_info)
            visdict = {
                'inputs': image,
                'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
                'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=True)
            }
            # render shape
            faces = self.smplx.faces_tensor
            shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                    faces=faces.expand(batch_size, -1, -1),
                                    image_size=image.shape[-1],
                                    background=image)
            visdict['shape'] = shape_image
            for key in opdict.keys():
                if 'image' in key:
                    visdict[key] = opdict[key]
            vis_image = util.visualize_grid(visdict, return_gird=True)

            return vis_image
        else:
            return None
                
    def run(self,
            savepath=None,
            epoches=1000,
            batch_size=1,
            vis_step=100,
            data_type='fix_shoulder',
            args=None):
        '''
        Args:
            iter_list: list, number of iterations for each stage
            savepath: str, path to save the results
            args: dict, additional arguments
        '''
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=min(batch_size, 4),
                                                 pin_memory=True,
                                                 drop_last=False)
        
        vispath = os.path.join(savepath, 'optimize_steps')
        os.makedirs(vispath, exist_ok=True)
        for epoch in tqdm(range(epoches)):
            if epoch % vis_step == 0:
                self.save(savepath, self.dataset, epoch=epoch)
            for i, batch in enumerate(dataloader):
                util.move_dict_to_device(batch, self.device)
                vis_image = self.optimize(batch,
                                vis = i==len(dataloader)-1,
                                use_mask=True,
                                use_rendering=True)
                if vis_image is not None:
                    cv2.imwrite(os.path.join(vispath, f'{epoch:06d}.png'), vis_image)
                    print(os.path.join(vispath, f'{epoch:06d}.png'), vis_image.shape)
                    
        self.save(savepath, self.dataset, epoch=epoch)
        cache_data(self.dataset)
        
    def save(self, savepath, dataset, vis=True, epoch=0):
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 drop_last=False)
        subject = self.posemodel.subject
        video_path = os.path.join(savepath, f'epoch_{epoch:06d}.mp4')
        writer = imageio.get_writer(video_path, fps=10)
        for batch in tqdm(dataloader):
            util.move_dict_to_device(batch, self.device)
            batch = self.posemodel(batch)
            
            image = batch['image']
            batch_size = image.shape[0]
            batch['beta'] = self.beta.expand(batch_size, -1)
            pixie_param = {
                    'shape': self.beta.expand(batch_size, -1),
                    'full_pose': batch['full_pose'],
                    'light': batch['light'],
                    'cam': batch['cam'],
                    'exp': batch['exp'],
                    'tex': self.tex.expand(batch_size, -1)
                }
            for i in range(batch_size):        
                frame_id = batch['frame_id'][i]        
                util.save_params(os.path.join(savepath, f'{subject}_f{frame_id}_param.pkl'), 
                                 pixie_param,
                                 ind=i)
            if vis:
                vis_path = os.path.join(savepath, f'epoch_{epoch:06d}')
                os.makedirs(vis_path, exist_ok=True)
                opdict = self.forward_model(batch)
                visdict = {
                    'inputs': image
                }
                # render shape
                faces = self.smplx.faces_tensor
                shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                        faces=faces.expand(batch_size, -1, -1),
                                        image_size=image.shape[-1],
                                        background=image)
                visdict['shape'] = shape_image
                for key in opdict.keys():
                    if 'image' in key:
                        visdict[key] = opdict[key]
                vis_image = util.visualize_grid(visdict, return_gird=True, print_key=False)
                h, w, _ = vis_image.shape
                size = h//batch_size
                for i in range(batch_size):        
                    frame_id = batch['frame_id'][i]        
                    cv2.imwrite(os.path.join(vis_path, f'{subject}_{frame_id}.png'), vis_image[i*size:(i+1)*size])
                    writer.append_data(vis_image[i*size:(i+1)*size, :, ::-1])
        writer.close()
                
        # save one obj example
        util.write_obj(os.path.join(vis_path, f'{epoch:06}.obj'),
            opdict['trans_verts'][0].detach().cpu().numpy(), 
            faces.cpu().numpy())
        
''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
 [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',
                        type=str,
                        default='/is/cluster/yfeng/Data/Projects-data/DELTA/datasets/face')
    # parser.add_argument('--subject', type=str, default='MVI_1810')
    parser.add_argument('--data_cfg', type=str, default=None)
    parser.add_argument('--subject', type=str, default='person_0000')
    parser.add_argument('--data_type', type=str, default='fix_shoulder')
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--epoches', type=int, default=500)
    parser.add_argument('--iters', type=str, default='2000,500,2000')
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--light_type', type=str, default='SH')
    parser.add_argument('--seed', type=int, default=9988)
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--use_id", default=False, action="store_true")
    args = parser.parse_args()

    # load dataset
    # load dataset
    if args.data_cfg is None:
        subject = args.subject
        data_cfg_file = os.path.join('/home/yfeng/Projects/DELTA/configs/data/face', f'{args.subject}.yml')
    else:
        subject = args.subject
        data_cfg_file = args.data_cfg
    data_cfg = update_cfg(cfg, data_cfg_file).dataset
    data_cfg.image_size = args.image_size
    data_cfg.load_lmk = True
    ### DEBUG    
    # data_cfg.path = '/home/yfeng/Data/Projects-data/DHA/datasets/processed_data/dha_face'

    # batch for cluster running
    imagepath_list = glob(os.path.join(data_cfg.path, subject, 'image', f'{args.subject}_*.png'))
    imagepath_list = sorted(imagepath_list)
    imagepath_list = imagepath_list[args.batch_idx * args.batch_num:(args.batch_idx + 1) * args.batch_num]
    savepath = os.path.join(args.datapath, subject, f'smplx_all')
    iters = [int(x) for x in args.iters.split(',')]
    print(f'run tracking, results will be saved in {savepath}')
        
    dataset = Dataset.from_config(data_cfg, mode='train') 
    sample = dataset[0]
    optimizer = SMPLX_optimizer(dataset, args.device, light_type=args.light_type)
    optimizer.run(savepath, epoches=args.epoches, batch_size=args.batch_size, vis_step=args.vis_step, data_type=args.data_type)
        
# python smplx_detail_tracking_single.py  --batch_idx 500 --batch_num 1 --batch_size 1  --iters 2000