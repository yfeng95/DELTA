""" Rendering Functions
About Volume Rendering: 
the key is how to sample the rays and points in each ray. 
in nerf paper:
    use all rays, and train coarse mlp to help with sampling points
nerfacc:
    use occupancy grid, run ray marching to help sampling
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .volume_helper import generate_image_rays, fancy_integration, perturb_points, sample_pdf
from ..utils import util

class VanillaSampler(nn.Module):
    """ Vanilla Sampler used in orignal NeRF paper
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, n_points = 128, n_coarse_points = 64):
        super().__init__()
        self.n_coarse_points = n_coarse_points
        self.n_points = n_points
    
    @torch.no_grad()
    def sample_points(self, rays, n_points, n_coarse_points=None, perturb=False, z_coarse=None, weights=None, combine=True):
        '''
        '''
        device = rays.device
        batch_size = rays.shape[0]
        center = rays[..., :3]
        direction = rays[..., 3:6]
        near = rays[..., 6]
        far = rays[..., 7]
        # sample coarse points
        if z_coarse is None:
            z_steps = torch.linspace(0, 1, n_points, device=device)
            z_steps = z_steps.unsqueeze(0).expand(batch_size, -1)  # (B, Kc)
            z_steps = z_steps[:, None, :, None]  #[batch_size, 1, n_points, 1]
            z_vals = near[:, :, None, None] * (1 - z_steps) + far[:, :, None, None] * z_steps
            xyz = center[:, :, None, :] + z_vals * direction[:, :, None, :]
            ## pertub_points
            if perturb:
                xyz, z_vals = perturb_points(xyz, z_vals, direction, device=device)
        # sample fine points depends on weights
        else:
            z_vals_coarse = z_coarse.reshape(-1, n_coarse_points)
            z_vals_mid = 0.5 * (z_vals_coarse[..., :-1] + z_vals_coarse[..., 1:])
            z_vals = sample_pdf(z_vals_mid,
                                weights[:, 1:-1].detach(),
                                n_points,
                                det=True).detach()
            if combine:
                z_vals = torch.cat([z_coarse, z_vals], dim=-1)
                z_vals, _ = torch.sort(z_vals, dim=-1)
            z_vals = z_vals.reshape(rays.shape[0], rays.shape[1], -1, 1)
            xyz = center[:, :, None, :] + z_vals * direction[:, :, None, :]
        # import ipdb; ipdb.set_trace()
        
        xyz = xyz.view(batch_size, -1, 3)
        return xyz, z_vals

    def forward(self, nerf_model, batch=None, rays=None, perturb=False,
                white_bg=False, last_back=False,  # setting for coarse rendering
                clamp_mode='relu', noise_std=0., last_rgb=None):
        # if n_coarse_points is not 0, then first sample nerf to get the density
        # if nerf_coarse_model is not None, then use nerf_coarse_model to sample coarse points
        # then sampling fine points around the surface
        output = {}
        #--1.random sampling 
        # normally perturb when training, but not when testing
        assert self.n_coarse_points > 0, "n_coarse_points should be larger than 0"
        xyz_coarse, z_vals_coarse = self.sample_points(rays, n_points=self.n_coarse_points, perturb=perturb)
        #--2.calculate weights for fine sampling 
        if self.n_points > 0:
            # use nerf coarse model if it is not None
            if nerf_model.nerf_coarse is not None:
                rgb, sigma = nerf_model(xyz_coarse, batch, return_correction=False, use_coarse=True)
                rgb = rgb.view(rays.shape[0], rays.shape[1], -1, 3)
                sigma = sigma.view(rays.shape[0], rays.shape[1], -1, 1)
                if last_rgb is not None:
                    rgb = torch.cat([rgb[:,:,:-1,:3].clone(), last_rgb[:,:,None,:]], dim=2)

                rgbs_coarse, _, weights_coarse = fancy_integration(torch.cat([rgb, sigma], dim=-1),
                                                    z_vals_coarse,
                                                    white_back=white_bg,
                                                    last_back=last_back,
                                                    clamp_mode=clamp_mode,
                                                    noise_std=noise_std)
                # if last_back:
                alphas_coarse = torch.sum(weights_coarse[...,:-1], dim=-1, keepdim=True) # ignore the last weight, which is the background
                # else:
                #     alphas_coarse = torch.sum(weights_coarse, dim=-1, keepdim=True) # ignore the last weight, which is the background
                output['rgbs_coarse'] = rgbs_coarse
                output['alphas_coarse'] = alphas_coarse
            # if no nerf_coarse model, then use the same nerf model to sample fine points
            # when running this, no grad for the nerf model
            else:
                with torch.no_grad():
                    rgbs, sigma = nerf_model(xyz_coarse, batch, return_correction=False, use_coarse=False)
                    rgbs = rgbs.view(rays.shape[0], rays.shape[1], -1, 3)
                    sigma = sigma.view(rays.shape[0], rays.shape[1], -1, 1)
                    _, _, weights_coarse = fancy_integration(torch.cat([rgbs, sigma], dim=-1),
                                                        z_vals_coarse,
                                                        white_back=white_bg,
                                                        last_back=last_back,
                                                        clamp_mode=clamp_mode,
                                                        noise_std=noise_std)
            z_vals_coarse = z_vals_coarse.reshape(-1, self.n_coarse_points)
            weights_coarse = weights_coarse.reshape(-1, self.n_coarse_points)
            fine_xyz, z_vals = self.sample_points(rays, n_points=self.n_points, n_coarse_points=self.n_coarse_points,
                                                perturb=perturb, z_coarse=z_vals_coarse, weights=weights_coarse)
        else:
            fine_xyz = xyz_coarse
            z_vals = z_vals_coarse
        return fine_xyz, z_vals, output

# from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering
class RayMarchingSampler(nn.Module):
    """ Ray marching sampling used in InstantNGP paper
    use nerfacc API
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, n_points = 128, n_update_steps=10, aabb=[-2,-2,-2,2,2,2]):
        super().__init__()
        self.n_points = n_points
        self.n_update_steps = n_update_steps
        # init an occupancy grid
        # density grid
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=aabb,
            resolution=128,
            contraction_type=ContractionType.AABB,
            )
        
    def to(self, device):
        self.occupancy_grid.to(device)
        return self
    
    def forward(self, nerf_model, batch=None, rays=None, stratified=True):
        device = rays.device
        batch_size = rays.shape[0]
        rays = rays.reshape(-1, rays.shape[-1])
        center = rays[..., :3]
        direction = rays[..., 3:6]
        near = rays[..., 6]
        far = rays[..., 7]
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = center[ray_indices]
            t_dirs = direction[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            positions = positions[None,...]
            return nerf_model.query_density(positions, condition=batch)
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = center[ray_indices]
            t_dirs = direction[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            positions = positions[None,...]
            return nerf_model(positions, condition=batch)
        
        # import ipdb; ipdb.set_trace()
        #-- run ray marching
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o=center,
            rays_d=direction,
            # scene_aabb=scene_aabb,
            t_min=near,
            t_max=far,
            grid=self.occupancy_grid,
            sigma_fn=sigma_fn,
            render_step_size=1e-2,
            stratified=stratified,
        )
        #-- update occupancy grid 
        import ipdb; ipdb.set_trace()
        return xyz, z_vals    

# vanilla volume rendering
class VolumeRenderer(nn.Module):
    """ vanilla volume rendering
        Args:
            _type_: _description_
        Returns:
            _type_: _description_
    """
    def __init__(self,
                 image_size: int = 512, # image size
                 near: float = -1.5, # near plane
                 far: float = 1.5, # far plane
                 sample_type: str = 'coarse', # sampling helper
                 n_rays: int = 1024 * 32, # number of rays
                 n_rays_patch_size: int = 0, # number of rays, sampling in patch
                 n_coarse_points: int = 64, # number of coarse points on each ray
                 n_points: int = 128, # number of points on each ray
                 test_n_rays: int = 1024 * 32, # number of rays for testing
                 white_bg: bool = False, # whether to use white background
                 depth_std: float = 0.0, # std for depth noise
                 dist_thresh: float=10.0, # distance threshold for deform
                 ) -> None:
        super().__init__()
        # init volume rendering
        self.sample_type = sample_type
        self.image_size = image_size
        self.n_rays = n_rays
        self.n_rays_patch_size = n_rays_patch_size
        self.test_n_rays = test_n_rays
        self.white_bg = white_bg
        self.depth_std = depth_std
        self.dist_thresh = dist_thresh
        
        # init rays in the image space
        self.near = near
        self.far = far
        pixel_rays = generate_image_rays(self.image_size, self.image_size, near, far)
        self.register_buffer('pixel_rays', pixel_rays)
        
        # init point sampler
        if self.sample_type == 'coarse':
            self.sampler = VanillaSampler(n_points, n_coarse_points)
        elif self.sample_type == 'raymarching':
            self.sampler = RayMarchingSampler(n_points)
        elif self.sample_type == 'mesh':
            # train a coarse mesh to help sampling
            self.sampler = MeshSampler(n_points)
    
    @torch.no_grad()
    def sample_rays(self, 
                    batch: dict,
                    random_sample: bool = False,
                    mesh_info: dict = None,
                    ):
        """ Sample rays from batch, also process the batch data
        Args:>
            batch: batch of data, e.g. images, when sampling the rays, also sample rgbs color from the image for the loss. 
                    e.g. camera pose, process the ray origin and direction in the world space.
                    e.g. background, set the ray far 
            random_sample: whether to sample rays randomly
                sample_chunk: sample rays in chunks, if None, sample all rays in the batch
                sample_patch_size: sample rays in patches (size x size), if not zero, sample rays in patches, otherwise sample rays in the whole image. 
                            This is to recover rays in a image patch, e.g. for the loss of the patch, perceptual loss. 
        """
        batch_size = batch['cam'].shape[0]
        rays = self.pixel_rays.expand(batch_size, -1, -1, -1)
        device = batch['cam'].device
        #-- hybrid rendering
        # 0. set the far as the mesh surface point
        # 1. set the mesh and background color in the far
        if mesh_info is not None and 'mesh_vis_image' in mesh_info and 'mesh_depth_image' in mesh_info:
            vis_image = mesh_info['mesh_vis_image'].detach()
            depth_image = mesh_info['mesh_depth_image'].detach().squeeze()
            cam_mean = batch['cam'][:,0].mean()
            far = depth_image  + self.depth_std*cam_mean/5.86 ## TODO 
            vis = vis_image.clone().squeeze(1)
            rays = rays.clone()
            rays_far = far.clone() * vis + rays.clone()[:, :, :, 7] * (1 - vis)
            rays[:, :, :, 7] = rays_far 
            # TODO: set the mesh color in the far
        #-- ray sampling, to save memory
        
        ## random rays for each image
        if random_sample:
            coords = torch.stack(
                torch.meshgrid(torch.linspace(0, self.image_size - 1, self.image_size),
                                torch.linspace(0, self.image_size - 1, self.image_size)), -1)
            n_rays = self.n_rays
            coords = torch.reshape(coords, [-1, 2]).long().to(device)
            select_inds = torch.randperm(coords.shape[0])[:n_rays]
            coords = coords[select_inds]
            
            if self.n_rays_patch_size > 0:
                half_size = self.n_rays_patch_size // 2
                while True:
                    tmp_mask = torch.zeros([self.image_size, self.image_size]).to(device)
                    center = torch.randint(low=half_size * 2,
                                            high=self.image_size - half_size * 2,
                                            size=(2,))
                    center = [self.image_size//2, self.image_size//2]
                    # import ipdb; ipdb.set_trace()
                    tmp_mask[center[0] - half_size:center[0] + half_size,
                                center[1] - half_size:center[1] + half_size] = 1.
                    inds = torch.nonzero(tmp_mask)
                    gt_mask = batch['mask']
                    patch_mask = gt_mask[:, :, inds[:, 0], inds[:, 1]]
                    if patch_mask.sum() > self.n_rays_patch_size**2 * batch_size / 3.:
                        break
                coords[:self.n_rays_patch_size**2] = inds
            rays = rays[:, coords[:, 0], coords[:, 1]]

        else:
            rays = rays.reshape(batch_size, -1, rays.shape[-1])
            coords = torch.stack(
                torch.meshgrid(torch.linspace(0, self.image_size - 1, self.image_size),
                                torch.linspace(0, self.image_size - 1, self.image_size)), -1)
            coords = torch.reshape(coords, [-1, 2]).long().to(device)
        return rays, coords
    
    def forward(self, 
                nerf_model: nn.Module,
                batch: dict = None,
                rays: torch.Tensor = None,
                coords: torch.Tensor = None,
                mesh_info: dict = None,
                render_normal: bool = False,
                train: bool = True,
                ) -> torch.Tensor: 
        """ Forward pass of the raymarching
        Args:
            nerf_model: the nerf model
            rays: [B, N_rays, 8], [origin, direction, near, far]
            nerf_model: nerf model
        """
        output = {}
        # if train:
        #     noise_std = max(0, 1.0 - batch['global_step']/400.)
        # else:
        noise_std = 0.
        
        #-- sample rays if not given, for training
        if rays is None:
            rays, coords = self.sample_rays(batch, random_sample=True, mesh_info=mesh_info)
        # sample rgb from mesh for this ray
        if mesh_info and 'mesh_image' in mesh_info:
            last_rgb = mesh_info['mesh_image'].permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
            last_rgb_valid = mesh_info['mesh_mask'].permute(0,2,3,1)[:, coords[:, 0], coords[:, 1]]
            if self.white_bg:
                last_rgb = last_rgb * last_rgb_valid + (1 - last_rgb_valid)
            last_back = True
        else:
            if self.white_bg:
                last_rgb = torch.ones_like(rays[:, :, :3])
            else:
                last_rgb = torch.zeros_like(rays[:, :, :3])
            last_back = True        
        # sample points
        x, z_vals, coarse_output = self.sampler(nerf_model, batch, rays, perturb=train,
                                 white_bg=self.white_bg, last_back=last_back,  # setting for coarse rendering
                                 clamp_mode='relu', noise_std=noise_std, last_rgb=last_rgb)
        output.update(coarse_output)
        
        #-- forward nerf model
        rgb, sigma, correction = nerf_model(x, batch, return_correction=True)
        rgb = rgb.view(rays.shape[0], rays.shape[1], -1, 3)
        sigma = sigma.view(rays.shape[0], rays.shape[1], -1, 1)
        if last_rgb is not None:
            rgb = torch.cat([rgb[:,:,:-1,:3].clone(), last_rgb[:,:,None,:]], dim=2)

        # filter, if point is too far, set the sigma to 0
        # TODO: move this to deform model
        if self.dist_thresh < 5.:
            cam_mean = batch['cam'][:,0].mean()
            curr_thresh = self.dist_thresh*cam_mean/5.86 ## TODO
            x_dist = nerf_model.deform(x, batch, return_dist=True)
            x_valid = torch.lt(x_dist, curr_thresh).float().squeeze()
            # x_valid = x_valid.reshape(batch_size, -1, 1)
            x_valid = x_valid.view(sigma.shape)
            sigma = sigma * x_valid
        #-- integrate the output along the ray
        rgbs, depths, weights = fancy_integration(torch.cat([rgb, sigma], dim=-1),
                                                  z_vals,
                                                  white_back=self.white_bg,
                                                  last_back=last_back,
                                                  clamp_mode='relu',
                                                  noise_std=noise_std)
        # if last_back:
        alphas = torch.sum(weights[...,:-1], dim=-1, keepdim=True) # ignore the last weight, which is the background
        # else:
        #     alphas = torch.sum(weights, dim=-1, keepdim=True) # ignore the last weight, which is the background
        output.update({
            'rgbs': rgbs,
            'depths': depths,
            'alphas': alphas,
            'weights': weights,
            'coords': coords, # record coords for computing losses
            'correction': correction,
        })
        if render_normal:
            x, xyz_transform = nerf_model.deform(x, batch, return_transform=True)
            normals = nerf_model.query_normal(x).detach()
            # transform from canonical space to image space
            if True:
                curr_transform = torch.inverse(xyz_transform.detach())
                flip_transform = torch.zeros_like(curr_transform)
                flip_transform[:, :, 0, 0] = -1.
                flip_transform[:, :, 1, 1] = 1.
                flip_transform[:, :, 2, 2] = 1.
                flip_transform[:, :, 2, 2] = 1.
                normals = util.batch_transform(curr_transform, normals)
                normals = util.batch_transform(flip_transform, normals)
                # normal = F.normalize(normal, p=2, dim=-1, eps=1e-6)            
            normals = normals.view(normals.shape[0], rays.shape[1], -1, 3)
            normals = torch.sum(weights[...,None] * normals, -2)
            normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
            output['normals'] = normals
        return output
    
    @torch.no_grad()
    def forward_image(self, nerf_model, batch, mesh_info=None, render_normal=False):
        """ generate image for test
        """
        output = {}
        if mesh_info is not None:
            batch['offset'] = mesh_info.get('mesh_offset', None)
        rays, coords = self.sample_rays(batch, random_sample=False, mesh_info=mesh_info)
        batch_size = rays.shape[0]
        chunk = self.test_n_rays
        for i in range(self.image_size**2 // chunk):
            chunk_idx = list(range(chunk * i, chunk * (i + 1)))
            rays_chunk = rays[:, chunk_idx]
            coords_chunk = coords[chunk_idx]
            output_chunk = self.forward(nerf_model, batch, rays_chunk, coords=coords_chunk, mesh_info=mesh_info, 
                                        render_normal=render_normal,
                                        train=False)
            # combine the output
            for k in ['rgbs', 'depths', 'alphas', 'normals', 'alpha_last', 'rgbs_coarse']:
                if k not in output_chunk:
                    continue
                if k in output:
                    output[k] = torch.cat([output[k], output_chunk[k]], dim=1)
                else:
                    output[k] = output_chunk[k]
        # convert to image format
        for k, v in output.items():
            output[k] = v.reshape(batch_size, self.image_size, self.image_size, -1).permute(0, 3, 1, 2)   
        output['nerf_image'] = output['rgbs']
        output['nerf_mask'] = output['alphas']
        if 'normals' in output:
            output['normal_image'] = output['normals']
        if 'rgbs_coarse' in output:
            output['nerf_coarse_image'] = output['rgbs_coarse']
        return output