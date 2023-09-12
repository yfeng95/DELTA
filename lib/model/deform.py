""" deformation field, e.g. backward skinning for body models
"""
import torch
from torch import nn
from pytorch3d.ops.knn import knn_points

from ..utils import util, camera_util

def smplx_lbsmap_top_k(
            lbs_weights: torch.Tensor, # [B, n_verts, n_joints], weights for each joint
            verts_transform: torch.Tensor, # [B, n_verts, 3, 3], rotation matrix for each vertex
            points,
            template_points,
            K=1,
            addition_info=None,
            return_neigh_idx=False):
    '''
    Args:  
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    ##
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std**2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs]  # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(
        -torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1)
        / weight_std2)  # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist)  # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(
        -1, keepdim=True)  # (bs, n_rays*K, k_neigh)

    xyz_neighbs_transform = util.batch_index_select(verts_transform,
                                                    neighbs)  # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) *
                              xyz_neighbs_transform,
                              dim=2)  # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2,
                         keepdim=True)  # (bs, n_rays*K, 1)
    if return_neigh_idx:
        return xyz_dist, xyz_transform, neighbs, xyz_neighbs_weight
    return xyz_dist, xyz_transform


class BackDeformField(nn.Module):
    """back deformation field"""
    def __init__(self,
            basemodel: nn.Module, 
            cam_mode: str = 'orth', # camera mode, orth (orthographic) or persp (perspective)
            deform_cond: str = None, # learn correction for the deformation field
            k_neigh: int = 6, # number of neighbors for knn
            ) -> None:
        super().__init__()
        self.k_neigh = k_neigh
        self.cam_mode = cam_mode
        # template mesh vertices from SMPLX
        self.base = basemodel
        # learn correction for the deformation field
        if deform_cond is not None:
            learn_correction = True
            self.learn_correction = learn_correction
        else:
            self.learn_correction = False
        if self.learn_correction:
            from .ngp import NGPNet
            init_aabb = [-1., -1., -1., 1., 1., 1.]
            self.correct_model = NGPNet(aabb=init_aabb, input_dim=3, cond_dim=3, output_dim=3, last_op=torch.tanh, scale=0.01,
                                    log2_hashmap_size=8, n_levels=8)
            
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, 
                x: torch.Tensor, 
                condition: dict, 
                return_transform: bool = False,
                return_neigh_idx: bool = False,
                return_corretion: bool = False,
                return_dist: bool = False,
                ):
        """ Given the deformed x, return the canonical x
        Args:
            x: deformed x (query point in the obervation space), [B, N, 3]
            condition: condition for the deformation field, includes:   
                cam: [B, 3], camera parameters, if orth, [scale, tx, ty], if persp, [focal, tx, ty]
                pose: [B, n_joints, 3, 3], rotation matrix
                
        """
        batch_size = x.shape[0]
        device = x.device
        # reproject x into base model (smplx) space (inverse camera projection)
        posed_x = self.cam_project(x, condition['cam'], inv=True)
        
        # backward transformation from base model, only for surface point
        verts_transform, lbs_weights = self.base.backward_skinning(
                        full_pose = condition['full_pose'],
                        shape_params = condition['beta'], 
                        offset = condition.get('offset', None),
                        transl = condition.get('transl', None),
                    )
        # import ipdb; ipdb.set_trace()
        # use knn to get the transform for given x
        if 'posed_verts' not in condition:
            verts = self.base.verts[None, ...].repeat(batch_size, 1, 1)
            forward_verts_transform = self.base.forward_skinning(
                                    full_pose=condition['full_pose'], 
                                    shape_params=condition['beta'], 
                                    offset=condition.get('offset', None),
                                    )        
            posed_verts = util.batch_transform(forward_verts_transform, verts)
            # note:  the backward_skinning is the inverse of forward_skinning (without offset)
            # back_verts_transform = torch.inverse(forward_verts_transform)
        surface_verts = posed_verts
        if return_neigh_idx or self.learn_correction:
            x_dist, x_transform, neighbs, xyz_neighbs_weight = smplx_lbsmap_top_k(lbs_weights,
                                                    verts_transform,
                                                    posed_x,
                                                    surface_verts,
                                                    K=self.k_neigh,
                                                    return_neigh_idx=True)
            xyz_neighbs_info = util.batch_index_select(posed_verts, neighbs).detach()
            xyz_posed_verts = torch.sum(xyz_neighbs_weight.unsqueeze(-1) * xyz_neighbs_info, dim=2)
            
        else:
            x_dist, x_transform = smplx_lbsmap_top_k(lbs_weights,
                                                    verts_transform,
                                                    posed_x,
                                                    surface_verts,
                                                    K=self.k_neigh)
        cano_x = util.batch_transform(x_transform, posed_x)
        if self.learn_correction:
            # learn correction for the deformation field
            # note: the correction is in the canonical space
            correction = self.correct_model(cano_x, cond=xyz_posed_verts)
            cano_x = cano_x + correction
        else:
            correction = None
        if return_neigh_idx:
            return cano_x, xyz_posed_verts, correction
        if return_dist:
            return x_dist
            
        if return_transform:
            return cano_x, x_transform
        
        if return_corretion and self.learn_correction:
            return cano_x, correction
        return cano_x
    
    def cam_project(self, points, cam, inv=False):
        if self.cam_mode == 'orth':
            if inv:
                proj_points = camera_util.batch_orth_proj_inv(points, cam.squeeze(-1))
            else:
                proj_points = camera_util.batch_orth_proj(points, cam.squeeze(-1))
        else:
            # ['fx', 'fy', 'cx', 'cy', 'quat', 'T']
            # cam_new[:,[0,-1]] /= 46.
            if inv:
                proj_points = camera_util.perspective_project_inv(points,
                                                                  focal=cam[:, 0].mean(),
                                                                  transl=cam[:, 1:])
            else:
                proj_points = camera_util.perspective_project(points,
                                                              focal=cam[:, 0].mean(),
                                                              transl=cam[:, 1:])
        return proj_points
    
    