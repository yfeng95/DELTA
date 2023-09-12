import torch
from torch import nn
import numpy as np

from .model.smplx import SMPLX
from .model.siren import GeoSIREN, ParamNet
from .model.deform import BackDeformField
from .render.mesh_render import Pytorch3dMeshRenderer
from .render.mesh_helper import add_SHlight
from .render.volume_render import VolumeRenderer
from .utils import rotation_converter, util, camera_util

class MeshAvatar(nn.Module):
    """ Mesh based avatar in canonical space
        Args:
            basemodel (nn.Module): base model for the avatar, here we use SMPLX model for expressive body, could also be other models, e.g. SMPL only for body or FLAME
            cfg (dict): configuration for the mesh avatar
        """
    def __init__(
        self, 
        basemodel: nn.Module, 
        cfg: dict=None,
        image_size: int=512,
        cam_mode: str='orth',
    ) -> None:
        super().__init__()
        if cfg is None:
            from .core.config import get_cfg_defaults
            cfg = get_cfg_defaults().mesh
        # template mesh vertices from SMPLX
        self.base = basemodel
        self.cfg = cfg
        self.cam_mode = cam_mode
        
        # base surface points: xyz and faces from model template
        self.part_idx_dict = self.base.get_part_index()
        
        # geo model: xyz in mesh surface -> z offset (can be conditioned or not)
        if self.cfg.geo_cond_type is None: # no conditioning, no posed dependent
            cond_dim = 0
        elif self.cfg.geo_cond_type == 'posed_verts': # conditioned on posed verts
            cond_dim = 3
        if self.cfg.geo_net == 'siren':
            self.geo_model = GeoSIREN(input_dim=self.cfg.geo_in_dim, z_dim=1, hidden_dim=128, output_dim=self.cfg.geo_out_dim, 
                                    last_op=torch.tanh, scale=self.cfg.geo_scale)
        elif self.cfg.geo_net == 'param':
            self.geo_model = ParamNet(size=(len(self.base.verts), self.cfg.geo_out_dim), init_way=torch.zeros,last_op=torch.tanh, scale=self.cfg.geo_scale)
        elif self.cfg.geo_net == 'ngp':
            from .model.ngp import NGPNet
            init_aabb = [-1., -1., -1., 1., 1., 1.]
            self.geo_model = NGPNet(aabb=init_aabb, input_dim=self.cfg.geo_in_dim, cond_dim=cond_dim, output_dim=self.cfg.geo_out_dim, last_op=torch.tanh, scale=self.cfg.geo_scale,
                                    log2_hashmap_size=8, n_levels=8)
        # color model: xyz in mesh surface -> color
        if self.cfg.color_cond_type is None: # no conditioning, no posed dependent
            cond_dim = 0
        elif self.cfg.color_cond_type == 'posed_verts': # conditioned on posed verts
            cond_dim = 3
        if self.cfg.color_net == 'siren':
            self.color_model = GeoSIREN(input_dim=3,
                                        z_dim=1,
                                        hidden_dim=128,
                                        output_dim=3,
                                        last_op=torch.sigmoid,
                                        scale=1.)
        elif self.cfg.color_net == 'param':
            self.color_model = ParamNet(size=(len(self.base.verts), 3), init_way=torch.zeros,last_op=torch.sigmoid, scale=1.)
        elif self.cfg.color_net == 'ngp':
            from .model.ngp import NGPNet
            init_aabb = [-1., -1., -1., 1., 1., 1.]
            self.color_model = NGPNet(aabb=init_aabb, input_dim=3, cond_dim=cond_dim, output_dim=3, last_op=torch.sigmoid, scale=1.)
        
        # setup renderer
        self.render = Pytorch3dMeshRenderer(image_size=image_size)
    
    def to(self, device):
        self.base.to(device)
        self.geo_model.to(device)
        self.color_model.to(device)
        self.render.to(device)
        return self
    
    def cam_project(self, points, cam, inv=False):
        if self.cam_mode == 'orth':
            if inv:
                proj_points = util.batch_orth_proj_inv(points, cam.squeeze(-1))
            else:
                proj_points = util.batch_orth_proj(points, cam.squeeze(-1))
        else:
            if inv:
                proj_points = camera_util.perspective_project_inv(points,
                                                                  focal=cam[:, 0].mean(),
                                                                  transl=cam[:, 1:])
            else:
                proj_points = camera_util.perspective_project(points,
                                                              focal=cam[:, 0].mean(),
                                                              transl=cam[:, 1:])
        return proj_points
    
    def forward(self, beta, pose, cam, exp=None, lights=None,
                renderImage=False,
                renderMask=False,
                renderShape=False,
                renderDepth=False,
                background=None,
                clean_offset=False):
        """ for mesh avatar, given pose, return the mesh vertices and faces
        Args:
            pose (tensor): [B, 55, 3, 3] pose parameters in rotation matrix
        Returns:
            output (dict): outputs of the mesh model, including:
                verts (tensor): [B, N, 3] mesh vertices
                 
        """
        batch_size = cam.shape[0]
        device = cam.device
        output = {}
        
        #-- canonical mesh: fixed template
        base_verts = self.base.verts[None, ...].repeat(batch_size, 1, 1)
        faces = self.base.faces[None, ...].repeat(batch_size, 1, 1)
        # get transform from given beta and pose
        verts_transform = self.base.forward_skinning(
                                    full_pose=pose, 
                                    shape_params=beta, 
                                    exp=exp,
                                    offset=None
                                    )        
        #-- networks: geo and color
        # aleays query point in the template as input
        input = base_verts.detach().clone()
        # for mesh offset
        # condition: posing the template verts, use it as condition
        if self.cfg.geo_cond_type == 'posed_verts':
            base_posed_verts = util.batch_transform(verts_transform, base_verts).detach()
            cond = base_posed_verts
        else:
            cond = None
        verts_offset = self.geo_model(input, cond=cond)
        # set offset to zero
        if clean_offset:
            verts_offset[:] = 0.
        # for mesh color
        if self.cfg.color_cond_type == 'posed_verts':
            base_posed_verts = util.batch_transform(verts_transform, base_verts).detach()
            cond = base_posed_verts
        else:
            cond = None
        colors = self.color_model(input, cond=cond)
        
        #-- apply offset to the canonical mesh
        verts = base_verts + verts_offset
        #-- apply deformation: pose + exp
        posed_verts = util.batch_transform(verts_transform, verts)
        #-- apply camera transformation
        trans_verts = self.cam_project(posed_verts, cam)
        output['mesh_offset'] = verts_offset
        output['mesh_posed_verts'] = posed_verts
        output['mesh_trans_verts'] = trans_verts
        output['mesh_faces'] = faces
        output['mesh_colors'] = colors
        
        #-- render
        if renderShape:
            shape_image = self.render.render_shape(trans_verts, faces, background=background)
            output['shape_image'] = shape_image
        # render depth, mainly for nerf hybrid rendering
        if renderDepth:
            depth_image, vis_image = self.render.render_depth(trans_verts, faces)
            output['mesh_depth_image'] = depth_image
            output['mesh_vis_image'] = vis_image
        # render image, for photometric loss and visualization
        if renderImage:
            if self.cfg.use_light and lights is not None:
                # calculate normals 
                trans_verts_new = trans_verts.clone()
                trans_verts_new[:, :, 0] = -trans_verts_new[:, :, 0]
                normals = util.vertex_normals(trans_verts_new, faces)
                shadings = add_SHlight(normals, lights)
                lit_colors = colors * shadings
                attrs = torch.cat([colors, normals, shadings, lit_colors], dim=-1)
                render_out, _ = self.render(trans_verts, faces, attrs=attrs)
                output['mesh_image'] = render_out[:, 9:12]
                output['mesh_albedo_image'] = render_out[:, :3]
                output['mesh_normal_image'] = render_out[:, 3:6]
                output['mesh_shading_image'] = render_out[:, 6:9]
            else:                
                image, _ = self.render(trans_verts, faces, attrs=colors)
                output['mesh_image'] = image
            
        # render silhouette mask, for silhouette loss
        if renderMask:
            mask_image = self.render.render_silhouette(trans_verts, faces)
            output['mesh_mask'] = mask_image
        return output


class NeRFAvatar(nn.Module):
    """ NeRf based avatar in canonical space
        Args:
            basemodel: base model for the avatar, here we use SMPLX model for expressive body, could also be other models, e.g. SMPL only for body or FLAME
            cfg: config file
            image_size: image size for rendering
        Returns:
            _type_: _description_
    NeRF based avatar contains,
    1. canonical model: includeing base mesh model, e.g. SMPLX, SMPL, FLAME and the nerf networks for geometry and color (could be mlp or ngp)
    2. deformation model: to defrom the obervation space to canonical space, could be knn or root-finding
    3. rendering model: how to sample rays and points. Depending on the point sampling way, we may need to train a nerf caorse model or a grid
    """
    def __init__(self,
                 basemodel,
                 cfg
                 ) -> None:
        super().__init__()
        self.cfg = cfg
        self.nerf_cond = cfg.nerf_cond
        #--- nerf model in canonical space
        if cfg.nerf_net == 'mlp':
            # from .model.mlp import VanillaNeRFRadianceField
            # self.nerf = VanillaNeRFRadianceField()
            from .model.nerf import NeRF
            self.nerf = NeRF(
                freqs_xyz=10,
                freqs_dir=4,
                use_view=False
            )
            if cfg.use_coarse_model:
                self.nerf_coarse = NeRF(
                    freqs_xyz=10,
                    freqs_dir=4,
                    use_view=False
                )
            else:
                self.nerf_coarse = None
        elif cfg.nerf_net == 'ngp':
            from .model.ngp import NGPradianceField
            ngp_aabb = cfg.ngp_aabb
            init_aabb = [-ngp_aabb, -ngp_aabb, -ngp_aabb, ngp_aabb, ngp_aabb, ngp_aabb] #torch.tensor([-2,-2.,-2.,2.,2.,2.], dtype=torch.float32)
            if cfg.nerf_cond is not None:
                use_viewdirs = True
            else:
                use_viewdirs = False
            self.nerf = NGPradianceField(aabb=init_aabb, unbounded=False, use_viewdirs=use_viewdirs, cond_type=cfg.nerf_cond, 
                                         log2_hashmap_size=19, n_levels=cfg.ngp_n_levels)
            if cfg.use_coarse_model:
                self.nerf_coarse = NGPradianceField(aabb=init_aabb, unbounded=False, use_viewdirs=use_viewdirs, cond_type=cfg.nerf_cond, 
                                         log2_hashmap_size=19, n_levels=cfg.ngp_n_levels)
            else:
                self.nerf_coarse = None
        
        # deform/warping field based on backward skinning
        self.deform_cond = cfg.deform_cond
        self.deform = BackDeformField(basemodel=basemodel, deform_cond=cfg.deform_cond,
                                    cam_mode = 'orth',
                                    k_neigh=cfg.k_neigh,
                                    )
        # rendering model
        self.render = VolumeRenderer(image_size=self.cfg.image_size, 
                                    near=self.cfg.near,
                                    far=self.cfg.far,
                                    n_rays=self.cfg.n_rays,
                                    n_rays_patch_size=self.cfg.n_rays_patch_size,
                                    n_coarse_points=self.cfg.n_coarse_points,
                                    n_points=self.cfg.n_points,
                                    test_n_rays=self.cfg.test_n_rays,
                                    sample_type=self.cfg.sample_type,
                                    white_bg=self.cfg.white_bg,
                                    depth_std=self.cfg.depth_std,
                                    dist_thresh=self.cfg.dist_thresh,
                                    )

    def query_opacity(self, x, condition, step_size):
        density = self.query_density(x, condition)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity
        
    def query_density(self, x, condition):
        """ Query density from the NeRF model
        """
        x = self.deform(x, condition)
        return self.nerf.query_density(x)
    
    @torch.cuda.amp.autocast(enabled=False)
    def query_normal(self, x, condition=None, delta=0.02, use_coarse=False):
        """ Query normal from the NeRF model
        """
        nerf_model = self.nerf if not use_coarse else self.nerf_coarse
        # x = self.deform(x, condition)
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # sigma = self.query_density(x, condition=condition)
            sigma = nerf_model.query_density(x)
            alpha = 1 - torch.exp(-delta * torch.relu(sigma))
            normal = torch.autograd.grad(
                outputs=alpha,
                inputs=x,
                grad_outputs=torch.ones_like(alpha, requires_grad=False, device=alpha.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return normal
    
    def canonical_normal(self, use_coarse=False):
        epsilon = 0.01
        dis_threshold = 0.01
        points = self.deform.base.verts.clone().detach()
        # points = self.verts.detach()
        points += torch.randn_like(points) * dis_threshold * 0.5
        points_neighbs = points + torch.randn_like(points) * epsilon
        points_normal = self.query_normal(points, use_coarse=use_coarse)
        points_neighbs_normal = self.query_normal(points_neighbs)
        points_normal = points_normal / (torch.norm(points_normal, p=2, dim=-1, keepdim=True) +
                                         1e-5)
        points_neighbs_normal = points_neighbs_normal / (
            torch.norm(points_neighbs_normal, p=2, dim=-1, keepdim=True) + 1e-5)
        return points_normal, points_neighbs_normal
    
    def forward(self, 
                x: torch.Tensor, # [B, np, 3]
                condition: dict = None,  # dict of conditions, e.g. pose, camera, etc.
                return_correction: bool = False,
                use_coarse: bool = False,
                ):
        if self.nerf_cond == 'posed_verts' or self.deform_cond == 'posed_verts':
            x, xyz_posed_verts, correction = self.deform(x, condition, return_neigh_idx=True)
        else:
            x = self.deform(x, condition)
            correction = None
        if self.nerf_cond == 'neck_pose': 
            neck_pose = condition['full_pose'][:, 22].detach()
            neck_pose = rotation_converter.batch_matrix2axis(neck_pose)
            cond = neck_pose[:,None,:].expand(-1, x.shape[1], -1)
        elif self.nerf_cond == 'posed_verts':
            cond = xyz_posed_verts.detach()
        else:
            cond = None
        if self.nerf_coarse is not None and use_coarse:
            rgb, sigma = self.nerf_coarse(x, cond)
        else:
            rgb, sigma = self.nerf(x, cond)
        if return_correction:
            return rgb, sigma, correction
        else:
            return self.nerf(x, cond)
        
    def forward_rays(self, batch, mesh_info=None):
        output = self.render(self, batch, rays=None, mesh_info=mesh_info, train=True)
        return output
    
    def forward_image(self, batch, mesh_info=None, render_normal=False):
        ''' render image
        '''
        output = self.render.forward_image(self, batch, mesh_info, render_normal)
        return output
        
        
        
        
    
class DELTA(nn.Module):
    """ DELTA model
    Args:
        cfg (dict): configuration for DELTA model
        init_beta (torch.Tensor, optional): [1, n_shape] initial beta for the model. Defaults to None.
    """
    def __init__(self, cfg, init_beta=None):
        super(DELTA, self).__init__()
        self.cfg = cfg
        self.cfg.nerf.white_bg = self.cfg.dataset.white_bg
        self.cfg.nerf.image_size = self.cfg.dataset.image_size

        #-- define canonical space, then setup avatars
        pose = torch.zeros([55, 3], dtype=torch.float32)  # 55
        angle = 30 * np.pi / 180.
        pose[1, 2] = angle
        pose[2, 2] = -angle
        canonical_pose_matrix = rotation_converter.batch_euler2matrix(pose)
        self.canonical_pose = canonical_pose_matrix
        
        #-- setup avatar model, in canonical space
        if init_beta is None:
            beta = torch.zeros([1, self.cfg.model.n_shape], dtype=torch.float32)
        else:
            beta = init_beta.reshape([1, -1])
        self.register_parameter('beta', torch.nn.Parameter(beta))
        # setup base model (SMPLX)
        basemodel = SMPLX(self.cfg.model)
        # set canonical pose
        basemodel.set_canonical_space(canonical_pose_matrix)
        self.base = basemodel
        # mesh based avatar
        self.mesh = MeshAvatar(basemodel=basemodel, cfg=self.cfg.mesh)
        # nerf based avatar
        if self.cfg.use_nerf:
            self.nerf = NeRFAvatar(basemodel=basemodel, 
                                   cfg=self.cfg.nerf)
    def to(self, device):
            super().to(device)
            self.mesh.to(device)
            if self.cfg.use_nerf:
                self.nerf.to(device)
            return self
    
    def forward(self, batch, run_mesh=True, run_nerf=True, 
                render_image=False, render_normal=False, render_shape=False):
        batch_size = batch['cam'].shape[0]
        beta = self.beta.repeat(batch_size, 1)
        batch['beta'] = beta
        output = {}
        if self.cfg.use_mesh and run_mesh:
            mesh_output = self.mesh(beta, batch['full_pose'], batch['cam'], batch.get('exp', None), batch.get('light', None),
                                    renderImage=True,
                                    renderMask=True,
                                    renderShape=render_shape,
                                    renderDepth=self.cfg.use_nerf) # render depth only when nerf is used, for ray stopping
            output = {**output, **mesh_output}
        else:
            mesh_output = None
        if self.cfg.use_nerf and run_nerf:
            if render_image: # when testing, return image rendered by nerf
                nerf_output = self.nerf.forward_image(batch, mesh_info=mesh_output, render_normal=render_normal)
                output = {**output, **nerf_output}
            else: # when training, return the output of random sampled rays
                nerf_output = self.nerf.forward_rays(batch, mesh_info=mesh_output)
                output = {**output, **nerf_output}
        return output
    
    # for demo visualization
    def forward_vis(self, batch, returnMask=False):
        output = self.forward(batch, run_mesh=True, run_nerf=True, render_image=True)
        visout = {}
        # hybrid render
        batch_size = batch['cam'].shape[0]
        mesh_rendering = self.mesh(self.beta.repeat(batch_size, 1), 
                                    batch['full_pose'], 
                                    batch['cam'], 
                                    batch.get('exp', None),
                                    renderShape=True
                                    )
        shape_image = mesh_rendering['shape_image']
        if self.cfg.dataset.load_mouth:
            # import ipdb; ipdb.set_trace() 
            visout['mesh'] = shape_image
            mesh_image = output['mesh_image']
            mouth_mask = batch['mouth_mask']
            shape_image = shape_image*(1-mouth_mask) + mesh_image*mouth_mask

        if 'nerf_image' in output:
            nerf_mask = output['nerf_mask']
            nerf_image = output['nerf_image']
            if 'mesh_vis_image' in output:
                mesh_mask = output['mesh_vis_image']
                render_hybrid = nerf_image * nerf_mask + shape_image * (1 - nerf_mask)
                mask = (nerf_mask + mesh_mask).clamp(0, 1)
                visout['render'] = nerf_image*mask + (1-mask)
                visout['render_hybrid'] = render_hybrid  
                if returnMask:
                    visout['nerf_mask'] = nerf_mask
                    visout['mesh_mask'] = mesh_mask  
            else:
                visout['render'] = nerf_image  
                visout['shape'] = shape_image
        else:
            visout['render'] = shape_image
            
        return visout
        