"""
Ref: https://github.com/sxyu/pixel-nerf/blob/master/render/nerf.py
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl

Nerf: https://www.matthewtancik.com/nerf
Given: image, camera, MLP model
Nerf rendering:
1. camera => rays (origins, directions, near, far)
    ray for each pixel in images space
2. sampling point: k
3. composite


"""
import ipdb
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from . import util
from torch.autograd import Variable

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    with torch.no_grad():
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, allow_unused=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in[1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0]) 
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

def jacobian_simple(y, x):
    return torch.autograd.grad([y.sum()], [x], create_graph=True, retain_graph=True, only_inputs=True)[0]
    
def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224, raster_settings=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        self.sigma = 1e-4
        if raster_settings is None:
            raster_settings = {
                'image_size': image_size,
                'blur_radius': 0.0,
                'faces_per_pixel': 1,
                'bin_size': None,
                'max_faces_per_bin':  None,
                'perspective_correct': False, #True,
                'clip_barycentric_coords': True,
                'gamma': 1e-4
            }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, blur_radius=None, attributes=None, soft=False, sigma=1e-8, faces_per_pixel=1, gamma=1e-4):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        # translate z 
        # z_min = fixed_vertices[:,:,-1].min()
        # fixed_vertices[:,:,-1] = fixed_vertices[:,:,-1] #- z_min + 2
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        if blur_radius is None:
            blur_radius = self.raster_settings.blur_radius
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
            # clip_barycentric_coords=raster_settings.clip_barycentric_coords,
        )
        # import ipdb; ipdb.set_trace()
        vismask = (pix_to_face > -1).float().squeeze(-1)
        depth = zbuf.squeeze(-1)
        
        if soft:
            from pytorch3d.renderer.blending import _sigmoid_alpha
            colors = torch.ones_like(bary_coords)
            N, H, W, K = pix_to_face.shape
            pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
            pixel_colors[..., :3] = colors[..., 0, :]
            alpha = _sigmoid_alpha(dists, pix_to_face, sigma)
            pixel_colors[..., 3] = alpha
            pixel_colors = pixel_colors.permute(0,3,1,2)
            return pixel_colors

        if attributes is None:
            return depth, vismask
        else:
            vismask = (pix_to_face > -1).float()
            D = attributes.shape[-1]
            attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
            N, H, W, K, _ = bary_coords.shape
            mask = pix_to_face == -1
            pix_to_face = pix_to_face.clone()
            pix_to_face[mask] = 0
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
            pixel_vals[mask] = 0  # Replace masked values in output.
            pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
            pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
            return pixel_vals

    def render_shape(self, vertices, faces, images=None, lights=None, blur_radius=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        transformed_vertices = vertices.clone()
        # set lighting
        # if lights is None:
        #     light_positions = torch.tensor(
        #         [
        #         [-1,1,1],
        #         [1,1,1],
        #         [-1,-1,1],
        #         [1,-1,1],
        #         [0,0,1]
        #         ]
        #     )[None,:,:].expand(batch_size, -1, -1).float()
        #     light_intensities = torch.ones_like(light_positions).float()*1.7
        #     lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        if lights is None:
            light_positions = torch.tensor(
                [
                [-5, 5, -5],
                [5, 5, -5],
                [-5, -5, -5],
                [5, -5, -5],
                [0, 0, -5],
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()

            light_intensities = torch.ones_like(light_positions).float()*1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
            
        # transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, faces)
        normals = util.vertex_normals(vertices,faces); face_normals = util.face_vertices(normals, faces)
        transformed_normals = util.vertex_normals(transformed_vertices, faces); transformed_face_normals = util.face_vertices(transformed_normals, faces)
        face_colors = torch.ones_like(face_vertices)*180/255.
        attributes = torch.cat([face_colors, 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals], 
                        -1)
        # rasterize
        rendering = self.forward(transformed_vertices, faces, attributes=attributes, blur_radius=blur_radius)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.ones_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        return shape_images
    
    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)
    
class NeRFRenderer(nn.Module):
    def __init__(
        self,
        n_coarse=64,
        n_fine=0,
        n_fine_depth=0,
        # noise_std=0.0,
        noise_std=0.0,
        depth_std=0.01,
        chunk=100000,
        white_bg=True,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
        training=True,
        depth_sample=False,
        perturb=0., # factor to perturb the sampling position on the ray (for coarse model only)
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.chunk = chunk
        self.white_bkgd = chunk
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
            
        self.using_fine = n_fine > 0
        self.training = training
        self.white_bg = white_bg
        self.depth_sample = depth_sample
        self.perturb = perturb

    def sample_coarse(self, rays, n_coarse=None):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        if n_coarse is None:
            n_coarse = self.n_coarse
        near, far = rays['near'], rays['far']  # (B, 1)
        device = near.device; B = near.shape[0]

        step = 1.0 / n_coarse
        # import ipdb; ipdb.set_trace()
        z_steps = torch.linspace(0, 1, n_coarse, device=device)  # (Kc)
        # z_steps = torch.linspace(0, 1 - step, n_coarse, device=device)  # (Kc)
        # z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps = z_steps.unsqueeze(0).expand(B, -1)  # (B, Kc)
        ## TODO: + random noise
        if not self.lindisp: # use linear sampling in depth space
            z_vals = near * (1-z_steps) + far * z_steps
        else: # use linear sampling in disparity space
            z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

        if self.perturb > 0: # perturb sampling depths (z_vals)
            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
            lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
            
            perturb_rand = self.perturb * torch.rand_like(z_vals)
            z_vals_perturbed = lower + (upper - lower) * perturb_rand
            z_vals[1:-1] = z_vals_perturbed[1:-1]
        return z_vals

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays['direction'].device
        B = rays['direction'].shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        # near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        near, far = rays['near'], rays['far']  # (B, 1)

        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth, n_sample=None):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        if n_sample is None:
            n_sample = self.n_fine_depth
        # import ipdb; ipdb.set_trace()
        # z_samp = depth.repeat((1, n_sample))
        z_samp = depth.expand((-1, n_sample))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays['far']), rays['far'])
        return z_samp

    def composite(self, model, rays, z_samp, sample_mode='coarse', far=False, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape
            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K)
            # import ipdb; ipdb.set_trace()
            # if far:
            #     delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            # else:
            #     delta_inf = rays['far'] - z_samp[:, -1:]
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # infty (B, 1)
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)
            # (B, K, 3)

            points = rays['center'][:,None,:] + z_samp.unsqueeze(2) * rays['direction'][:,None,:] 
            # points = points.reshape(-1, 3)  # (B*K, 3)
            with torch.enable_grad():
            
                points.requires_grad_()

                
                rays[f'{sample_mode}_xyz'] = points
                
                out = model(rays, sample_mode=sample_mode)
                
                rgbs = out[..., :3]  # (B, K, 3)
                sigmas = out[..., 3]  # (B, K)
                # print('rgbs: ', rgbs.min(), rgbs.max())
                # print('sigmas: ', sigmas.min(), sigmas.max())
                # if rays['body_inside']:
                #     hard_alpha = rays['hard_alpha']
                #     # import ipdb; ipdb.set_trace()
                #     sigmas[:,-1] = 1e4*hard_alpha[:,0] + sigmas[:,-1]*(1-hard_alpha[:,0])
                sigmas_last = sigmas[:,-1]

                if self.training and self.noise_std > 0.0:
                    sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

                # compute the gradients in log space of the alphas, for NV TV occupancy regularizer
                alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
                deltas = None
                # sigmas = None
                alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-8], -1)  # (B, K+1) = [1, a1, a2, ...]

                T = torch.cumprod(alphas_shifted, -1)  # (B)
                weights = alphas * T[:, :-1]  # (B, K)
                # import ipdb; ipdb.set_trace()
                alphas = None
                alphas_shifted = None

                rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
                # import ipdb; ipdb.set_trace()
                depth_final = torch.sum(weights * z_samp, -1)  # (B)

                ## normals final
                # import ipdb; ipdb.set_trace()
                # bjac_chunk = jacobian_simple(sigmas, points) # [B, 64, 3]
                # normal = F.normalize(bjac_chunk, p=2, dim=2)
                # normal_final = torch.sum(weights[:,:,None]*normal, axis=-2) 

            # bjac_chunk = tf.reshape(bjac_chunk, (bjac_chunk.shape[0], 3)) # safe squeezing
            # normal_chunk = -tf.linalg.l2_normalize(bjac_chunk, axis=1)
            # Accumulate samples into expected depth and normals
            # weights = model.accumulate_sigma(sigma, z, rayd) # (n_rays, n_samples)
            # occu = tf.reduce_sum(weights, -1) # (n_rays,)
            # # Estimated depth is expected distance
            # exp_depth = tf.reduce_sum(weights * z, axis=-1) # (n_rays,)
            # # Computed weighted normal along each ray
            # exp_normal = tf.reduce_sum(weights[:, :, None] * normal, axis=-2)
            normal_final = depth_final
            if self.white_bg:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
            
            return (
                weights,
                rgb_final,
                depth_final,
                sigmas_last,
                normal_final
            )

    def forward(
        self,
        model,
        rays,
        want_weights=False,
        n_coarse=None
    ):
        """
        :model nerf model, should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        for single-object point batch;
        or (SB, B, (r, g, b, sigma)) when called with (SB, B, (x, y, z)), for multi-object
        NeRF super-batch * per object point batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        for single-object OR (SB, B, 8) for super-batch
        :param want_weights if true, returns compositing weights (B, K)
        :return rgb_fine, depth_fine, rgb_coarse, depth_coarse [, weights_fine, weights_coarse]
        """
        with profiler.record_function("renderer_forward"):
            z_coarse = self.sample_coarse(rays, n_coarse)  # (B, Kc)
            coarse_composite = self.composite(
                model,
                rays,
                z_coarse,
                sample_mode='coarse'
            )
            # only output uncertainties for fine model
            weights, rgb, depth, sigmas_last, normal_last = coarse_composite
            outputs = {
                'rgb_coarse': rgb,
                'depth_coarse': depth,
                'weights_coarse': weights,
                'normal_coarse': normal_last
            }

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)

                fine_composite = self.composite(
                    model,
                    rays,
                    z_combine_sorted,
                    sample_mode='fine'
                )
                outputs['rgb_fine'] = fine_composite[1]
                outputs['depth_fine'] = fine_composite[2]
                outputs['weights_fine'] = fine_composite[0]
                outputs['sigmas_last'] = sigmas_last #fine_composite[-1]

            return outputs


    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            n_far=conf.get_int("n_far", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=white_bkgd,
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )
