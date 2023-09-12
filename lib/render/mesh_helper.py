'''
rasterization
basic rasterize
render shape (for visualization)
render texture (need uv information)
'''

import torch
from torch import nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
import numpy as np
from ..utils import util

pi = np.pi
constant_factor = torch.tensor(
            [
                pi * (np.sqrt(1. / pi) / 2.),
                # L1, m-1
                (2 * pi / 3.) * (np.sqrt(3. / pi) / 2.0),
                # L1, m
                (2 * pi / 3.) * (np.sqrt(3. / pi) / 2.0),
                (2 * pi / 3.) * (np.sqrt(3. / pi) / 2.0),
                # L2, m-2
                (pi / 4.) * np.sqrt(15.0 / pi) / 2.0,
                # N2_1 (y*z)
                (pi / 4.) * (np.sqrt(15.0 / pi) / 2.0),
                # N2_0 (2*z^2 - x^2 - y^2))
                (pi / 4.) * (np.sqrt(5.0 / pi) / 4.0),
                # N2_1 (z*x)
                (pi / 4.) * (np.sqrt(15.0 / pi) / 2.0),
                # N2_2 (x^2-y^2)
                (pi / 4.) * (np.sqrt(15.0 / pi) / 2.0)
                ]
        ).float()

def add_SHlight(normals, sh_coeff, relu=True):
    '''
    https://github.com/r00tman/NeRF-OSR/blob/d8c8f6ae615e7e5455a30b75f3710832bb244fda/sph_util.py#L29
    https://github.com/HavenFeng/TRUST/blob/5e44ab5e26ab976902ae53798cbcd35d4803d341/lib/renderer.py#L273
    normals: [bz, nv, 3]
    sh_coeff: [bz, 9, 3]
    '''
    # Renormalize the normals in case they have been interpolated.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    curr_constant_factor = constant_factor.to(normals.device)
    x = normals[...,0]
    y = normals[...,1]
    z = normals[...,2]
    
    sh = torch.stack([
            x*0. + 1, 
            y,
            z, 
            x, 
            x*y, 
            y*z, 
            2*(z**2 - x**2 - y**2), 
            z*x, 
            x**2 - y**2
            ], 
            -1) # [bz, nv, 9]
    sh = sh*curr_constant_factor[None,None,:] #[bz, n, 9]
    shading = sh_coeff[:,None,:,:]*sh[:,:,:,None] # [bz, n, 9, 3]  
    shading_sum = torch.sum(shading, dim=2) # [bz, n, 3]
    if relu:
        shading_sum = F.relu(shading_sum)
    return shading_sum

def add_SHlight_implicit(sh, sh_coeff, relu=True):
    '''
    https://github.com/r00tman/NeRF-OSR/blob/d8c8f6ae615e7e5455a30b75f3710832bb244fda/sph_util.py#L29
    https://github.com/HavenFeng/TRUST/blob/5e44ab5e26ab976902ae53798cbcd35d4803d341/lib/renderer.py#L273
    sh: [bz, nv, 9]
    sh_coeff: [bz, 9, 3]
    '''
    # Renormalize the normals in case they have been interpolated.
    curr_constant_factor = constant_factor.to(sh.device)
    sh = sh*curr_constant_factor[None,None,:] #[bz, n, 9]
    shading = sh_coeff[:,None,:,:]*sh[:,:,:,None] # [bz, n, 9, 3]  
    shading_sum = torch.sum(shading, dim=2) # [bz, n, 3]
    if relu:
        shading_sum = F.relu(shading_sum)
    return shading_sum
    
def add_pointlight(normals, lights):
    '''
        vertices: [bz, nv, 3]
        lights: [bz, nlight, 6]
    returns:
        shading: [bz, nv, 3]
    '''
    light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
    directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
    # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
    shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
    return shading.mean(1)
    
def add_directionlight(normals, lights=None):
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
    
    # import ipdb; ipdb.set_trace()
    return shading.mean(1)


def pytorch3d_rasterize(vertices, faces, image_size, attributes=None, 
                        soft=False, blur_radius=0.0, sigma=1e-8, faces_per_pixel=1, gamma=1e-4, 
                        perspective_correct=False, clip_barycentric_coords=True):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        # import ipdb; ipdb.set_trace()
        # pytorch3d rasterize
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=perspective_correct,
            clip_barycentric_coords=clip_barycentric_coords,
            # max_faces_per_bin = faces.shape[1],
            bin_size = 0
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

# For visualization
def render_shape(vertices, faces, image_size, background=None, lights=None, blur_radius=0., shift=True, colors=None):
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
    if shift:
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10

    # Attributes
    face_vertices = util.face_vertices(vertices, faces)
    normals = util.vertex_normals(vertices,faces); face_normals = util.face_vertices(normals, faces)
    transformed_normals = util.vertex_normals(transformed_vertices, faces); transformed_face_normals = util.face_vertices(transformed_normals, faces)
    if colors is None:
        face_colors = torch.ones_like(face_vertices)*180/255.
    else:
        face_colors = util.face_vertices(colors, faces)

    attributes = torch.cat([face_colors, 
                    transformed_face_normals.detach(), 
                    face_vertices.detach(), 
                    face_normals], 
                    -1)
    # rasterize
    rendering = pytorch3d_rasterize(transformed_vertices, faces, image_size=image_size, attributes=attributes, blur_radius=blur_radius)

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

    shading = add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
    shaded_images = albedo_images*shading_images

    # alpha_images = alpha_images*pos_mask
    if background is None:
        shape_images = shaded_images*alpha_images + torch.ones_like(shaded_images).to(vertices.device)*(1-alpha_images)
    else:
        shape_images = shaded_images *alpha_images + background*(1-alpha_images)
    return shape_images
    

def render_texture(transformed_vertices, faces, image_size,
                    vertices, albedos, 
                    uv_faces, uv_coords,
                    lights=None, light_type='point'):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        face_uvcoords = util.face_vertices(uv_coords, uv_faces)
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        # import ipdb; ipdb.set_trace()
        # normalize to 0, 100
        # transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        # transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10

        # import ipdb; ipdb.set_trace()
        # attributes
        face_vertices = util.face_vertices(vertices, faces)
        normals = util.vertex_normals(vertices, faces); face_normals = util.face_vertices(normals, faces)
        transformed_normals = util.vertex_normals(transformed_vertices, faces); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([face_uvcoords, 
                                transformed_face_normals.detach(), 
                                face_vertices.detach(), 
                                face_normals], 
                                -1)
        # rasterize
        rendering = pytorch3d_rasterize(transformed_vertices, faces, attributes)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        outputs = {
            'images': images*alpha_images,
            'albedo_images': albedo_images*alpha_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs