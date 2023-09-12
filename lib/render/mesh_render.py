""" Rendering Functions
"""
import torch
from torch import nn
import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes

from .mesh_helper import render_shape, pytorch3d_rasterize, add_SHlight
from ..utils import util

# pytorch3d rasterization
class Pytorch3dMeshRenderer(nn.Module):
    """ render mesh using pytorch3d rasterization

    Args:
        image_size (int): rendering size
        faces (torch.Tensor, Optional): [1, n_faces, 3] face indices
    Returns:
        _type_: _description_
    """
    def __init__(self, image_size=512, faces=None) -> None:
        super().__init__()
        self.image_size = image_size
        self.faces = faces
        # -- silhouette mesh rendering for mask loss when training
        ## add orth camera for pytorch3d render function, since the API needs
        ## but the camera here has no effect on the vertices, so make sure the input verts is already in image space 
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros([1, 3])
        batch_size = 1
        cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(
                        R=R.expand(batch_size, -1, -1),
                        T=T.expand(batch_size, -1),
                        znear=0.0)
        blend_params = pytorch3d.renderer.BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = pytorch3d.renderer.RasterizationSettings(
                        image_size=self.image_size,
                        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                        faces_per_pixel=50,
                        bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = pytorch3d.renderer.MeshRenderer(
                        rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                        shader=pytorch3d.renderer.SoftSilhouetteShader(blend_params=blend_params))
    
    def to(self, device):
        # self.silhouette_renderer.to(device)
        self.silhouette_renderer.rasterizer.cameras.to(device)
        return self
    
    @torch.cuda.amp.autocast(enabled=False)
    def render_shape(self, trans_verts, faces=None, image_size=None, background=None):
        image_size = self.image_size if image_size is None else image_size
        trans_verts = trans_verts.clone().detach()
        trans_verts[:,:,2] = trans_verts[:,:,2] + 10
        shape_image = render_shape(vertices=trans_verts,
                                       faces=faces,
                                       image_size=image_size,
                                       background=background)
        return shape_image
    
    @torch.cuda.amp.autocast(enabled=False)
    def render_depth(self, trans_verts, faces=None, image_size=None, background=None):
        image_size = self.image_size if image_size is None else image_size
        depth_verts = trans_verts.clone().detach()
        depth_verts[..., -1] = depth_verts[..., -1] + 10.
        depth_image, vis_image = pytorch3d_rasterize(vertices=depth_verts,
                                                        faces=faces,
                                                        image_size=image_size,
                                                        blur_radius=0.)
        depth_image = (depth_image - 10.) * vis_image
        return depth_image, vis_image
    
    @torch.cuda.amp.autocast(enabled=False)
    def render_silhouette(self, trans_verts, faces=None, image_size=None):
        """ render silhouette using pytorch3d rasterization
        note that, for silhouette, we need to flip the xy axis of the vertices
        Args
        """
        image_size = self.image_size if image_size is None else image_size
        trans_verts_mask = trans_verts.clone()
        trans_verts_mask[:, :, :2] = -trans_verts_mask[:, :, :2]
        trans_verts_mask[:, :, 2] = trans_verts_mask[:, :, 2] + 10
        mesh = Meshes(verts=trans_verts_mask,
                      faces=faces
                     )
        silhouette = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:, 3:]
        return silhouette
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, trans_verts, faces=None, attrs = None, image_size=None):
        """ render mesh using pytorch3d rasterization
        Args
        """
        image_size = self.image_size if image_size is None else image_size
        trans_verts = trans_verts.clone()
        trans_verts[:,:,2] = trans_verts[:,:,2] + 10
        face_attrs = util.face_vertices(attrs, faces)
        image = pytorch3d_rasterize(trans_verts,
                                    faces,
                                    image_size=image_size,
                                    attributes=face_attrs)
        alpha_image = image[:, [-1]]
        return image[:,:-1], alpha_image
    