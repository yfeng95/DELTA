''' Projections
Pinhole Camera Model

Ref: 
http://web.stanford.edu/class/cs231a/lectures/lecture2_camera_models.pdf
https://github.com/YadiraF/face3d/blob/master/face3d/mesh/transform.py
https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer
https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/implementing-virtual-pinhole-camera
https://kornia.readthedocs.io/en/v0.1.2/pinhole.html

Pinhole Camera Model:
1. world space to camera space: 
    Normally, the object is represented in the world reference system, 
    need to first map it into camera system/coordinates/space
    P_camera = [R|t]P_world
    [R|t]: camera to world transformation matrix, defines the camera position and oritation
    Given: where the camera is in the world system. (extrinsic/external)
    represented by: 
        look at camera: eye position, at, up direction
2. camera space to image space: 
    Then project the obj into image plane
    P_image = K[I|0]P_camera
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    Given: settings of te camera. (intrinsic/internal)
    represented by:
        focal lengh, image/film size 
        or fov, near, far
        Angle of View: computed from the focal length and the film size parameters.
    perspective projection:
        x' = f*x/z; y' = f*y/z

Finally:
P_image = MXP_world
M = K[R T]
homo: 4x4
'''
import torch
import numpy as np
import torch.nn.functional as F

# --------------------- 0. camera projection
def transform(points, intrinsic, extrinsic):
    '''  Perspective projection
    Args:
        points: [bz, np, 3]
        intrinsic: [bz, 3, 3]
        extrinsic: [bz, 3, 4]
    '''
    points = torch.matmul(points, extrinsic[:,:3,:3].transpose(1,2)) + extrinsic[:,:3,3][:,None,:]
    points = torch.matmul(points, intrinsic.transpose(1,2))

    ## if homo
    # vertices_homo = camera_util.homogeneous(mesh['vertices'])
    # transformed_verts = torch.matmul(vertices_homo, extrinsic.transpose(1,2))
    return points

def perspective_project(points, focal=None, image_hw=None, extrinsic=None, transl=None):
    ''' points from world space to ndc space (for pytorch3d rendering)
    TODO
    '''
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype
    # non homo
    if points.shape[-1] == 3:
        if transl is not None:
            points = points + transl[:,None,:]
        if extrinsic is not None:
            points = torch.matmul(points, extrinsic[:,:3,:3].transpose(1,2)) + extrinsic[:,:3,3][:,None,:]
        if focal is not None:
            # import ipdb; ipdb.set_trace()
            if image_hw is not None:
                H, W = image_hw
                fx = 2*focal/H
                fy = 2*focal/W
            else:
                fx = fy = focal

            # 2/H is for normalization
            # intrinsic = torch.tensor(
            #         [[fx, 0, 0],
            #          [0, fy, 0],
            #          [0,   0,  1]], device=device, dtype=dtype)[None,...].repeat(batch_size, 1, 1)
            intrinsic = torch.tensor(
                    [[1, 0, 0],
                     [0, 1, 0],
                     [0,   0,  1]], device=device, dtype=dtype)[None,...].repeat(batch_size, 1, 1)
            intrinsic_xy = intrinsic[:,:2]*fx
            intrinsic_z = intrinsic[:,2:]
            intrinsic = torch.cat([intrinsic_xy, intrinsic_z], dim = 1)
            # if points.requires_grad:
            #     import ipdb; ipdb.set_trace()
            points = torch.matmul(points, intrinsic.transpose(1,2))
            # perspective distortion
            # points[:,:,:2] = points[:,:,:2]/(points[:,:,[2]]+1e-5) # inplace, has problem for gradient
            z = points[:,:,[2]]
            xy = points[:,:,:2]/(z+1e-6)
            points = torch.cat([xy, z], dim=-1)
        return points

def perspective_project_inv(points, focal=None, image_hw=None, extrinsic=None, transl=None):
    ''' points from world space to ndc space (for pytorch3d rendering)
    TODO
    '''
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype
    # non homo
    if points.shape[-1] == 3:
        if focal is not None:
            # import ipdb; ipdb.set_trace()
            if image_hw is not None:
                H, W = image_hw
                fx = 2*focal/H
                fy = 2*focal/W
            else:
                fx = fy = focal
            # 2/H is for normalization
            intrinsic = torch.tensor(
                    [[1, 0, 0],
                     [0, 1, 0],
                     [0,   0,  1]], device=device, dtype=dtype)[None,...].repeat(batch_size, 1, 1)
            intrinsic_xy = intrinsic[:,:2]*fx
            intrinsic_z = intrinsic[:,2:]
            intrinsic = torch.cat([intrinsic_xy, intrinsic_z], dim = 1)
            
            z = points[:,:,[2]]
            xy = points[:,:,:2]*(z+1e-5)
            points = torch.cat([xy, z], dim=-1)
            
            intrinsic = torch.inverse(intrinsic)
            points = torch.matmul(points, intrinsic.transpose(1,2))
        if transl is not None:
            points = points - transl[:,None,:]
        if extrinsic is not None:
            points = torch.matmul(points, extrinsic[:,:3,:3].transpose(1,2)) + extrinsic[:,:3,3][:,None,:]
        return points


    # TODO: homo

# --------------------- 1. world space to camera space: 
def look_at(eye, at=[0, 0, 0], up=[0, 1, 0]):
    """
    "Look at" transformation of vertices.
    standard camera space: 
        camera located at the origin. 
        looking down negative z-axis. 
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], 
           [0,   1]]
    Args:
      eye: [3,] the XYZ world space position of the camera.
      at: [3,] a position along the center of the camera's gaze.
      up: [3,] up direction 
    Returns:
      extrinsic: R, t
    """
    device = eye.device
    # if list or tuple convert to numpy array
    if isinstance(at, list) or isinstance(at, tuple):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    # if numpy array convert to tensor
    elif isinstance(at, np.ndarray):
        at = torch.from_numpy(at).to(device)
    elif torch.is_tensor(at):
        at.to(device)

    if isinstance(up, list) or isinstance(up, tuple):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    elif isinstance(up, np.ndarray):
        up = torch.from_numpy(up).to(device)
    elif torch.is_tensor(up):
        up.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    elif torch.is_tensor(eye):
        eye = eye.to(device)

    batch_size = eye.shape[0]
    if eye.ndimension() == 1:
        eye = eye[None, :].repeat(batch_size, 1)
    if at.ndimension() == 1:
        at = at[None, :].repeat(batch_size, 1)
    if up.ndimension() == 1:
        up = up[None, :].repeat(batch_size, 1)

    # create new axes
    # eps is chosen as 0.5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)
    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    camera_R = r
    camera_t = eye
    # Note: x_new = R(x - t) 
    return camera_R, camera_t

def get_extrinsic(R, t, homo=False):
    batch_size = R.shape[0]
    device = R.device
    extrinsic = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    extrinsic[:, :3, :3] = R
    extrinsic[:, :3, [-1]] = -torch.matmul(R, t[:,:,None])
    if homo:
        return extrinsic
    else:
        return extrinsic[:,:3,:]

#------------------------- 2. camera space to image space (perspective projection): 
def get_intrinsic(focal, H, W, cx=0., cy=0., homo=False, batch_size=1., device='cuda:0'):
    '''
    given different control parameteres
    TODO: generate intrinsic matrix from other inputs
    P = np.array([[near/right, 0, 0, 0],
                 [0, near/top, 0, 0],
                 [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                 [0, 0, -1, 0]])
    '''
    intrinsic = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    f = focal
    K = torch.tensor(
                    [[f/(H/2), 0, cx],
                     [0, f/(W/2), cy],
                     [0,   0,  1]], device=device, dtype=torch.float32)[None,...]
    intrinsic[:, :3, :3] = K
    if homo:
        return intrinsic
    else:
        return intrinsic[:,:3,:3]
    

#------------------------- composite intrinsic and extrinsic into one matrix
def compose_matrix(K, R, t):
    '''
    Args:
        K: [N, 3, 3]
        R: [N, 3, 3]
        t: [N, 3]
    Returns:
        P: [N, 4, 4]
    ## test if homo is the same as no homo:
    batch_size, nv, _ = trans_verts.shape
    trans_verts = torch.cat([trans_verts, torch.ones([batch_size, nv, 1], device=trans_verts.device)], dim=-1)
    trans_verts = torch.matmul(trans_verts, P.transpose(1,2))[:,:,:3]
    (tested, the same)
    # t = -Rt
    '''
    batch_size = K.shape[0]
    device = K.device
    intrinsic = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    extrinsic = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    intrinsic[:, :3, :3] = K
    extrinsic[:, :3, :3] = R
    # import ipdb; ipdb.set_trace()
    extrinsic[:, :3, [-1]] = -torch.matmul(R, t[:,:,None])
    P = torch.matmul(intrinsic, extrinsic)
    return P, intrinsic, extrinsic

# def perspective_project(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
#     '''
#     Calculate projective transformation of vertices given a projection matrix
#     Input parameters:
#     K: batch_size * 3 * 3 intrinsic camera matrix
#     R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
#     dist_coeffs: vector of distortion coefficients
#     orig_size: original size of image captured by the camera
#     Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
#     pixels and z is the depth
#     '''
#     # instead of P*x we compute x'*P'
#     vertices = torch.matmul(vertices, R.transpose(2,1)) + t
#     x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
#     x_ = x / (z + eps)
#     y_ = y / (z + eps)

#     # Get distortion coefficients from vector
#     k1 = dist_coeffs[:, None, 0]
#     k2 = dist_coeffs[:, None, 1]
#     p1 = dist_coeffs[:, None, 2]
#     p2 = dist_coeffs[:, None, 3]
#     k3 = dist_coeffs[:, None, 4]

#     # we use x_ for x' and x__ for x'' etc.
#     r = torch.sqrt(x_ ** 2 + y_ ** 2)
#     x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
#     y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
#     vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
#     vertices = torch.matmul(vertices, K.transpose(1,2))
#     u, v = vertices[:, :, 0], vertices[:, :, 1]
#     v = orig_size - v
#     # map u,v from [0, img_size] to [-1, 1] to use by the renderer
#     u = 2 * (u - orig_size / 2.) / orig_size
#     v = 2 * (v - orig_size / 2.) / orig_size
#     vertices = torch.stack([u, v, z], dim=-1)
#     return vertices

def to_homo(points):
    '''
    points: [N, num of points, 2/3]
    '''
    batch_size, num, _ = points.shape
    points = torch.cat([points, torch.ones([batch_size, num, 1], device=points.device, dtype=points.dtype)], dim=-1)
    return points
    
def homogeneous(points):
    """
    Concat 1 to each point
    :param points (..., 3)
    :return (..., 4)
    """
    return F.pad(points, (0, 1), "constant", 1.0)


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def batch_orth_proj_inv(X_trans, camera):
    ''' remove orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    # camera = camera.clone().view(-1, 1, 3)
    # Xn = X_trans/camera[:, :, 0:1]

    # X_xy = Xn[:, :, :2] - camera[:, :, 1:]
    # X = torch.cat([X_xy, Xn[:,:,2:]], 2)
    # return X
    # print(X_trans.device, camera.device)
    # exit()
    # import ipdb; ipdb.set_trace()
    camera_n = camera.view(-1, 1, 3)
    Xn = X_trans/camera_n[:, :, 0:1]

    X_xy = Xn[:, :, :2] - camera_n[:, :, 1:]
    X = torch.cat([X_xy, Xn[:,:,2:]], 2)
    return X
