"""
original from https://github.com/vchoutas/smplx
modified by Vassilis and Yao
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import os
import yaml

from .lbs import Struct, to_tensor, to_np, lbs, vertices2landmarks, JointsFromVerticesSelector, find_dynamic_lmk_idx_and_bcoords
from ..utils import util
## SMPLX 
J14_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head',
]
SMPLX_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplx', 'right_eye_smplx', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1', 'head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
extra_names = ['head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
SMPLX_names += extra_names

part_indices = {}
part_indices['body'] = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24, 123,
                                124, 125, 126, 127, 132, 134, 135, 136, 137, 138, 143])
part_indices['torso'] = np.array([  0,   1,   2,   3,   6,   9,  12,  13,  14,  15,  16,  17,  18,
                                19,  22,  23,  24,  55,  56,  57,  58,  59,  76,  77,  78,  79,
                                80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                                93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                            119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                            132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144])
part_indices['head'] = np.array([ 12,  15,  22,  23,  24,  55,  56,  57,  58,  59,  60,  61,  62,
                                63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                                76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
                                89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
                            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                            115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 134, 136,
                            137])
part_indices['face'] = np.array([ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['upper'] = np.array([ 12, 13, 14, 55, 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['hand'] = np.array([ 20,  21,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
                        36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                        49,  50,  51,  52,  53,  54, 128, 129, 130, 131, 133, 139, 140,
                        141, 142, 144])
part_indices['left_hand'] = np.array([ 20,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                        37,  38,  39, 128, 129, 130, 131, 133])
part_indices['right_hand'] = np.array([ 21,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                        52,  53,  54, 139, 140, 141, 142, 144])
# kinematic tree 
head_kin_chain = [15,12,9,6,3,0]

#--smplx joints
# 00 - Global
# 01 - L_Thigh
# 02 - R_Thigh
# 03 - Spine
# 04 - L_Calf
# 05 - R_Calf
# 06 - Spine1
# 07 - L_Foot
# 08 - R_Foot
# 09 - Spine2
# 10 - L_Toes
# 11 - R_Toes
# 12 - Neck
# 13 - L_Shoulder
# 14 - R_Shoulder
# 15 - Head
# 16 - L_UpperArm
# 17 - R_UpperArm
# 18 - L_ForeArm
# 19 - R_ForeArm
# 20 - L_Hand
# 21 - R_Hand
# 22 - Jaw
# 23 - L_Eye
# 24 - R_Eye

class SMPLX(nn.Module):
    """
    Given smplx parameters, this class generates a differentiable SMPLX function
    which outputs a mesh and 3D joints
    """
    def __init__(self, config):
        super(SMPLX, self).__init__()
        print("creating the SMPLX Decoder")
        self.cfg = config
        ss = np.load(config.smplx_model_path, allow_pickle=True)
        smplx_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(smplx_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(smplx_model.v_template), dtype=self.dtype))
        # The shape components and expression
        # expression space is the same as FLAME
        shapedirs = to_tensor(to_np(smplx_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:config.n_shape], shapedirs[:,:,300:300+config.n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = smplx_model.posedirs.shape[-1]
        posedirs = np.reshape(smplx_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype)) 
        self.register_buffer('J_regressor', to_tensor(to_np(smplx_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(smplx_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(smplx_model.weights), dtype=self.dtype))
        # for face keypoints
        self.register_buffer('lmk_faces_idx', torch.tensor(smplx_model.lmk_faces_idx, dtype=torch.long))
        self.register_buffer('lmk_bary_coords', torch.tensor(smplx_model.lmk_bary_coords, dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', torch.tensor(smplx_model.dynamic_lmk_faces_idx, dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', torch.tensor(smplx_model.dynamic_lmk_bary_coords, dtype=self.dtype))
        # pelvis to head, to calculate head yaw angle, then find the dynamic landmarks
        self.register_buffer('head_kin_chain', torch.tensor(head_kin_chain, dtype=torch.long))

        self.n_shape = config.n_shape
        self.n_pose = num_pose_basis

        #-- initialize parameters 
        # shape and expression
        self.register_buffer('shape_params', nn.Parameter(torch.zeros([1, config.n_shape], dtype=self.dtype), requires_grad=False))
        self.register_buffer('expression_params', nn.Parameter(torch.zeros([1, config.n_exp], dtype=self.dtype), requires_grad=False))
        # pose: represented as rotation matrx [number of joints, 3, 3]
        self.register_buffer('global_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('head_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('neck_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('jaw_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('eye_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(2,1,1), requires_grad=False))
        self.register_buffer('body_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(21,1,1), requires_grad=False))
        self.register_buffer('left_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))
        self.register_buffer('right_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))

        if config.extra_joint_path:
            self.extra_joint_selector = JointsFromVerticesSelector(
                fname=config.extra_joint_path)
        self.use_joint_regressor = True
        self.keypoint_names = SMPLX_names
        if self.use_joint_regressor:
            with open(config.j14_regressor_path, 'rb') as f:
                j14_regressor = pickle.load(f, encoding='latin1')
            source = []
            target = []
            for idx, name in enumerate(self.keypoint_names):
                if name in J14_NAMES:
                    source.append(idx)
                    target.append(J14_NAMES.index(name))
            source = np.asarray(source)
            target = np.asarray(target)
            self.register_buffer('source_idxs', torch.from_numpy(source))
            self.register_buffer('target_idxs', torch.from_numpy(target))
            joint_regressor = torch.from_numpy(
                j14_regressor).to(dtype=torch.float32)
            self.register_buffer('extra_joint_regressor', joint_regressor)
            self.part_indices = part_indices
        
        ## use high resolution version
        
            
            
    def forward(self, shape_params=None, expression_params=None,
                    global_pose=None, body_pose=None,
                    jaw_pose=None, eye_pose=None,
                    left_hand_pose=None, right_hand_pose=None, full_pose=None,
                    offset=None, transl=None, return_T = False):
        """
            Args:
                shape_params: [N, number of shape parameters]
                expression_params: [N, number of expression parameters]
                global_pose: pelvis pose, [N, 1, 3, 3]
                body_pose: [N, 21, 3, 3]
                jaw_pose: [N, 1, 3, 3]
                eye_pose: [N, 2, 3, 3]
                left_hand_pose: [N, 15, 3, 3]
                right_hand_pose: [N, 15, 3, 3]
            Returns:
                vertices: [N, number of vertices, 3]
                landmarks: [N, number of landmarks (68 face keypoints), 3]
                joints: [N, number of smplx joints (145), 3]
        """
        if shape_params is None:
            batch_size = full_pose.shape[0]
            shape_params = self.shape_params.expand(batch_size, -1)
        else:
            batch_size = shape_params.shape[0]
        if expression_params is None:
            expression_params = self.expression_params.expand(batch_size, -1)

        if full_pose is None:
            if global_pose is None:
                global_pose = self.global_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if body_pose is None:
                body_pose = self.body_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if jaw_pose is None:
                jaw_pose = self.jaw_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if eye_pose is None:
                eye_pose = self.eye_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if left_hand_pose is None:
                left_hand_pose = self.left_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if right_hand_pose is None:
                right_hand_pose = self.right_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            full_pose = torch.cat([global_pose, body_pose,
                                jaw_pose, eye_pose,
                                left_hand_pose, right_hand_pose], dim=1)
                                
        shape_components = torch.cat([shape_params, expression_params], dim=1)
        if offset is not None:
            if len(offset.shape) == 2:
                template_vertices = (self.v_template+offset).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1) + offset
        else:
            template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        # smplx
        vertices, joints, A, T, shape_offsets, pose_offsets = lbs(shape_components, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          pose2rot = False)
        # face dynamic landmarks
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
                find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.head_kin_chain)
            )
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)
        
        final_joint_set = [joints, landmarks]
        if hasattr(self, 'extra_joint_selector'):
            # Add any extra joints that might be needed
            extra_joints = self.extra_joint_selector(vertices, self.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)
        if self.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.extra_joint_regressor, vertices)
            joints[:, self.source_idxs] = (
                joints[:, self.source_idxs].detach() * 0.0 +
                reg_joints[:, self.target_idxs] * 1.0
            )
        ### translate z. 
        # original: -0.3 ~ 0.5
        # now: + 0.5
        # vertices[:,:,-1] = vertices[:,:,-1] + 1
        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)
            if return_T:
                A[..., :3, 3] += transl.unsqueeze(dim=1)
                T[..., :3, 3] += transl.unsqueeze(dim=1)

        if return_T:
            return vertices, landmarks, joints, A, T, shape_offsets, pose_offsets
        else:
            return vertices, landmarks, joints
        
    
    def set_canonical_space(self, pose):
        """ setup mesh in canonical space, given defined canonical pose
        Args:
            pose (torch.Tensor): [N, 55, 3, 3] or [55, 3, 3]
        """
        pose = pose.view(-1, 55, 3, 3)
        # get smplx mesh in given canonical pose
        canonical_verts, _, _, A, T, shape_offsets, pose_offsets = self.forward(full_pose=pose, return_T=True)
        verts = canonical_verts.squeeze()
        faces = self.faces_tensor.squeeze()
        # record transform for later use
        canonical_transform = T
        canonical_offsets = shape_offsets + pose_offsets
        self.register_buffer('canonical_transform', canonical_transform)
        self.register_buffer('canonical_offsets', canonical_offsets)
        # convert to high resolution mesh
        if self.cfg.use_highres:
            self.register_buffer('smplx_verts', verts)
            self.register_buffer('smplx_faces', faces)
            highres_path = self.cfg.highres_path
            embedding_path = os.path.join(highres_path, 'embedding.npz')
            mesh_path = os.path.join(highres_path, 'quad_mesh.obj')
            if not os.path.exists(embedding_path) or not os.path.exists(mesh_path):
                raise ValueError('High resolution mesh not found!')
            # load high resolution mesh embedding
            subdiv_embedding = np.load(embedding_path)
            # Faces of the low_resolution mesh that each upsampled vertex is embedded in
            nearest_faces = subdiv_embedding['nearest_faces']
            # Barycentric coordinates of each upsampled vertices
            b_coords = subdiv_embedding['b_coords']
            # Get subdivided vertices using the Barycentric coordinates
            b_coords = torch.from_numpy(b_coords)
            subdiv_verts = verts[faces[nearest_faces, 0]] * b_coords[:, 0:1] + \
                            verts[faces[nearest_faces, 1]] * b_coords[:, 1:2] + \
                            verts[faces[nearest_faces, 2]] * b_coords[:, 2:]
            verts = subdiv_verts.squeeze().float()
            # Get subdivided faces
            from pytorch3d.io import load_obj
            _, faces_idx, _ = load_obj(mesh_path)
            faces = faces_idx.verts_idx.squeeze()
            b_coords = b_coords.float()
            nearest_faces = torch.from_numpy(nearest_faces).long()
        if self.cfg.add_inner_mouth:
            if not os.path.exists(self.cfg.inner_mouth_path):
                raise ValueError('inner mouth path not exists')
            from pytorch3d.io import load_obj
            _, faces_idx, _ = load_obj(self.cfg.inner_mouth_path)
            new_faces = faces_idx.verts_idx.squeeze()
            mouth_faces = new_faces[9976:]
            flame_idx = np.load(self.cfg.flame_ids_path)
            mouth_faces = torch.from_numpy(flame_idx)[mouth_faces]
            faces = torch.cat([faces, mouth_faces], dim=0)
        self.register_buffer('verts', verts)
        self.register_buffer('faces', faces)
        if self.cfg.use_highres:
            self.register_buffer('b_coords', b_coords)
            self.register_buffer('nearest_faces', nearest_faces)
        
    def get_part_index(self):
        """get part index for each part, e.g. face, hand. For different regualrization
        """
        with open(self.cfg.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.cfg.flame_ids_path)
        with open(self.cfg.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        face_idx = list(flame_vertex_mask['face']) \
                        + list(flame_vertex_mask['left_eyeball']) \
                        + list(flame_vertex_mask['right_eyeball'])
        flame_valid = []
        for key in flame_vertex_mask:
            if key != 'neck':
                flame_valid += list(flame_vertex_mask[key])
        part_idx_dict = {
            'face': flame_idx[face_idx], 
            'hand': list(hand_idx['left_hand']) + list(hand_idx['right_hand']),
            'flame': flame_idx,
            'flame_valid': flame_idx[flame_valid]
        }
        if self.cfg.use_highres:
            for key in part_idx_dict.keys():
                part_idx = part_idx_dict[key]
                smplx_color = torch.zeros([self.verts.shape[0]])
                smplx_color[part_idx] = 1.
                highres_color = smplx_color[self.smplx_faces[self.nearest_faces, 0]] * self.b_coords[:, 0] + \
                                smplx_color[self.smplx_faces[self.nearest_faces, 1]] * self.b_coords[:, 1] + \
                                smplx_color[self.smplx_faces[self.nearest_faces, 2]] * self.b_coords[:, 2]
                highres_idx = torch.nonzero(highres_color).squeeze()
                part_idx_dict[key] = highres_idx.cpu().numpy().astype(np.int32)
            
        return part_idx_dict
    
    @torch.cuda.amp.autocast(enabled=False)
    def backward_skinning(self, full_pose, shape_params, exp=None, offset=None, transl=None):
        """  backward skinning transformation for surface points in the smplx model
        Args:
            full_pose: [N, 55, 3, 3], full pose matrix
            shape_params: [N, n_shape], shape parameters
            offset: [N, nv, 3], geometry offset in canonical space, if learn it from mesh avatar
            transl: [N, 3], translation in canonical space
        """
        batch_size = full_pose.shape[0]
        # a. forward skinning to get forward transformation
        posed_verts, _, _, joints_transform, curr_vertices_transform, shape_offsets, pose_offsets = \
                    self.forward(full_pose=full_pose, 
                                shape_params=shape_params, 
                                expression_params=exp,
                                transl=transl, 
                                return_T=True)
        curr_offsets = shape_offsets + pose_offsets
        
        # b. calculate full transformation from observation space to canonical space
        # path: pose -> smplx template -> canonical template
        vertices_transform = torch.inverse(curr_vertices_transform)
        vertices_transform_3x1 = vertices_transform[..., :3,3] - curr_offsets + self.canonical_offsets
        vertices_transform_3x3 = torch.cat(
                            [vertices_transform[..., :3, :3], 
                            vertices_transform_3x1[..., None]], dim=-1)  #3x4
        vertices_transform = torch.cat([vertices_transform_3x3, 
                                        vertices_transform[..., 3:, :]],
                                        dim=-2)
        vertices_transform = torch.matmul(self.canonical_transform.clone().repeat(batch_size, 1, 1, 1), 
                                          vertices_transform)
        if self.cfg.use_highres:
            verts_index = self.faces_tensor[self.nearest_faces]
            nearest_verts_transform = util.batch_index_select(vertices_transform,
                                                                verts_index)  #[bz, nv, 3, 4, 4]
            vertices_transform = (nearest_verts_transform *
                                    self.b_coords[None, :, :, None, None]).sum(2)
            if offset is not None:
                vertices_transform[:, :, :3, 3] = vertices_transform[:, :, :3, 3] - offset
            lbs_weights = (util.batch_index_select(self.lbs_weights[None, ...], verts_index) *
                            self.b_coords[None, :, :, None]).sum(2).squeeze()
        else:
            if offset is not None:
                vertices_transform[:, :, :3, 3] = vertices_transform[:, :, :3, 3] - offset
            lbs_weights = self.lbs_weights
        return vertices_transform, lbs_weights
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_skinning(self, full_pose, shape_params, exp=None, offset=None, transl=None):
        """  forward skinning transformation for surface points in the smplx model
        Args:
            full_pose: [N, 55, 3, 3], full pose matrix
            shape_params: [N, n_shape], shape parameters
            offset: [N, nv, 3], geometry offset in canonical space, if learn it from mesh avatar
            transl: [N, 3], translation in canonical space
        """
        batch_size = full_pose.shape[0]
        # a. forward skinning to get forward transformation
        posed_verts, _, _, joints_transform, curr_vertices_transform, shape_offsets, pose_offsets = \
                    self.forward(full_pose=full_pose, 
                                shape_params=shape_params, 
                                expression_params=exp,
                                transl=transl, 
                                return_T=True)
        curr_offsets = shape_offsets + pose_offsets
        
        # b. calculate full transformation from canonical space to observation space
        # path: canonical -> smplx template -> observation space
        vertices_transform = torch.inverse(self.canonical_transform.clone().repeat(
                batch_size, 1, 1, 1))
        vertices_transform_3x1 = vertices_transform[..., :3, 3] + curr_offsets - self.canonical_offsets 
        vertices_transform_3x3 = torch.cat(
            [vertices_transform[..., :3, :3], vertices_transform_3x1[..., None]], dim=-1)  #3x4
        vertices_transform = torch.cat([vertices_transform_3x3, vertices_transform[..., 3:, :]],
                                        dim=-2)
        vertices_transform = torch.matmul(curr_vertices_transform,
                                            vertices_transform)  #[bz, n_smplx_v, 4, 4]
        if self.cfg.use_highres:
            verts_index = self.faces_tensor[self.nearest_faces]
            nearest_verts_transform = util.batch_index_select(vertices_transform,
                                                                verts_index)  #[bz, nv, 3, 4, 4]
            vertices_transform = (nearest_verts_transform *
                                self.b_coords[None, :, :, None, None]).sum(2)
        # canonical_subject_verts = self.verts[None, ...] + offset
        # posed_verts = util.batch_transform(verts_transform, canonical_subject_verts)
        
        return vertices_transform


#---
# notes: tranform matrix for backward skinning, can also be calculated by the inverse of the forward skinning matrix