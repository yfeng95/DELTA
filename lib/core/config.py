'''
Default configs
'''
from yacs.config import CfgNode as CN
import os

cfg = CN()

workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 

# project settings
cfg.project = 'DELTA'
cfg.group = None
cfg.exp_name = None
cfg.expdir = os.path.join(workdir, 'exp')
cfg.num_gpus = 1
cfg.mixed_precision = True #False

# load models
cfg.ckpt_path = ''
cfg.mesh_ckpt_path = ''
cfg.nerf_ckpt_path = ''
cfg.pose_ckpt_path = ''

# ---------------------------------------------------------------------------- #
# Options for DELTA model
# ---------------------------------------------------------------------------- #
cfg.use_mesh = True
cfg.opt_mesh = True
cfg.mesh = CN()
cfg.mesh.use_geo = True
cfg.mesh.geo_net = 'siren'
cfg.mesh.geo_cond_type = None
cfg.mesh.geo_scale = 0.001
cfg.mesh.geo_in_dim = 3
cfg.mesh.geo_out_dim = 1 # 1 or 3, 1 for z, 3 for xyz

cfg.mesh.color_net = 'siren'
cfg.mesh.color_cond_type = None
cfg.mesh.color_out_dim = 1 # 1 or 3, 1 for z, 3 for xyz

cfg.mesh.use_light = False

# nerf model
cfg.use_nerf = False
cfg.nerf = CN()
cfg.nerf.near = -1.5
cfg.nerf.far = 1.5
cfg.nerf.nerf_net = 'mlp'
cfg.nerf.use_coarse_model = False # if true, train a coarse model to sample points
cfg.nerf.ngp_n_levels = 16
cfg.nerf.ngp_aabb = 1.5
cfg.nerf.nerf_cond = None
cfg.nerf.deform_net = 'knn'
cfg.nerf.deform_cond = None
cfg.nerf.render_type = 'vanilla' # 'vanilla' or 'nerfacc'
cfg.nerf.sample_type = 'coarse' # method to help sampling points, coarse or raymarching
cfg.nerf.n_rays = 1024 * 32 # sample rays, to avoid OOM
cfg.nerf.n_rays_patch_size = 0
cfg.nerf.n_points = 128
cfg.nerf.n_coarse_points = 64
cfg.nerf.test_n_rays = 1024 * 32
cfg.nerf.delta_last = 1e10
cfg.nerf.render_normal = False
cfg.nerf.depth_std = 0. 
cfg.nerf.dist_thresh = 10.
cfg.nerf.k_neigh = 6
# pose model
cfg.use_posemodel = True
cfg.posemodel = CN()
cfg.posemodel.use_perspective = False
cfg.posemodel.appearance_dim = 0
cfg.posemodel.deformation_dim = 0

# ---------------------------------------------------------------------------- #
# Options for Training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.optimizer = 'adam'
cfg.train.batch_size = 1
cfg.train.max_epochs = 1000
cfg.train.max_steps = 20000
# learning rate
cfg.train.beta_lr = 1e-4
cfg.train.mesh_geo_lr = 1e-4
cfg.train.mesh_color_lr = 1e-4
cfg.train.nerf_lr = 1e-4
# for pose model
cfg.train.cam_lr = 1e-4
cfg.train.pose_lr = 1e-4
cfg.train.exp_lr = 1e-3
cfg.train.light_lr = 1e-2
# logger
cfg.train.log_steps = 100
cfg.train.checkpoint_steps = 500
cfg.train.val_steps = 1000

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.type = None # dataset type, e.g. face_video, body_video
cfg.dataset.path = '' # dataset path
cfg.dataset.subject = None # subject name
cfg.dataset.white_bg = False
cfg.dataset.image_size = 512
cfg.dataset.n_images = 1000000
cfg.dataset.load_normal = False
cfg.dataset.load_lmk = False
cfg.dataset.load_light = True #False
cfg.dataset.load_fits = True
cfg.dataset.load_perspective = False
cfg.dataset.load_hair = False ### load hair for body video
cfg.dataset.load_mouth = False ### load mouth
# training setting
cfg.dataset.train = CN()
cfg.dataset.train.frame_start = 0
cfg.dataset.train.frame_end = 10000
cfg.dataset.train.frame_step = 4
cfg.dataset.val = CN()
cfg.dataset.val.frame_start = 400
cfg.dataset.val.frame_end = 500
cfg.dataset.val.frame_step = 4
cfg.dataset.test = CN()
cfg.dataset.test.frame_start = 400
cfg.dataset.test.frame_end = 500
cfg.dataset.test.frame_step = 4

# ---------------------------------------------------------------------------- #
# Options for losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.w_rgb = 1.
cfg.loss.w_rgb_nerf = 1.
cfg.loss.w_patch_mrf = 0.#0005
cfg.loss.w_patch_perceptual = 0.#0005
cfg.loss.w_alpha = 0.
cfg.loss.alpha_region = 'all'
cfg.loss.w_reg_normal = 0.
cfg.loss.w_reg_correction = 0.
cfg.loss.w_hard = 0.
cfg.loss.w_hard_scale = 1.
# weights for mesh losses
cfg.loss.mesh_rgb_region = 'skin'
cfg.loss.mesh_alpha_region = 'all'
cfg.loss.mesh_w_rgb = 1.
cfg.loss.mesh_w_rgb_skin = 0.
cfg.loss.mesh_w_mrf = 0.
cfg.loss.mesh_w_perceptual = 0.
cfg.loss.mesh_w_alpha = 0.
cfg.loss.mesh_w_alpha_inside = 10.
cfg.loss.mesh_w_alpha_all = 0.
cfg.loss.mesh_w_alpha_skin = 0.
cfg.loss.mesh_w_normal = 0.
cfg.loss.mesh_w_alpha_face = 0.
cfg.loss.mesh_mask_add_hair = False
cfg.loss.mesh_w_alpha_skin = 0.
cfg.loss.mesh_w_reg_offset = 100.
cfg.loss.mesh_w_reg_edge = 10.0
cfg.loss.mesh_w_reg_laplacian = 0.

cfg.loss.skin_consistency_type = 'verts_all_mean'
cfg.loss.mesh_skin_consistency = 0.001
cfg.loss.nerf_hard = 0.
cfg.loss.nerf_hard_scale = 1.
cfg.loss.mesh_reg_wdecay = 0.
# regs
cfg.loss.geo_reg = True
cfg.loss.reg_beta_l1 = 1e-4
cfg.loss.reg_cam_l1 = 1e-4
cfg.loss.reg_pose_l1 = 1e-4
cfg.loss.reg_a_norm = 1e-4

cfg.loss.reg_beta_temp = 1e-4
cfg.loss.reg_cam_temp = 1e-4
cfg.loss.reg_pose_temp = 1e-4
cfg.loss.nerf_reg_dxyz_w = 1e-4
##
cfg.loss.reg_lap_w = 1.0
cfg.loss.reg_edge_w = 10.0
cfg.loss.reg_normal_w = 0.01
cfg.loss.reg_offset_w_face = 500.
cfg.loss.reg_offset_w_body = 0.
cfg.loss.use_new_edge_loss = False
## new
cfg.loss.pose_reg = False
cfg.loss.background_reg = False
cfg.loss.nerf_reg_normal_w = 0. #0.01

# ---------------------------------------------------------------------------- #
# Options for Body model (SMPLX)
# ---------------------------------------------------------------------------- #
cfg.data_dir = os.path.join(workdir, 'data')
cfg.model = CN()
cfg.model.use_highres=True
cfg.model.add_inner_mouth=True
cfg.model.highres_path = os.path.join(cfg.data_dir, 'subdiv_level_1') 
cfg.model.inner_mouth_path = os.path.join(cfg.data_dir, 'head_template_mesh_mouth.obj') 
cfg.model.topology_path = os.path.join(cfg.data_dir, 'SMPL_X_template_FLAME_uv.obj') 
cfg.model.topology_smplxtex_path = os.path.join(cfg.data_dir, 'smplx_tex.obj')
cfg.model.topology_smplx_hand_path = os.path.join(cfg.data_dir, 'smplx_hand.obj')
cfg.model.smplx_model_path = os.path.join(cfg.data_dir, 'SMPLX_NEUTRAL_2020.npz')
# cfg.model.smplx_model_path = os.path.join(cfg.data_dir, 'SMPLX_NEUTRAL_lockedhead_exp.npz')
cfg.model.face_mask_path = os.path.join(cfg.data_dir, 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(cfg.data_dir, 'uv_face_eye_mask.png')
cfg.model.tex_path = os.path.join(cfg.data_dir, 'FLAME_albedo_from_BFM.npz')
cfg.model.extra_joint_path = os.path.join(cfg.data_dir, 'smplx_extra_joints.yaml')
cfg.model.j14_regressor_path = os.path.join(cfg.data_dir, 'SMPLX_to_J14.pkl')
cfg.model.flame2smplx_cached_path = os.path.join(cfg.data_dir, 'flame2smplx_tex_1024.npy')
cfg.model.smplx_tex_path = os.path.join(cfg.data_dir, 'smplx_tex.png')
cfg.model.mano_ids_path = os.path.join(cfg.data_dir, 'MANO_SMPLX_vertex_ids.pkl')
cfg.model.flame_ids_path = os.path.join(cfg.data_dir, 'SMPL-X__FLAME_vertex_ids.npy')
cfg.model.flame_vertex_masks_path = os.path.join(cfg.data_dir, 'FLAME_masks.pkl')
cfg.model.fr_model_path = os.path.join(cfg.data_dir, 'resnet50_ft_weight.pkl')
cfg.model.flame_tex_path = os.path.join(cfg.data_dir, 'FLAME_texture.npz')
# use higher dimension for face tracking
cfg.model.uv_size = 512 #256
cfg.model.n_shape = 300 #10
cfg.model.n_tex = 100 #50
cfg.model.n_exp = 100 #10 
cfg.model.n_body_cam = 3
cfg.model.n_head_cam = 3
cfg.model.n_hand_cam = 3
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uvtex_type = 'SMPLX' # FLAME or SMPLX
cfg.model.use_tex = True # whether to use flame texture model

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

