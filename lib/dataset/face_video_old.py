# import torchvision.transforms as transforms
from ipaddress import ip_address
from operator import irshift
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import torch
import os
from glob import glob
from ..utils import rotation_converter, util
# 'male-3-outdoor'


class NerfDataset(torch.utils.data.Dataset):
    """Synthetic_agora Dataset"""

    def __init__(self, cfg, mode='train', given_imagepath_list=None):
        super().__init__()
        subject = cfg.subject
        self.dataset_path = os.path.join(cfg.path, subject)
        self.subject_id = subject
        root_dir = os.path.join(self.dataset_path, 'cache')
        os.makedirs(root_dir, exist_ok=True)
        self.pose_cache_path = os.path.join(root_dir, 'pose.pt')
        self.cam_cache_path = os.path.join(root_dir, 'cam.pt')
        self.exp_cache_path = os.path.join(root_dir, 'exp.pt')
        self.beta_cache_path = os.path.join(root_dir, 'beta.pt')
        self.tex_cache_path = os.path.join(root_dir, 'tex.pt')
        self.light_cache_path = os.path.join(root_dir, 'light.pt')
        if given_imagepath_list is None:
            imagepath_list = []
            if not os.path.exists(self.dataset_path):
                print(f'{self.dataset_path} not exists, please check the data path')
                exit()
            imagepath_list = glob(os.path.join(self.dataset_path, 'matting', f'{subject}_*.png'))
            imagepath_list = sorted(imagepath_list)
            if mode != 'all':
                frame_start = getattr(cfg, mode).frame_start
                frame_end = getattr(cfg, mode).frame_end
                frame_step = getattr(cfg, mode).frame_step
                imagepath_list = imagepath_list[frame_start:min(len(imagepath_list), frame_end
                                                               ):frame_step]
        else:
            imagepath_list = given_imagepath_list

        self.data = imagepath_list
        if cfg.n_images < 10:
            self.data = self.data[:cfg.n_images]
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.image_size = cfg.image_size
        self.white_bg = cfg.white_bg
        self.load_lmk = cfg.load_lmk
        self.load_normal = cfg.load_normal
        self.load_fits = cfg.load_fits
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ## load smplx
        imagepath = self.data[index]
        image = imread(imagepath) / 255.
        imagename = imagepath.split('/')[-1].split('.')[0]
        alpha_image = image[:, :, -1:]
        image = image[:, :, :3]

        if self.white_bg:
            image = image[..., :3] * alpha_image + (1. - alpha_image)
        else:
            image = image[..., :3] * alpha_image

        # ## add alpha channel
        image = np.concatenate([image, alpha_image[:, :, :1]], axis=-1)
        image = resize(image, [self.image_size, self.image_size])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = image[3:]
        image = image[:3]

        # mask = (mask > 0.5).float() # for hair matting, remove this
        frame_id = int(imagename.split('_f')[-1])
        name = self.subject_id
        data = {
            'idx': index,
            'frame_id': frame_id,
            'name': name,
            'imagepath': imagepath,
            'image': image,
            'mask': mask,
        }

        # --- load keypoints
        if self.load_lmk and os.path.exists(
                os.path.join(self.dataset_path, 'landmark2d', f'{imagename}.txt')):
            lmk = np.loadtxt(os.path.join(self.dataset_path, 'landmark2d', f'{imagename}.txt'))
            # normalize lmk
            lmk = torch.from_numpy(lmk).float() / self.image_size
            lmk = lmk * 2. - 1
            lmk = np.concatenate([lmk, np.ones([lmk.shape[0], 1])], axis=-1)
            data['lmk'] = lmk
            ## iris
            iris = np.loadtxt(os.path.join(self.dataset_path, 'iris', f'{imagename}.txt'))
            # normalize lmk
            iris = torch.from_numpy(iris).float()
            iris[:,:2] = iris[:,:2] / self.image_size
            iris[:,:2] = iris[:,:2] * 2. - 1
            data['iris'] = iris

        # --- load camera and pose
        # pkl_file = os.path.join(self.dataset_path, 'smplx_all_id', f'{imagename}_param.pkl')
        # if not os.path.exists(pkl_file):
        pkl_file = os.path.join(self.dataset_path, 'smplx_all', f'{imagename}_param.pkl')
        if not os.path.exists(pkl_file):
            pkl_file = os.path.join(self.dataset_path, 'smplx_single', f'{imagename}_param.pkl')
        if self.load_fits and os.path.exists(os.path.join(pkl_file)):
            # load pickle
            with open(pkl_file, 'rb') as f:
                codedict = pickle.load(f)
            param_dict = {}
            for key in codedict.keys():
                if isinstance(codedict[key], str):
                    param_dict[key] = codedict[key]
                else:
                    param_dict[key] = torch.from_numpy(codedict[key])
            beta = param_dict['shape'].squeeze()
            # full_pose = param_dict['full_pose'].squeeze()
            if 'full_pose' in param_dict:
                cam = param_dict['cam'].squeeze()
                full_pose = param_dict['full_pose']
                light = param_dict['light']
            # else:
            #     jaw_pose = param_dict.get('jaw_pose', torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1)).squeeze()
            #     eye_pose = param_dict.get('eye_pose', torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)).squeeze()
            #     # eye_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            #     hand_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(15, 1, 1)
            #     full_pose = torch.cat([
            #         param_dict['global_pose'], param_dict['body_pose'], jaw_pose[None,...], eye_pose, hand_pose,
            #         hand_pose
            #     ], dim=0)
            #     # hand_pose, jaw_pose], dim=0)
            #     cam = param_dict['body_cam'].squeeze()
            #     light = param_dict.get('lights', torch.ones([1,4,6], dtype=torch.float32))
            exp = param_dict['exp']
            frame_id = f'{frame_id:06}'
            data['cam'] = cam
            data['full_pose'] = full_pose
            data['beta'] = beta
            data['exp'] = exp
            data['light'] = light
            if 'tex' in param_dict:
                data['tex'] = param_dict['tex'].squeeze() 

        # --- masks from hair matting and segmentation
        ''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
        [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
        # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        '''
        parsing_file = os.path.join(self.dataset_path, 'face_parsing', f'{imagename}.png')
        if os.path.exists(os.path.join(parsing_file)):
            semantic = imread(parsing_file)
            labels = np.unique(semantic)
            
            if 'b0_0' in self.subject_id:
                mask_np = (mask.squeeze().numpy()*255).astype(np.uint8)
                skin_cloth_region = np.ones_like(mask_np).astype(np.float32)
                skin_cloth_region[semantic==17] = 0
                skin_cloth_region[mask_np<100] = 0
                face_region = np.zeros_like(semantic)
                face_labels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
                for label in face_labels:
                    face_region[semantic == label] = 255
                
                skin_cloth_region = resize(skin_cloth_region, [self.image_size, self.image_size])
                face_region = resize(face_region, [self.image_size, self.image_size])
                skin_cloth_region = torch.from_numpy(skin_cloth_region).float()[None, ...]
                face_region = torch.from_numpy(face_region).float()[None, ...]
                data['hair_mask'] = mask * (1 - skin_cloth_region)
                data['skin_mask'] = skin_cloth_region
                data['face_mask'] = face_region
                # cv2.imwrite('mask.png', mask_np)
                # cv2.imwrite('mask_hair.png', (data['hair_mask'][0].numpy()*255).astype(np.uint8))
                # cv2.imwrite('mask_nonhair.png', (data['skin_mask'][0].numpy()*255).astype(np.uint8))
                # cv2.imwrite('mask_face.png', (data['face_mask'][0].numpy()*255).astype(np.uint8))
                # exit()
            else:
                skin_cloth_region = np.zeros_like(semantic)
                face_region = np.zeros_like(semantic)
                # fix semantic labels, if there's background inside the body, then make it as skin
                mask_np = mask.squeeze().numpy().astype(np.uint8)*255
                semantic[(semantic+mask_np)==255] = 1
                for label in labels[:-1]:    
                     # last label is hair/hat
                    if label == 0 or label == 17 or label == 18:
                        continue
                    skin_cloth_region[semantic == label] = 255
                    # if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]:
                    if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]:
                        face_region[semantic == label] = 255
                skin_cloth_region = resize(skin_cloth_region, [self.image_size, self.image_size])
                face_region = resize(face_region, [self.image_size, self.image_size])
                skin_cloth_region = torch.from_numpy(skin_cloth_region).float()[None, ...]
                face_region = torch.from_numpy(face_region).float()[None, ...]
                data['hair_mask'] = mask * (1 - skin_cloth_region)
                data['skin_mask'] = skin_cloth_region
                data['face_mask'] = face_region
            ### face and skin
            if self.mode == 'val':
                face_neck_region = np.ones_like(semantic)*255
                face_neck_region[semantic == 0] = 0
                face_neck_region[semantic == 15] = 0
                face_neck_region[semantic == 16] = 0
                face_neck_region[semantic == 18] = 0
                face_neck_region = resize(face_neck_region, [self.image_size, self.image_size])
                face_neck_region = torch.from_numpy(face_neck_region).float()[None, ...]
                data['face_neck_mask'] = face_neck_region
           
        # --- load normals
        normal_path = os.path.join(self.dataset_path, 'face_normals', f"{imagename}.png")
        if self.load_normal and os.path.exists(normal_path):
            normal = imread(normal_path) / 255.
            normal = resize(normal, [self.image_size, self.image_size])
            normal = torch.from_numpy(normal.transpose(2, 0, 1)).float()
            # normalize
            normal = normal * 2 - 1.
            data['normal_image'] = normal
            data['normal_mask'] = (normal[[0]]>-1.).float()
        return data