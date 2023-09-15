from skimage.transform import resize
from skimage.io import imread
import pickle
from tqdm import tqdm
import numpy as np
import torch
import os
from glob import glob
from ..utils import rotation_converter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, subject, image_size=512, white_bg=False, 
                 frame_start=0, frame_end=10000, frame_step=1,
                 given_imagepath_list=None, cache_data=False,
                 load_normal=False, load_lmk=False, load_light=False, load_fits=True, load_hair=False):
        """ dataset
        Args:
            path (str): path to dataset
            subject (str): subject name
            image_size (int, optional): image size. Defaults to 512.
            white_bg (bool, optional): whether to use white background. Defaults to False.
            frame_start (int, optional): start frame. Defaults to 0.
            frame_end (int, optional): end frame. Defaults to 10000.
            frame_step (int, optional): frame step. Defaults to 1.
            given_imagepath_list (list, optional): specify image path list. Defaults to None.
            cache_data (bool, optional): whether to cache data. Defaults to False.
        """
        super().__init__()
        self.dataset_path = os.path.join(path, subject)
        self.subject = subject
    
        if given_imagepath_list:
            imagepath_list = given_imagepath_list        
        else:
            imagepath_list = []
            assert os.path.exists(self.dataset_path), f'path {self.dataset_path} does not exist'
            imagepath_list = glob(os.path.join(self.dataset_path, 'image', f'{subject}_*.png'))
            imagepath_list = sorted(imagepath_list)
            imagepath_list = imagepath_list[frame_start:min(len(imagepath_list), frame_end):frame_step]

        self.data = imagepath_list
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.image_size = image_size
        self.white_bg = white_bg
        self.load_normal = load_normal
        self.load_lmk = load_lmk
        self.load_light = load_light
        self.load_fits = load_fits
        self.load_hair = load_hair
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image
        imagepath = self.data[index]
        image = imread(imagepath) / 255.
        imagename = imagepath.split('/')[-1].split('.')[0]
        image = image[:, :, :3]
        frame_id = int(imagename.split('_f')[-1])
        frame_id = f'{frame_id:06d}'

        # load mask
        maskpath = os.path.join(self.dataset_path, 'matting', f'{imagename}.png')
        alpha_image = imread(maskpath) / 255.
        alpha_image = (alpha_image > 0.5).astype(np.float32)
        alpha_image = alpha_image[:, :, -1:]
        if self.white_bg:
            image = image[..., :3] * alpha_image + (1. - alpha_image)
        else:
            image = image[..., :3] * alpha_image
        # add alpha channel
        image = np.concatenate([image, alpha_image[:, :, :1]], axis=-1)
        image = resize(image, [self.image_size, self.image_size])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = image[3:]
        image = image[:3]

        data = {
            'idx': index,
            'frame_id': frame_id,
            'name': self.subject,
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
        pkl_file = os.path.join(self.dataset_path, 'smplx_all', f'{imagename}_param.pkl')
        if not os.path.exists(pkl_file):
            pkl_file = os.path.join(self.dataset_path, 'pixie', f'{imagename}_param.pkl')
        # if not os.path.exists(pkl_file):
        #     pkl_file = os.path.join(self.dataset_path, 'smplx_single', f'{imagename}_param.pkl')
        # if self.load_fits and os.path.exists(os.path.join(pkl_file)):
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
            full_pose = param_dict['full_pose'].squeeze()
            cam = param_dict['cam'].squeeze()
        else:
            jaw_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0) #param_dict['jaw_pose']
            eye_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2,1,1)
            # hand_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(15,1,1)
            full_pose = torch.cat([param_dict['global_pose'], param_dict['body_pose'],
                                jaw_pose, eye_pose, 
                                # hand_pose, hand_pose], dim=0)        
                                param_dict['left_hand_pose'], param_dict['right_hand_pose']], dim=0)                
            cam = param_dict['body_cam'].squeeze()
            beta = torch.cat([beta, torch.zeros(300-beta.shape[0])], dim=0)
            # beta[:] = 0.
        
        ## for snapshot dataset, load given pose
        if 'casual' in self.subject:
            name = self.subject
            posepath = f'/ps/scratch/yfeng/Data/Projects-data/MerfHuman/video_data/snapshot/{name}/pose/{name}_frame{int(frame_id):04}.npy'
            if os.path.exists(posepath):
                axis_pose = np.load(posepath).reshape(24, 3) #72
                axis_pose = torch.from_numpy(axis_pose)
                #
                pose = rotation_converter.batch_axis2matrix(axis_pose) # + 1e-6
                # full_pose[:24] = pose
                ind = list(range(21))
                # ind = [0,1,2,3,4,5,6,9,12,13,14,15] #,16,17,18,19]
                # ind = [0,1,2,3,4,5,6,9,12,16,17,18,19]
                # ind = [0]
                full_pose[ind] = pose[ind]
            
        ## set hand pose to zero
        # full_pose[-30:] = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(30,1,1)
        exp = torch.zeros_like(param_dict['exp'].squeeze()[:10])
        data['full_pose'] = full_pose
        data['cam'] = cam
        data['exp'] = torch.cat([exp, torch.zeros(100-exp.shape[0])], dim=0)
        data['exp'][:] = 0.
        data['beta'] = beta
        
        # --- masks from hair matting and segmentation
        ''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
        [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
        # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        '''
        ## 
        if self.load_hair:
            ## load target mask
            target_path = os.path.join(self.dataset_path, 'hair_segmentation', f"_target.png")
            target_mask = imread(target_path)/255.
            target_mask = resize(target_mask, [self.image_size, self.image_size])
            target_mask = torch.from_numpy(target_mask)[None,...]
            data['target_mask'] = target_mask
            ## 
            data['mask'] = data['target_mask']*data['mask']
            # load hair segmentation
            seg_image_path = os.path.join(self.dataset_path, 'hair_segmentation', f"{imagename}.png")
            cloth_seg = imread(seg_image_path)/255.
            cloth_seg = resize(cloth_seg, [self.image_size, self.image_size])
            cloth_mask = torch.from_numpy(cloth_seg)[None,...]
            cloth_mask = (cloth_mask > 0.5).float()
            # cloth_mask = ((mask + cloth_mask) > 1.5).float()
            skin_mask = ((mask - cloth_mask) > 0.5).float()
            data['nonskin_mask'] = cloth_mask*target_mask
            data['skin_mask'] = skin_mask*target_mask
        else:
            seg_image_path = os.path.join(self.dataset_path, 'cloth_segmentation', f"{imagename}.png")
            cloth_seg = imread(seg_image_path)/255.
            cloth_seg = resize(cloth_seg, [self.image_size, self.image_size])
            cloth_mask = torch.from_numpy(cloth_seg[:,:,:3].sum(-1))[None,...]
            cloth_mask = (cloth_mask > 0.).float()
            cloth_mask = ((mask + cloth_mask) > 1.5).float()
            skin_mask = ((mask - cloth_mask) > 0.5).float()
            data['nonskin_mask'] = cloth_mask
            data['skin_mask'] = skin_mask
        return data
    
    @classmethod
    def from_config(cls, cfg, mode='train'):
        return cls(
            path=cfg.path, 
            subject=cfg.subject, 
            image_size=cfg.image_size,
            white_bg=cfg.white_bg,
            frame_start=getattr(cfg, mode).frame_start, 
            frame_end=getattr(cfg, mode).frame_end, 
            frame_step=getattr(cfg, mode).frame_step,
            load_normal=cfg.load_normal,
            load_lmk=cfg.load_lmk,
            load_light=cfg.load_light,
            load_fits=cfg.load_fits,
            load_hair=cfg.load_hair
        )
        
