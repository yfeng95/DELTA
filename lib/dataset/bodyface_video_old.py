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
                 load_normal=False, load_lmk=False, load_light=False, load_fits=True, 
                 mode='train'):
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
        assert len(self.data) > 0, f"Can't find data; make sure datapath {self.dataset_path} is correct"

        self.image_size = image_size
        self.white_bg = white_bg
        self.load_normal = load_normal
        self.load_lmk = load_lmk
        self.load_light = load_light
        self.load_fits = load_fits
        self.mode = mode
        
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
            # iris = np.loadtxt(os.path.join(self.dataset_path, 'iris', f'{imagename}.txt'))
            # # normalize lmk
            # iris = torch.from_numpy(iris).float()
            # iris[:,:2] = iris[:,:2] / self.image_size
            # iris[:,:2] = iris[:,:2] * 2. - 1
            # data['iris'] = iris

        # --- load camera and pose
        # pkl_file = os.path.join(self.dataset_path, 'smplx_all', f'{imagename}_param.pkl')
        # if not os.path.exists(pkl_file):
        #     pkl_file = os.path.join(self.dataset_path, 'smplx_single', f'{imagename}_param.pkl')
        # if not os.path.exists(pkl_file):
        #     print(pkl_file)
        #     exit()
        pkl_file = os.path.join(self.dataset_path, 'face_smplx', f'{imagename}_param.pkl')
        if self.load_fits and os.path.exists(os.path.join(pkl_file)):
            with open(pkl_file, 'rb') as f:
                codedict = pickle.load(f)
            param_dict = {}
            for key in codedict.keys():
                if isinstance(codedict[key], str):
                    param_dict[key] = codedict[key]
                else:
                    param_dict[key] = torch.from_numpy(codedict[key])
            data['cam'] = param_dict['cam'].squeeze()
            data['full_pose'] = param_dict['full_pose']
            data['beta'] = param_dict['shape'].squeeze() if 'shape' in param_dict else torch.zeros([300], dtype=torch.float)
            data['exp'] = param_dict['exp']
            if self.load_light and 'light' in param_dict:
                data['light'] = param_dict['light']
            if 'tex' in param_dict:
                data['tex'] = param_dict['tex'].squeeze() 

        # --- masks from hair matting and segmentation
        ''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
        [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
        # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        '''
        # parsing_file = os.path.join(self.dataset_path, 'face_parsing', f'{imagename}.png')
        # if os.path.exists(os.path.join(parsing_file)):
        #     semantic = imread(parsing_file)
        #     labels = np.unique(semantic)
            
        #     skin_cloth_region = np.zeros_like(semantic)
        #     face_region = np.zeros_like(semantic)
        #     # fix semantic labels, if there's background inside the body, then make it as skin
        #     mask_np = mask.squeeze().numpy().astype(np.uint8)*255
        #     semantic[(semantic+mask_np)==255] = 1
        #     for label in labels[:-1]:    
        #             # last label is hair/hat
        #         if label == 0 or label == 17 or label == 18:
        #             continue
        #         skin_cloth_region[semantic == label] = 255
        #         # if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]:
        #         if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]:
        #             face_region[semantic == label] = 255
        #     skin_cloth_region = resize(skin_cloth_region, [self.image_size, self.image_size])
        #     face_region = resize(face_region, [self.image_size, self.image_size])
        #     skin_cloth_region = torch.from_numpy(skin_cloth_region).float()[None, ...]
        #     face_region = torch.from_numpy(face_region).float()[None, ...]
        #     data['nonskin_mask'] = mask * (1 - skin_cloth_region)
        #     data['hair_mask'] = mask * (1 - skin_cloth_region)
        #     data['skin_mask'] = skin_cloth_region
        #     data['face_mask'] = face_region
        #     ### face and skin
        #     if self.mode == 'test':
        #         face_neck_region = np.ones_like(semantic)*255
        #         face_neck_region[semantic == 0] = 0
        #         face_neck_region[semantic == 15] = 0
        #         face_neck_region[semantic == 16] = 0
        #         face_neck_region[semantic == 18] = 0
        #         face_neck_region = resize(face_neck_region, [self.image_size, self.image_size])
        #         face_neck_region = torch.from_numpy(face_neck_region).float()[None, ...]
        #         data['face_neck_mask'] = face_neck_region
                
        # --- load normals
        ## load data for backview
        global_pose = data['full_pose'][0:1]
        # convert to yaw angle
        global_yaw = rotation_converter._compute_euler_from_matrix(global_pose)[0]
        global_yaw = global_yaw*180/np.pi
        is_back=False
        ## facing away the camera
        data['global_yaw'] = global_yaw
        if global_yaw[0] > 0.:
            ## load lmk 
            if os.path.exists(os.path.join(self.dataset_path, 'body_landmark2d', f'{imagename}.txt')):
                lmk = np.loadtxt(os.path.join(self.dataset_path, 'body_landmark2d', f'{imagename}.txt'))
                # normalize lmk
                lmk = torch.from_numpy(lmk).float() / self.image_size
                lmk = lmk * 2. - 1
                lmk = np.concatenate([lmk, np.ones([lmk.shape[0], 1])], axis=-1)
                data['lmk'] = lmk
            ## load segmentation
            mask_path = os.path.join(self.dataset_path, 'body_matting', f'{imagename}.png')
            alpha_image = imread(maskpath) / 255.
            alpha_image = alpha_image[:, :, -1:]
            alpha_image = torch.from_numpy(alpha_image.transpose(2, 0, 1)).float()
            alpha_image = (alpha_image>0.1).float()
            data['mask'] = alpha_image
            ## load hair mask
            hairseg_path = os.path.join(self.dataset_path, 'clipseg', f'{imagename}.png')
            hairseg = imread(hairseg_path) / 255.
            hairseg = hairseg[:, :, None]
            hairseg = torch.from_numpy(hairseg.transpose(2, 0, 1)).float()
            data['nonskin_mask'] = mask * hairseg
            data['hair_mask'] = mask * hairseg
            data['skin_mask'] = mask * (1-hairseg)
            data['face_mask'] = mask * (1-hairseg)
            is_back = True
        data['is_back'] = is_back
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
            mode=mode
        )
        
