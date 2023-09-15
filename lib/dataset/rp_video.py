from skimage.transform import resize
from skimage.io import imread
import pickle
from tqdm import tqdm
import numpy as np
import torch
import os
from glob import glob
import cv2
from ..utils import rotation_converter

### renderpeople synthetic data
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
            imagepath_list = glob(os.path.join(self.dataset_path, 'image', f'{subject}_*.jpg'))
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
        name = self.subject
        imagepath = self.data[index]
        image = imread(imagepath) / 255.
        imagename = imagepath.split('/')[-1].split('.')[0]
        image = image[:, :, :3]
        # frame_id = int(imagename.split('_f')[-1])
        # frame_id = f'{frame_id:06d}'
        frame_id = int(imagename.split('_frame')[-1])
        # import ipdb; ipdb.set_trace()
        # load mask
        maskpath = imagepath.replace('image', 'mask')
        # maskpath = imagepath.replace('image', 'icon').replace('.jpg', '_mask.jpg')
        alpha_image = cv2.imread(maskpath)/255.
        
        if self.white_bg:
            image = image[...,:3]*alpha_image + (1.-alpha_image)
        else:
            image = image[...,:3]*alpha_image

        # ## add alpha channel
        image = np.concatenate([image, alpha_image[:,:,:1]], axis=-1)
        image = resize(image, [self.image_size, self.image_size])
        image = torch.from_numpy(image.transpose(2,0,1)).float()
        mask = image[3:]
        image = image[:3]
        mask = (mask > 0.5).float()
        
        data = {
            'idx': index,
            'frame_id': frame_id,
            'name': self.subject,
            'imagepath': imagepath,
            'image': image,
            'mask': mask,
        }

        
        # --- load camera and pose
        # load pickle 
        pkl_file = os.path.join(self.dataset_path, 'pixie', f'{imagename}_param.pkl')
        with open(pkl_file, 'rb') as f:
            codedict = pickle.load(f)
        param_dict = {}
        for key in codedict.keys():
            if isinstance(codedict[key], str):
                param_dict[key] = codedict[key]
            else:
                param_dict[key] = torch.from_numpy(codedict[key])
        # import ipdb; ipdb.set_trace()
        beta = param_dict['betas'].squeeze()
        beta = torch.cat([beta, torch.zeros(300-beta.shape[0])], dim=0)
        full_pose = param_dict['full_pose'].squeeze()
        cam = param_dict['cam'].squeeze()
        exp = torch.zeros_like(param_dict['expression'].squeeze())
        exp = torch.cat([exp, torch.zeros(100-exp.shape[0])], dim=0)
        frame_id = f'{frame_id:06}'
        data = {
            'idx': index,
            'frame_id': frame_id,
            'name': name,
            'imagepath': imagepath,
            'image': image,
            'mask': mask,
            'full_pose': full_pose,
            # 'transl': transl,
            'cam': cam,
            'beta': beta,
            'exp': exp
        }
        # --- masks from hair matting and segmentation
        # seg_image_path = os.path.join(self.dataset_path, 'cloth_segmentation', f"{imagename}.png")
        seg_image_path = os.path.join(self.dataset_path, 'hair_segmentation', f"{imagename}.jpg")
        cloth_seg = imread(seg_image_path)/255.
        cloth_seg = resize(cloth_seg, [self.image_size, self.image_size])
        cloth_mask = torch.from_numpy(cloth_seg[:,:,:3].sum(-1))[None,...]
        cloth_mask = (cloth_mask > 0.1).float()
        cloth_mask = ((mask + cloth_mask) > 1.5).float()
        skin_mask = ((mask - cloth_mask) > 0).float()
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
            mode=mode
        )
        
