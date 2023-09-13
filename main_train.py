# test
# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import argparse
import shutil
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.trainer import Trainer
from lib.core.config import get_cfg_defaults, update_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="DELTA training")
    ## single 
    parser.add_argument('--project', type=str, default = 'DELTA', help='project name for wandb')
    parser.add_argument('--expdir', type=str, default = 'exps', help='project dir')
    parser.add_argument('--group', type=str, default = 'debug', help='experiments group')
    parser.add_argument('--exp_name', type=str, default = None, help='specify experiments name, if not specified, use cfg names')
    parser.add_argument('--exp_cfg', type=str, default = 'configs/exp/face/nerf.yml', help='exp cfg file path')
    parser.add_argument('--data_cfg', type=str, default = 'configs/data/face/person_2_train.yml', help='data cfg file path')
    # parser.add_argument('--wandb_mode', type=str, default = 'online', help='wandb mode, online, offline or disabled')
    parser.add_argument('--resume', action="store_true", help='if not resume, will delete folders')
    parser.add_argument('--use_wandb', action="store_true", help='if not resume, will delete folders')
    # for visualization 
    parser.add_argument('--visualize', type=str, default = None, help='data cfg file path')
    parser.add_argument('--vispath', type=str, default = None, help='data cfg file path')
    
    # for debug
    parser.add_argument('--debug', action="store_true", default = False, help='delete folders')
    parser.add_argument('--single', action="store_true", default = False, help='delete folders')
    parser.add_argument('--few', action="store_true", default = False, help='delete folders')
    parser.add_argument('--check_existing', action="store_true", default = False, help='delete folders')
    ## for cluster training
    parser.add_argument('--data_cfg_dir', type=str, default = 'configs/data/face', help='dataset name')
    parser.add_argument('--data_cfg_idx', type=int, default = None, help='cfg file path idx, for cluster training')
    ## for load data
    parser.add_argument('--ckpt', type=str, default = None, help='load pretrained model')
    parser.add_argument('--nerf_ckpt', type=str, default = None, help='load pretrained model')
    parser.add_argument('--mesh_ckpt', type=str, default = None, help='load pretrained model')
    parser.add_argument('--pose_ckpt', type=str, default = None, help='load pretrained model')
    args = parser.parse_args()    
    
    if args.data_cfg_idx is not None and os.path.exists(args.data_cfg_dir):
        data_cfg_list = glob(os.path.join(args.data_cfg_dir, '*.yml'))
        data_cfg_list = sorted(data_cfg_list)
        args.data_cfg = data_cfg_list[args.data_cfg_idx]
    return args


if __name__ == '__main__':
    args = parse_args()
    #-- load cfg
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, args.exp_cfg)
    cfg = update_cfg(cfg, args.data_cfg)

    #-- project setting
    cfg.project = args.project 
    cfg.expdir = args.expdir
    cfg.group = args.group
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    else:
        cfg.exp_name = f'{cfg.dataset.subject}' 
    cfg.savedir = os.path.join(cfg.expdir, cfg.group, cfg.exp_name)
    
    if args.check_existing:
        imagepath = os.path.join(cfg.savedir, 'val_images/000000.jpg')
        if os.path.exists(imagepath):
            exit()
    # load pretrained model
    if args.ckpt is not None:
        if os.path.isfile(args.ckpt):
            cfg.ckpt_path = args.ckpt   
        elif os.path.isdir(args.ckpt):
            cfg.ckpt_path = os.path.join(args.ckpt, cfg.dataset.subject, 'model.tar')

    if args.pose_ckpt is not None:
        if os.path.isfile(args.pose_ckpt):
            cfg.pose_ckpt_path = args.pose_ckpt   
        elif os.path.isdir(args.pose_ckpt):
            cfg.pose_ckpt_path = os.path.join(args.pose_ckpt, cfg.dataset.subject, 'model.tar')
    
    cfg.resume = args.resume
    cfg.use_wandb = args.use_wandb
    if not cfg.resume:
        import shutil
        if os.path.isdir(cfg.savedir):
            shutil.rmtree(cfg.savedir)    
        
    os.makedirs(cfg.savedir, exist_ok=True)
    shutil.copy(args.data_cfg, os.path.join(cfg.savedir, 'data_config.yaml'))
    shutil.copy(args.exp_cfg, os.path.join(cfg.savedir, 'exp_config.yaml'))
    trainer = Trainer(config=cfg)
    trainer.fit()
