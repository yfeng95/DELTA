import os, sys
import argparse
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.visualizer import Visualizer
from lib.core.config import get_cfg_defaults, update_cfg

if __name__ == '__main__':
    from lib.core.config import get_cfg_defaults, update_cfg
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default = 'DELTA', help='project name for wandb')
    parser.add_argument('--expdir', type=str, default = 'exps', help='project dir')
    parser.add_argument('--group', type=str, default = 'released_version', help='experiments group')
    parser.add_argument('--exp_name', type=str, default = 'male1_hybrid_test0', help='specify experiments name, if not specified, use cfg names')
    parser.add_argument('--exp_idx', type=int, default = None, help='specify experiments name, if not specified, use cfg names')
    parser.add_argument('--visualize', type=str, default = 'capture', help='visualizaiton type')
    parser.add_argument('--body_model_path', type=str, default = '', help='if specified, then will use this model for body part')
    parser.add_argument('--clothing_model_path', type=str, default = '', help='if specified, then will use this model for clothing part')
    parser.add_argument('--image_size', type=int, default = 512, help='cfg file path')
    parser.add_argument('--video_type', type=str, default ='mp4', help='video type, gif or mp4')
    parser.add_argument('--fps', type=int, default = 10, help='fps for video, suggest 10 for novel view, and 30 for animation')
    parser.add_argument('--saveImages', action="store_true", help='save each image')    
    parser.add_argument('--frame_id', type=int, default = 0, help='frame id for novel view and mesh extraction')
    parser.add_argument('--animation_file', type=str, default = 'data/animation_actions.pkl', help='path for pose data')   
    parser.add_argument('--shape_scale', type=float, default = 0., help='change the shape of the subject')
    parser.add_argument('--max_yaw', type=int, default = 90, help='max yaw angle for novel view')
    args = parser.parse_args()
    # add exp
    if args.exp_idx is not None:
        exp_list = os.listdir(os.path.join(args.expdir, args.group))
        exp_list = sorted(exp_list)
        args.exp_name = exp_list[args.exp_idx]
        
    #-- project setting
    cfg = get_cfg_defaults()
    cfg.project = args.project 
    cfg.expdir = args.expdir
    cfg.group = args.group
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    else:
        cfg.exp_name = f'{cfg.dataset.subject}' 
    cfg.savedir = os.path.join(cfg.expdir, cfg.group, cfg.exp_name)
    
    # visualization
    cfg = update_cfg(cfg, os.path.join(cfg.savedir, 'exp_config.yaml'))
    cfg = update_cfg(cfg, os.path.join(cfg.savedir, 'data_config.yaml'))
    cfg.clean = False
    cfg.resume = True
    cfg.use_wandb = False
    cfg.dataset.white_bg = True
    
    if args.visualize == 'all':
        vis_list = ['capture', 'novel_view', 'extract_mesh', 'animate', 'change_shape']
        visualizer = Visualizer(config=cfg)
        for vis_type in vis_list:
            args.visualize = vis_type
            visualizer.run(args.visualize, args=args)
    else:
        visualizer = Visualizer(config=cfg)
        visualizer.run(args.visualize, args=args)
    
