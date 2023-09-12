import os
from loguru import logger
import wandb
from .util import visualize_grid

class WandbLogger(object):
    def __init__(self, dir, use_wandb=True, image_size=512, 
                 project=None, group=None, name=None, config=None, resume=False):
        """logger for training process

        Args:
            dir (str): directory to save results
            mode: 'train' or 'test'
            use_wandb (bool, optional): _description_. Defaults to False.
            
        """
        self.image_size = image_size
        self.name = name
        self.group = group
        logdir = os.path.join(dir, 'logs')
        os.makedirs(logdir, exist_ok=True)
        logger.add(os.path.join(logdir, f'{name}.log'))
        logger.info(f'start project[{project}] group[{group}] name[{name}],  results will be saved in {dir}')
        
        # path for saving images
        self.train_dir = os.path.join(dir, f'train_images')
        self.val_dir = os.path.join(dir, f'val_images')
        self.val_novel_view_dir = os.path.join(dir, f'val_novel_view_images')
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.val_novel_view_dir, exist_ok=True)
        
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandbdir = os.path.join(logdir, 'wandb')
            os.makedirs(wandbdir, exist_ok=True)
            wandb.init(
                id = f'{group}_{name}',
                # mode=wandb_mode,
                project=project,
                group=group,
                name=name,
                dir=wandbdir,
                resume=resume,
                config=config
            )
    
    def log_metrics(self, metrics, step):
        val_info = f"ExpName: {self.name} Step: {step}, validation scores: \n"
        for k, v in metrics.items():
            val_info = val_info + f'{k}: {v:.6f}, '
        logger.info(val_info)
        if self.use_wandb:
            wandb_info = {}
            for k, v in metrics.items():
                k = f'val/{k}'
                wandb_info[k] = v
            wandb.log(wandb_info, step=step)                 
    
    def log_loss(self, losses, step):
        loss_info = f"Group: {self.group}, ExpName: {self.name} Step: {step}\n"
        for k, v in losses.items():
            loss_info = loss_info + f'{k}: {v:.6f}, '
        logger.info(loss_info)
        if self.use_wandb:
            wandb_info = {}
            for k, v in losses.items():
                k = f'train/{k}'
                wandb_info[k] = v
            wandb.log(wandb_info, step=step)                 
    
    def log_image(self, visdict, step, mode='train'):
        if mode == 'train':
            visdir = self.train_dir
        elif mode == 'val':
            visdir = self.val_dir
        elif mode == 'val_novel_view':
            visdir = self.val_novel_view_dir
        savepath = os.path.join(visdir, f'{step:06}.jpg')
        grid_image = visualize_grid(visdict, savepath, return_gird=True, size=self.image_size)
        if self.use_wandb:
            images = wandb.Image(grid_image[:,:,[2,1,0]], caption="visualize")
            wandb.log({mode: images}, step=step)
    
    def state_dict():
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step
        }
        

