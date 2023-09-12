import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def depth2normal(depth, scale=None):
    dzdx =  depth[:,:,2:,2:] - depth[:,:,2:,:-2]
    dzdy =  depth[:,:,2:,2:] - depth[:,:,:-2,2:]
    # import ipdb; ipdb.set_trace()
    # gt_x =  gt[:,:,1:-1,1:] - gt[:,:,1:-1,:-1]
    # gt_y =  gt[:,:,1:,1:-1] - gt[:,:,:-1,1:-1]
    # diff = torch.mean((prediction_diff_x-gt_x)**2) + torch.mean((prediction_diff_y-gt_y)**2)
    # d = [-dzdx, -dzdy, 1]
    # normal = torch.cat([dzdx/2., dzdy/2., torch.ones_like(dzdy)], dim=1)
    if scale is None:
        scale = 1/dzdx.mean()
    normal = torch.cat([dzdx*scale, dzdy*scale, torch.ones_like(dzdy)], dim=1)
    normal = F.normalize(normal, p=2, dim=1)
    return normal