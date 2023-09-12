'''ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html'''

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def dist_setup(rank, world_size, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def dist_cleanup():
    dist.destroy_process_group()

