import os
import sys
import time
import copy
import shutil
import random
import torch
import cv2
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def init_distributed_mode(port='29500'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        import subprocess
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        master_port = os.environ.get('MASTER_PORT', port)
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        dist_url = 'env://'
        world_size = ntasks
        rank = proc_id
        gpu = proc_id % num_gpus
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus: {num_gpus}, addr:{addr}, master port: {master_port}')
    else:
        print('Not using distributed mode')
        distributed = False
        return
    distributed = True
    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()