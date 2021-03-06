import os
import yaml
import torch
import torch.nn as nn
import argparse
import src.utils.save as save
import src.utils.logging as logging
import src.utils.ddp as ddp
import tqdm
import time
import shutil
from torchvision import transforms
from thop.profile import profile
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class runner():
    def __init__(self, config_path, exp_name):        
        loca=time.strftime('%Y-%m-%d-%H-%M-%S')
        self.config_path = config_path
        self.exp_name = os.path.join(exp_name, loca)

        with open(self.config_path, 'r') as f: 
            self.config = yaml.safe_load(f)
        if 'ddp' in self.config['basic']:
            ddp.init_distributed_mode(self.config['basic']['distributed'])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ["RANK"])
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.local_rank=0
            self.rank = 0

        cwd = os.getcwd()
        self.filepath = 'exp/%s/' % self.exp_name
        if not os.path.exists(self.filepath) and self.rank == 0:
            os.makedirs(self.filepath)
            os.makedirs(self.filepath + 'logdir')
            os.makedirs(self.filepath + 'checkpoint')
        else:
            raise AssertionError("File has existed, please delete the file.")
        
        if self.rank == 0:
            with open(os.path.join("exp", self.exp_name, "config.yaml"), 'w') as f:
                yaml.dump(self.config, f)
            self.writer = SummaryWriter(os.path.join('exp', self.exp_name, 'logdir'))
            self.mmcv_logger = logging.get_logger('mmcv_logger', log_file=os.path.join('exp', self.exp_name, 'output.log')) 
            self.mmcv_logger.info('Train log')

        if self.rank == 0:
            self.mmcv_logger.info(self.config)

        self.seed = self.config['basic']['seed']
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)

        self.max_epoch, self.batch_size, self.lr, self.num_workers = \
            self.config['train']['max_epoch'], \
            self.config['train']['batch_size'], \
            float(self.config['train']['lr']), \
            self.config['train']['num_workers']

        self.train_transforms = []
        if 'train' in self.config['dataset']:
            if 'Resize' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.Resize(self.config['dataset']['train']['Resize']))
            if 'RandomHorizontalFlip' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.RandomHorizontalFlip())
            if 'RandomVerticalFlip' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.RandomVerticalFlip())
            if 'RandomCrop' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.RandomCrop(self.config['dataset']['train']['RandomCrop']))
            if 'ColorJitter' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.ColorJitter(*self.config['dataset']['train']['ColorJitter']))
            if 'RandomRotation' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.RandomRotation(self.config['dataset']['train']['RandomRotation']))
            self.train_transforms.append(transforms.ToTensor())
            if 'Normalize' in self.config['dataset']['train']:
                self.train_transforms.append(transforms.Normalize(mean=self.config['dataset']['train']['Normalize']['mean'],
                                                                std=self.config['dataset']['train']['Normalize']['std']))
        else:
            self.train_transforms.append(transforms.ToTensor())

        self.test_transforms = []
        if 'test' in self.config['dataset']:
            if 'Resize' in self.config['dataset']['test']:
                self.test_transforms.append(transforms.Resize(self.config['dataset']['test']['Resize']))
            self.test_transforms.append(transforms.ToTensor())
            if 'Normalize' in self.config['dataset']['test']:
                self.test_transforms.append(transforms.Normalize(mean=self.config['dataset']['test']['Normalize']['mean'],
                                                                std=self.config['dataset']['test']['Normalize']['std']))
        else:
            self.train_transforms.append(transforms.ToTensor())
        
        self.train_transforms = transforms.Compose(self.train_transforms)
        self.test_transforms = transforms.Compose(self.test_transforms)

        if 'write_iter' in self.config['basic']:
            self.write_iter = self.config['basic']['write_iter']
        else:
            self.write_iter = 100

        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
    
    def set_data(self):
        self.train_loader, self.test_loader = None

    def set_model(self, model):
        self.model = None

    def calc_params_macs_time(self, input=None):
        if input is None:
            input = torch.randn(1, 3, 224, 224)
        if self.config['basic']['device'] == 'gpu':
            input = input.cuda()
        macs, params = profile(self.model, inputs=(input, ))
        start = time.time()
        _ = self.model(input)
        time_used = time.time()-start
        return macs, params, time_used

    def train(self, train_one_epoch, test_one_epoch):
        if self.config['basic']['device'] == 'gpu':
            torch.cuda.manual_seed(self.seed)
            self.model = self.model.cuda()
            self.model_without_ddp = self.model
            if 'dp' in self.config['basic']:
                self.model = torch.nn.DataParallel(self.model)
                self.model_without_ddp = self.model.module
            if 'ddp' in self.config['basic']:
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
                self.model_without_ddp = self.model.module
        
        if self.rank == 0:
            self.mmcv_logger.info("Device: {}".format(next(self.model.parameters()).device))
            self.mmcv_logger.info("\n{}".format(self.model))
            self.mmcv_logger.info("********Training Start*******")
        
        optimizer_config = self.config['train']['optimizer']
        if optimizer_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,\
                betas=(optimizer_config['beta1'], optimizer_config['beta2']))
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,\
                momentum=optimizer_config['momentum'], weight_decay=float(optimizer_config['weight_decay']))

        schedule_config = self.config['train']['schedule']
        if schedule_config['type'] == 'Cosine':
            minlr = 0.0
            period = 0
            if schedule_config['minlr'] is not None:
                minlr = float(schedule_config['minlr'])
            if schedule_config['period'] is not None:
                period = schedule_config['period']
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR\
                (self.optimizer, period, eta_min=minlr, last_epoch=-1, verbose=False)
        
        elif schedule_config['type'] == 'MultiStep':
            gamma = 0.1
            milestones = []
            if schedule_config['gamma'] is not None:
                gamma = schedule_config['gamma']
            if schedule_config['milestones'] is not None:
                milestones = schedule_config['milestones']
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR\
                (self.optimizer, milestones, gamma=gamma, last_epoch=-1, verbose=False)


        best_acc = 0
        for i in range(self.max_epoch):
            if self.rank == 0:
                self.mmcv_logger.info('-----Epoch [{}/{}] START-----'.format(i + 1, self.max_epoch))
                self.writer.add_scalar('Lr/%s' % ('train'), self.optimizer.state_dict()['param_groups'][0]['lr'], i)

            train_one_epoch(i, self.max_epoch)
            acc = test_one_epoch(i, self.max_epoch)
            
            if self.config['basic']['save']['best'] == True and self.rank == 0:
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc
                    save.save_checkpoint({
                        'epoch': i,
                        'model': self.model_without_ddp.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc': best_acc
                    }, is_best, self.filepath + 'checkpoint/', "ep" + str(i) + '/')
            
            if (i+1) % self.config['basic']['save']['period'] == 0 and self.rank == 0:
                save.save_checkpoint({
                    'epoch': i,
                    'model': self.model_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_acc': best_acc
                }, is_best, self.filepath + 'checkpoint/', "ep" + str(i) + '/')            
            if self.rank == 0:
                self.mmcv_logger.info('-----Epoch [{}/{}] END  -----'.format(i + 1, self.max_epoch))
        if self.rank == 0:
            self.mmcv_logger.info("********Training End*********")
    
    def load_model(self, test_one_image=None, ckptpath=None):
        if ckptpath is None:
            ckptpath=os.path.join(self.filepath, "checkpoint/best/model_best.pth")
        self.model = self.model.cpu()
        checkpoint = torch.load(ckptpath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        if self.config['basic']['device'] == 'gpu':
            torch.cuda.manual_seed(self.seed)
            self.model = self.model.cuda()
        if self.rank == 0:
            self.mmcv_logger.info("Device: {}".format(next(self.model.parameters()).device))
            macs, params, time_used = self.calc_params_macs_time(test_one_image)
            self.mmcv_logger.info("MACs: {:.5f}G, Params: {:.5f}M, Inference Time: {:.5f}MS".format(macs / 1e9, params / 1e6, time_used * 1e3))
            self.mmcv_logger.info("\n{}".format(self.model))
            self.mmcv_logger.info("**********Load End***********")