import os
import yaml
import torch
import torch.nn as nn
import argparse
import src.utils.save as save
import src.utils.logging as logging
import src.model.mlp as mlp
import src.data.mnist as mnist
import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter

class runner():
    def __init__(self, config_path, exp_name):
        print(self.config_path)
        self.config_path = config_path
        self.exp_name = exp_name

        cwd = os.getcwd()
        self.filepath = 'exp/%s/' % exp_name
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
            os.makedirs(self.filepath + 'logdir')
            os.makedirs(self.filepath + 'checkpoint')
        else:
            raise AssertionError("File has existed, please delete the file.")

        self.writer = SummaryWriter(os.path.join('exp', exp_name, 'logdir'))

        self.mmcv_logger = logging.get_logger('mmcv_logger', log_file=os.path.join('exp', exp_name, 'output.log')) 
        self.mmcv_logger.info('Train log')

        with open(self.config_path, 'r') as f: 
            self.config = yaml.safe_load(f)
        self.mmcv_logger.info(self.config)

        self.seed = self.config['basic']['seed']
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)

        self.max_epoch, self.batch_size, self.lr = \
            self.config['train']['max_epoch'], \
            self.config['train']['batch_size'], \
            float(self.config['train']['lr'])
        
        if self.config['dataset']['name'] == 'mnist':
            self.train_loader, self.test_loader = mnist.get_dataset(self.config['dataset']['path'], self.batch_size)

        self.model = None
        self.optimizer = None
        
    def train(self, model, train_one_epoch, test_one_epoch):
        print(model)
        self.model = model(**self.config['model'])
        if self.config['basic']['device'] == 'gpu':
            torch.cuda.manual_seed(self.seed)
            self.model = self.model.cuda()
        self.mmcv_logger.info(self.model)

        optimizer_config = self.config['train']['optimizer']
        if optimizer_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(optimizer_config['beta1'], optimizer_config['beta2']))
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])

        best_acc = 0
        for i in range(self.max_epoch):
            self.mmcv_logger.info('-----Epoch [{}/{}] START-----'.format(i + 1, self.max_epoch))
            
            train_one_epoch(i, self.max_epoch)
            acc = test_one_epoch(i, self.max_epoch)
            
            if self.config['basic']['save']['best'] == True:
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc
                    save.save_checkpoint({
                        'epoch': i,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc': best_acc
                    }, is_best, self.filepath + 'checkpoint/', "ep" + str(i) + '/')
            
            if (i+1) % self.config['basic']['save']['period'] == 0:
                save.save_checkpoint({
                    'epoch': i,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_acc': best_acc
                }, is_best, self.filepath + 'checkpoint/', "ep" + str(i) + '/')            
            
            self.mmcv_logger.info('-----Epoch [{}/{}] END  -----'.format(i + 1, self.max_epoch))
