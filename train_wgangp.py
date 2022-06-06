import os
import yaml
import tqdm
import time
import torch
import shutil
import argparse
import matplotlib
import numpy as np
import pandas as pd
import torch.autograd
import torch.nn as nn
import scipy.io as sio
import src.runner as runner
import src.utils.save as save
import matplotlib.pyplot as plt
import torch.autograd as autograd
import src.utils.sr_utils as utils
import src.utils.logging as logging
import src.model.WGANGP.wgangp as wgangp
import src.data.mini_gan_data as mini_gan_data

from thop.profile import profile
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import  save_image
from torch.utils.tensorboard import SummaryWriter

args = argparse.ArgumentParser()
args.add_argument('--config_path', type=str, help='the path of config')
args.add_argument('--exp_name', type=str, help='the path of output')
args = args.parse_args()



class gan_runner(runner.runner):
    def __init__(self, config_path, exp_name):
        self.config_path = config_path
        self.exp_name = exp_name
        runner.runner.__init__(self, self.config_path, self.exp_name)
        self.generator = None
        self.discriminator = None
        self.optimizer_generator = None
        self.optimizer_discriminator = None
        self.generator_lr = None
        self.discriminator_lr = None
        self.z_dimension = self.config['model']['z_dimension']
        self.orgindata = sio.loadmat(self.config['dataset']['path'])
        self.traindata = np.array(self.orgindata['xx'])


    def set_data(self):
        if self.config['dataset']['name'] == 'mini_gan':
            self.train_loader = mini_gan_data.get_dataset(self.config['dataset']['path'], self.batch_size, self.num_workers)

    def set_model(self):
        self.generator = wgangp.Generator(self.config['model']['data_dim'], self.config['model']['hidden_dim'])
        self.discriminator = wgangp.Discriminator(self.config['model']['data_dim'], self.config['model']['hidden_dim'])

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda(0)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda(0)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(0),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config['train']['gp_lambda']
        return gradient_penalty


    def train_one_epoch(self, current_epoch, max_epoch):
        self.generator.train()
        self.discriminator.train()
        self.mmcv_logger.info("Generator LR: {}".format(self.optimizer_generator.state_dict()['param_groups'][0]['lr']))
        self.mmcv_logger.info("Discriminator LR: {}".format(self.optimizer_discriminator.state_dict()['param_groups'][0]['lr']))

        for i, (batch) in enumerate(self.train_loader):
            if self.config['basic']['device'] == 'gpu':
                batch = batch.float().cuda()

            one = torch.FloatTensor(batch.shape[0], 1).zero_() + 1
            minus_one = -1 * one

            for param in self.discriminator.parameters():
                param.requires_grad = True  # to avoid computation
            
            self.discriminator.zero_grad()
            real_point = Variable(batch).cuda()
            real_out = self.discriminator(real_point)
            real_out.backward(minus_one.cuda())
            z_discriminator = Variable(torch.randn(batch.shape[0], self.z_dimension)).cuda()
            fake_point = Variable(self.generator(z_discriminator)).cuda()
            fake_out = self.discriminator(fake_point)
            fake_out.backward(one.cuda())
            gradient_penalty = self.calc_gradient_penalty(real_point.data, fake_point.data)
            gradient_penalty.backward()
            self.optimizer_discriminator.step()
            discriminator_loss = fake_out - real_out + gradient_penalty
            wasserstein_distance = real_out - fake_out

            for param in self.discriminator.parameters():
                param.requires_grad = False
            
            self.generator.zero_grad()
            self.discriminator.zero_grad()
            z_generator = Variable(torch.randn(batch.shape[0], self.z_dimension)).cuda()
            fake_point = self.generator(z_generator)
            real_out = self.discriminator(fake_point)
            real_out.backward(minus_one.cuda())
            self.optimizer_generator.step()
            generator_loss = -real_out
            
            if (i+1) % self.write_iter == 0:
                self.mmcv_logger.info('Epoch [{}/{}], Generator Loss: {:.4f}'.format(current_epoch + 1, self.max_epoch, generator_loss.mean().item()))
                self.mmcv_logger.info('Epoch [{}/{}], Discriminator Loss: {:.4f}'.format(current_epoch + 1, self.max_epoch, discriminator_loss.mean().item()))
                self.mmcv_logger.info('Epoch [{}/{}], Wasserstein Distance: {:.4f}'.format(current_epoch + 1, self.max_epoch, wasserstein_distance.mean().item()))
                self.writer.add_scalar('Loss/%s' % ('train_generator'), generator_loss.mean().item(), current_epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/%s' % ('train_discriminator'), discriminator_loss.mean().item(), current_epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/%s' % ('Wasserstein Distance'), wasserstein_distance.mean().item(), current_epoch * len(self.train_loader) + i)

        x =  np.linspace(-1.2, 2.4, 200)
        y =  np.linspace(-1, 1.8, 200)
        X, Y = np.meshgrid(x, y)
        m, n = X.shape
        point = []
        for i in range(m):
            for j in range(n):
                point.append([X[i][j], Y[i][j]])
        point = np.array(point)
        fake_images = fake_point.cpu().data
        point_data = point.astype(np.float32)
        point_data = torch.from_numpy(point_data)
        point_data = point_data.cuda()
        decision = self.discriminator(point_data)
        plt.cla()
        fig = plt.figure(1, figsize=(8, 6))
        plt.scatter(point[:, 0], point[:, 1], c=decision.data.cpu().numpy()[:, 0], marker='.',cmap='gray')
        plt.scatter(self.traindata[:, 0], self.traindata[:, 1], c='#00CED1')
        plt.scatter(fake_images[:, 0], fake_images[:, 1], c='#0C143F')
        plt.draw()
        self.writer.add_figure('Visualization', fig, current_epoch+1)
        

    def train(self, train_one_epoch, test_one_epoch):
        if self.config['basic']['device'] == 'gpu':
            torch.cuda.manual_seed(self.seed)
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        self.mmcv_logger.info("Device: {}".format(next(self.generator.parameters()).device))
        self.mmcv_logger.info("Generator:\n{}".format(self.generator))
        self.mmcv_logger.info("Discriminator:\n{}".format(self.discriminator))
        self.mmcv_logger.info("********Training Start*******")

        optimizer_config_generator = self.config['train']['optimizer_generator']
        optimizer_config_discriminator = self.config['train']['optimizer_discriminator']

        self.generator_lr = float(optimizer_config_generator['lr'])
        self.discriminator_lr = float(optimizer_config_discriminator['lr'])

        if optimizer_config_generator['type'] == 'adamw':
            self.optimizer_generator = torch.optim.AdamW(self.generator.parameters(), lr=self.generator_lr,\
                betas=(optimizer_config_generator['beta1'], optimizer_config_generator['beta2']))
        elif optimizer_config_generator['type'] == 'sgd':
            self.optimizer_generator = torch.optim.SGD(self.generator.parameters(), lr=self.generator_lr,\
                momentum=optimizer_config_generator['momentum'], weight_decay=float(optimizer_config_generator['weight_decay']))
        elif optimizer_config_generator['type'] == 'RMSprop':
            self.optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr=self.generator_lr,\
                alpha=float(optimizer_config_generator['alpha']))

        if optimizer_config_discriminator['type'] == 'adamw':
            self.optimizer_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=self.discriminator_lr,\
                betas=(optimizer_config_discriminator['beta1'], optimizer_config_discriminator['beta2']))
        elif optimizer_config_discriminator['type'] == 'sgd':
            self.optimizer_discriminator = torch.optim.SGD(self.discriminator.parameters(), lr=self.discriminator_lr,\
                momentum=optimizer_config_discriminator['momentum'], weight_decay=float(optimizer_config_discriminator['weight_decay']))
        elif optimizer_config_discriminator['type'] == 'RMSprop':
            self.optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.discriminator_lr,\
                alpha=float(optimizer_config_discriminator['alpha']))

        best_acc = 0
        for i in range(self.max_epoch):
            self.mmcv_logger.info('-----Epoch [{}/{}] START-----'.format(i + 1, self.max_epoch))
            
            self.writer.add_scalar('Lr/%s' % ('train_generator'), self.optimizer_generator.state_dict()['param_groups'][0]['lr'], i)
            self.writer.add_scalar('Lr/%s' % ('train_discriminator'), self.optimizer_discriminator.state_dict()['param_groups'][0]['lr'], i)

            train_one_epoch(i, self.max_epoch)
            #acc = test_one_epoch(i, self.max_epoch)
            
            if self.config['basic']['save']['best'] == True:
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc
                    save.save_checkpoint({
                        'epoch': i,
                        'model_generator': self.optimizer_generator.state_dict(),
                        'model_discriminator': self.optimizer_discriminator.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc': best_acc
                    }, is_best, self.filepath + 'checkpoint/', "ep" + str(i) + '/')
            
            if (i+1) % self.config['basic']['save']['period'] == 0:
                save.save_checkpoint({
                    'epoch': i,
                    'model_generator': self.generator.state_dict(),
                    'model_discriminator': self.discriminator.state_dict(),
                    'generator_optimizer': self.optimizer_generator.state_dict(),
                    'discriminator_optimizer': self.optimizer_discriminator.state_dict()
                }, False, self.filepath + 'checkpoint/', "ep" + str(i) + '/')            
            
            self.mmcv_logger.info('-----Epoch [{}/{}] END  -----'.format(i + 1, self.max_epoch))
        self.mmcv_logger.info("********Training End*********")


def main():
    runner = gan_runner(args.config_path, args.exp_name)
    runner.set_data()
    runner.set_model()
    runner.train(runner.train_one_epoch, None)


if __name__ == "__main__":
    main()