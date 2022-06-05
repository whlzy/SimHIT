import os
import yaml
import torch
import torch.nn as nn
import argparse
import src.model.edsr as edsr
import src.data.div2k as div2k
import src.data.set5 as set5
import src.runner as runner
import src.utils.sr_utils as utils
import pandas as pd
import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter


args = argparse.ArgumentParser()
args.add_argument('--config_path', type=str, help='the path of config')
args.add_argument('--exp_name', type=str, help='the path of output')
args = args.parse_args()


class edsr_runner(runner.runner):
    def __init__(self, config_path, exp_name):
        self.config_path = config_path
        self.exp_name = exp_name
        runner.runner.__init__(self, self.config_path, self.exp_name)

    def set_data(self):
        if self.config['dataset']['train']['name'] == 'div2k':
            self.train_loader = div2k.get_dataset(
                self.config['dataset']['train']['path'], 
                self.config['dataset']['train']['scale'],
                self.config['dataset']['train']['ext'],
                self.config['dataset']['train']['is_train'],
                self.config['dataset']['train']['repeat_dataset'],
                self.config['dataset']['train']['training_dataset_number'],
                self.config['dataset']['train']['rgb_range'],
                self.config['dataset']['train']['colors_channel'],
                self.config['dataset']['train']['patch_size'],
                self.batch_size,
                self.num_workers
            )
        if self.config['dataset']['test']['name'] == 'set5':
            self.test_loader = set5.get_dataset(
                self.config['dataset']['test']['path'],
                self.config['dataset']['test']['scale'],
                self.config['dataset']['test']['batch_size'],
                self.num_workers
            )
        elif self.config['dataset']['test']['name'] == 'set14':
            None

    def set_model(self):
        if self.config['model']['name'] == 'r16f64':
            self.model = edsr.make_edsr_baseline(**self.config['model']['args'])
        else:
            self.model = edsr.make_edsr(**self.config['model']['args'])

    def train_one_epoch(self, current_epoch, max_epoch):
        self.model.train()
        self.mmcv_logger.info("LR: {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        for i, (lr_tensor, hr_tensor) in enumerate(self.train_loader):
            if self.config['basic']['device'] == 'gpu':
                lr_tensor = lr_tensor.cuda()
                hr_tensor = hr_tensor.cuda()
            outputs = self.model(lr_tensor)
            l1loss = nn.L1Loss()
            loss = l1loss(outputs, hr_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (i+1) % 100 == 0:
                self.mmcv_logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(current_epoch + 1, self.max_epoch, loss.item()))
                self.writer.add_scalar('Loss/%s' % ('train'), loss, current_epoch * len(self.train_loader) + i)
        self.lr_scheduler.step()

    def test_one_epoch(self, current_epoch, max_epoch):
        self.model.eval()
        avg_psnr, avg_ssim = 0, 0
        for lr_tensor, hr_tensor in self.test_loader:
            if self.config['basic']['device'] == 'gpu':
                lr_tensor = lr_tensor.cuda()
                hr_tensor = hr_tensor.cuda()
            outputs = self.model(lr_tensor)
            sr_img = utils.tensor2np(outputs.detach()[0])
            gt_img = utils.tensor2np(hr_tensor.detach()[0])
            crop_size = self.config['dataset']['test']['scale']
            cropped_sr_img = utils.shave(sr_img, crop_size)
            cropped_gt_img = utils.shave(gt_img, crop_size)
            
            if self.config['dataset']['test']['isY'] is True:
                im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
            else:
                im_label = cropped_gt_img
                im_pre = cropped_sr_img
            avg_psnr += utils.compute_psnr(im_pre, im_label)
            avg_ssim += utils.compute_ssim(im_pre, im_label)

        avg_psnr /= len(self.test_loader)
        avg_ssim /= len(self.test_loader)
        self.mmcv_logger.info('Epoch [{}/{}], PSNR: {:.4f}, SSIM: {:.4f}'.format(current_epoch + 1, self.max_epoch, avg_psnr, avg_ssim))
        self.writer.add_scalar('PSNR/%s' % ('test'), avg_psnr, current_epoch+1)
        self.writer.add_scalar('SSIM/%s' % ('test'), avg_ssim, current_epoch+1)
        return avg_psnr


def main():
    runner = edsr_runner(args.config_path, args.exp_name)
    runner.set_data()
    runner.set_model()
    runner.train(runner.train_one_epoch, runner.test_one_epoch)


if __name__ == "__main__":
    main()