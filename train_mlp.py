import os
import yaml
import torch
import torch.nn as nn
import argparse
import src.model.mlp as mlp
import src.data.mnist as mnist
import src.runner as runner
import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter


args = argparse.ArgumentParser()
args.add_argument('--config_path', type=str, help='the path of config')
args.add_argument('--exp_name', type=str, help='the path of output')
args = args.parse_args()


class mlp_runner(runner.runner):
    def __init__(self, config_path, exp_name):
        self.config_path = config_path
        self.exp_name = exp_name
        runner.runner.__init__(self, self.config_path, self.exp_name)
        self.model = mlp.MLP

    def set_data(self):
        if self.config['dataset']['name'] == 'mnist':
            self.train_loader, self.test_loader = mnist.get_dataset(self.config['dataset']['path'], 
                self.batch_size, self.train_transforms, self.test_transforms)

    def set_model(self):
        self.model = self.model(**self.config['model'])

    def train_one_epoch(self, current_epoch, max_epoch):
        self.model.train()
        self.mmcv_logger.info("LR: {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        for i, (images, labels) in enumerate(self.train_loader):
            if self.config['basic']['device'] == 'gpu':
                images = images.cuda()
                labels = labels.cuda()
            outputs = self.model(images)
            celoss = nn.CrossEntropyLoss()
            loss = celoss(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i+1) % 100 == 0:
                self.mmcv_logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(current_epoch + 1, self.max_epoch, loss.item()))
                self.writer.add_scalar('Loss/%s' % ('train'), loss, current_epoch * len(self.train_loader) + i)
        self.lr_scheduler.step()

    def test_one_epoch(self, current_epoch, max_epoch):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            if self.config['basic']['device'] == 'gpu':
                images = images.cuda()
                labels = labels.cuda()
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        self.mmcv_logger.info('Epoch [{}/{}], Accuracy: {:.4f}'.format(current_epoch + 1, self.max_epoch, 100.0*correct/total))
        self.writer.add_scalar('Accuracy/%s' % ('test'), 100.0*correct/total, current_epoch+1)
        return correct

def main():
    runner = mlp_runner(args.config_path, args.exp_name)
    runner.set_data()
    runner.set_model()
    runner.train(runner.train_one_epoch, runner.test_one_epoch)
    runner.load_model(test_one_image=torch.randn(1, 1, 24, 24), ckptpath=os.path.join(runner.filepath, 'checkpoint/', 'best/', 'model_best.pth'))

if __name__ == "__main__":
    main()
