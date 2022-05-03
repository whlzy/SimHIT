import os
import yaml
import torch
import torch.nn as nn
import argparse
import utils.logging as logging
import model.mlp as mlp
import data.mnist
import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter

args = argparse.ArgumentParser()
args.add_argument('--config_path', type=str, help='the path of config')
args.add_argument('--exp_name', type=str, help='the path of output')
args = args.parse_args()

seed = 202205
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)

cwd = os.getcwd()
filepath = 'exp/%s/' % args.exp_name
if not os.path.exists(filepath):
    os.makedirs(filepath)
    os.makedirs(filepath + 'logdir')
    os.makedirs(filepath + 'checkpoint')
else:
    raise AssertionError("File has existed, please delete the file.")

mmcv_logger = logging.get_logger('mmcv_logger', log_file=os.path.join('exp', args.exp_name, 'output.log')) 
mmcv_logger.info('mnist log')

with open(args.config_path, 'r') as f: 
    config = yaml.safe_load(f)
mmcv_logger.info(config)

net = mlp.MLP(**config['model'])
if config['device'] == 'gpu':
    torch.cuda.manual_seed(seed)
    net = net.cuda()
mmcv_logger.info(net)

max_epoch, batch_size, lr, op_beta1, op_beta2 = \
    config['train']['max_epoch'], \
    config['train']['batch_size'], \
    float(config['train']['optimizer']['lr']), \
    config['train']['optimizer']['beta1'], \
    config['train']['optimizer']['beta2']

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(op_beta1, op_beta2))

writer = SummaryWriter(os.path.join('exp', args.exp_name, 'logdir'))

if config['dataset']['name'] == 'mnist':
    train_loader, test_loader = data.mnist.get_dataset(config['dataset']['path'], batch_size)

def save_checkpoint(state, is_best, filepath1, filepath2, filename='checkpoint.pth'):
    r'''
    -exp
        -*
            -checkpoint
                -model0
                    checkpoint.pth
                -model1
                    checkpoint.pth
                ...
                -best
                    checkpoint.pth
    '''
    directory = filepath1 + filepath2
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        if not os.path.exists(filepath1 + 'best/'):
            os.makedirs(filepath1 + 'best/')
        shutil.copyfile(filename, filepath1 + 'best/' + 'model_best.pth')

def train(current_epoch, max_epoch):
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        if config['device'] == 'gpu':
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        celoss = nn.CrossEntropyLoss()
        loss = celoss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            mmcv_logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(current_epoch + 1, max_epoch, loss.item()))
            writer.add_scalar('Loss/%s' % ('train'), loss, current_epoch * len(train_loader) + i)

def test(current_epoch, max_epoch):
    net.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        if config['device'] == 'gpu':
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    mmcv_logger.info('Epoch [{}/{}], Accuracy: {:.4f}'.format(current_epoch + 1, max_epoch, 100.0*correct/total))
    writer.add_scalar('Accuracy/%s' % ('test'), 100.0*correct/total, current_epoch+1)
    return correct

def main():
    best_acc = 0
    for i in range(max_epoch):
        mmcv_logger.info('-----Epoch [{}/{}] START-----'.format(i + 1, max_epoch))
        
        train(i, max_epoch)
        acc = test(i, max_epoch)
        
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            save_checkpoint({
                'epoch': i,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, is_best, filepath + 'checkpoint/', "ep" + str(i) + '/')
        save_checkpoint({
            'epoch': i,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, is_best, filepath + 'checkpoint/', "ep" + str(i) + '/')            
        
        mmcv_logger.info('-----Epoch [{}/{}] END  -----'.format(i + 1, max_epoch))

if __name__ == "__main__":
    main()
