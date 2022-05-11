import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import shutil


def split(datapath, filepath, rate):
    datapath = os.path.join(datapath, 'train')
    lists = sorted(os.listdir(datapath))
    filepath = os.path.join(filepath, 'datalist')

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)
    
    num = 0
    for list in lists:
        imgs = sorted(os.listdir(os.path.join(datapath, list)))
        train_len = int(len(imgs) * rate)

        with open(os.path.join(filepath, 'total.txt'), 'a+') as fh:
            for img in imgs:
                path = os.path.join(datapath, list, img)
                fh.write(path + '  ' + str(num) + '\n')

        with open(os.path.join(filepath, 'train.txt'), 'a+') as f:
            for i in range(train_len):
                path = os.path.join(datapath, list, imgs[i])
                f.write(path + '    ' + str(num) + '\n')

        with open(os.path.join(filepath, 'val.txt'), 'a+') as f:
            for i in range(train_len, len(imgs)):
                path = os.path.join(datapath, list, imgs[i])
                f.write(path + '    ' + str(num) + '\n')

        with open(os.path.join(filepath, 'labels.txt'), 'a+') as f:
            f.write(str(num) + '  ' + list + '\n')

        num = num + 1


def default_loader(path):
    return Image.open(path).convert('RGB')


class PlantSeedlingsDataset(Dataset):
    def __init__(self, listpath, transform=None, loader=default_loader):
        super(PlantSeedlingsDataset, self).__init__()
        imgs = []
        with open(listpath, 'r') as f:
            for line in f:
                line = line.strip('\n')
                line = line.rstrip('\n')
                data = line.split('    ')
                imgs.append((data[0], int(data[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        label_numpy = np.array(label)
        label_tensor = torch.from_numpy(label_numpy)
        return img, label_tensor

    def __len__(self):
        return len(self.imgs)


def get_dataset(datapath, rate, train_transforms, test_transforms, batch_size, filepath):
    split(datapath, filepath, rate)
    
    train_data = PlantSeedlingsDataset(listpath=os.path.join(filepath, 'datalist', 'train.txt'),
                                       transform=transforms.Compose(train_transforms))
    val_data = PlantSeedlingsDataset(listpath=os.path.join(filepath, 'datalist', 'val.txt'),
                                     transform=transforms.Compose(test_transforms))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader