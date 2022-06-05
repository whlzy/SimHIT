import torch.utils.data as data
from os.path import join
from os import listdir
import os
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class set5(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        super(set5, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(img_modcrop(target, self.upscale))

        return input, target

    def __len__(self):
        return len(self.lr_filenames)


def get_dataset(dir, scale, test_batch_size, num_workers):
    hr_dir = os.path.join(dir, "Set5/")
    lr_dir = os.path.join(dir, "Set5_LR/x{}".format(scale))
    val_data = set5(hr_dir, lr_dir, scale)
    val_loader = DataLoader(dataset=val_data, num_workers=num_workers, batch_size=test_batch_size, shuffle=False)
    return val_loader