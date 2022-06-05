import torch.utils.data as data
import os.path
import cv2
import numpy as np
import random
import torch
import skimage.color as sc
from torch.utils.data import Dataset, DataLoader


def get_patch(*args, patch_size, scale):
    ih, iw = args[0].shape[:2]

    tp = patch_size  # target patch (HR)
    ip = tp // scale  # input patch (LR)

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]  # results
    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class div2k(data.Dataset):
    def __init__(self, root, scale, ext, is_train, repeat_dataset, 
                training_dataset_number, rgb_range, colors_channel, patch_size):
        self.scale = scale
        self.root = root
        self.ext = ext   # '.png' or '.npy'(default)
        self.is_train = is_train
        self.repeat_dataset = repeat_dataset
        self.rgb_range = rgb_range
        self.colors_channel = colors_channel
        self.training_dataset_number = training_dataset_number
        self.patch_size = patch_size
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_hr = os.path.join(self.root, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X' + str(self.scale))

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = set_channel(lr, hr, n_channels=self.colors_channel)
        lr_tensor, hr_tensor = np2Tensor(lr, hr, rgb_range=self.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.is_train:
            return self.training_dataset_number * self.repeat_dataset

    def _get_index(self, idx):
        if self.is_train:
            return idx % self.training_dataset_number
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.patch_size
        scale = self.scale
        if self.is_train:
            img_in, img_tar = get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr

def get_dataset(root, scale, ext, is_train, repeat_dataset, 
                training_dataset_number, rgb_range, colors_channel, patch_size, 
                train_batch_size, test_batch_size, num_workers=2):
    train_data = div2k(root, scale, ext, is_train, repeat_dataset, training_dataset_number, rgb_range, colors_channel, patch_size)
    train_loader = DataLoader(dataset=train_data, num_workers=num_workers, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return train_loader