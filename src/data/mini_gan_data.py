import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import matplotlib
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import shutil

class gan_data(Dataset):
    def __init__(self, path):
        super(gan_data, self).__init__()
        self.orgindata = sio.loadmat(path)
        self.traindata = np.array(self.orgindata['xx'])

    def __getitem__(self, index):
        data = self.traindata[index, :]
        return data

    def __len__(self):
        return self.orgindata['xx'].shape[0]

def get_dataset(datapath, batch_size, num_workers=2):
    train_data = gan_data(path=datapath)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader