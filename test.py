#coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab
import numpy as np  #二维数组
from torch.autograd import Variable
import torch
import torch.nn as nn
import matplotlib
from PIL import Image
import os


data2 = sio.loadmat("points.mat")

# matplotlib.use("TkAgg")
'''
载入训练数据
'''
data = sio.loadmat('points.mat')
xx_train=np.array(data['xx'])
plt.figure(1)
plt.scatter(xx_train[:,0],xx_train[:,1])
plt.show()

plt.savefig("filename.png")