import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import shutil


# 记录数据集到txt文件
def split(s_path, rate):
    s_path = os.path.join(s_path, 'train')
    lists = sorted(os.listdir(s_path))
    c = 0
    # 循环每个类下的图片
    for l in lists:
        imgs = sorted(os.listdir(os.path.join(s_path, l)))
        train_len = int(len(imgs) * rate)  # 训练集数量

        with open('./datasets/Plant_Seedlings/total.txt', 'a+') as fh:
            for img in imgs:
                path = os.path.join('./datasets/Plant_Seedlings/train', l, img)
                fh.write(path + '  ' + str(c) + '\n')

        # 划分训练集
        with open('./datasets/Plant_Seedlings/train.txt', 'a+') as f:
            for i in range(train_len):
                path = os.path.join('./datasets/Plant_Seedlings/train', l, imgs[i])
                f.write(path + '    ' + str(c) + '\n')  # 将图片路径与图片标签写入train.txt文件

        # 划分验证集
        with open('./datasets/Plant_Seedlings/val.txt', 'a+') as f:
            for i in range(train_len, len(imgs)):
                path = os.path.join('./datasets/Plant_Seedlings/train', l, imgs[i])
                f.write(path + '    ' + str(c) + '\n')

        with open('./datasets/Plant_Seedlings/labels.txt', 'a+') as f:
            f.write(str(c) + '  ' + l + '\n')

        c = c + 1



# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


# 创建自定义的数据集：继承torch.utils.data.Dataset
class PlantDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):  # 初始化参数：txt-存入数据的路径；transform-数据转换格式
        super(PlantDataset, self).__init__()  # 对继承来自父类的属性进行初始化
        imgs = []
        with open(txt, 'r') as f:  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
            for line in f:
                line = line.strip('\n')
                line = line.rstrip('\n')  # 删除本行字符串末尾的指定字符
                data = line.split('    ')  # 用空格分割本行数据
                imgs.append((data[0], int(data[1])))  # imgs包含图片路径和标签信息
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 该方法用于按照索引读取每个元素的具体内容
        path, label = self.imgs[index]  # path:图像路径；label: 图像标签
        img = self.loader(path)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 图像数据转换为Tensor
        # 标签数据转换为Tensor
        label_numpy = np.array(label)  # 将python类型的数据转成numpy
        label_tensor = torch.from_numpy(label_numpy)  # numpy -> tensor
        return img, label_tensor

    def __len__(self):
        return len(self.imgs)


def get_dataset(data_path, rate, batch_size):
    split(data_path, rate)
    # 数据转换
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
            transforms.RandomCrop(224),
            transforms.RandomRotation(90),
            transforms.ToTensor(),  # 将给定图像转为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }
    train_data = PlantDataset(txt=os.path.join(data_path, 'train.txt'), transform=data_transform['train'])
    val_data = PlantDataset(txt=os.path.join(data_path, 'val.txt'), transform=data_transform['val'])
    # 将数据集加载到DataLoader中
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, val_loader




