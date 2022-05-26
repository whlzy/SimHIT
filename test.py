import pandas as pd
import os
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import src.model.resnet as resnet
import src.model.senet as senet
import src.model.vgg as vgg
import src.data.PlantSeedlings as PlantSeedlings
import shutil
from torch.utils.data import Dataset, DataLoader

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

filepath = './exp/exp_plantseedlings_test'
img_path = './datasets/Plant_Seedlings/test'
batch_size = 32
num_workers = 4
images = []


def test_data_list(path):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)
    with open(os.path.join(filepath, 'test.txt'), 'a+') as f:
        imgs = sorted(os.listdir(path))
        for img in imgs:
            img_path = os.path.join(path, img)
            f.write(img_path + ' 0\n')
            images.append(img)


test_data_list(img_path)  # test.txt
test_data = PlantSeedlings.PlantSeedlingsDataset(listpath=os.path.join(filepath, 'test.txt'), transform=test_transforms)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# resnet18
model_resnet = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], 12)
model_resnet.load_state_dict(torch.load('./exp/exp_plantseedlings/test_resnet/2022-05-26-10-45-41/checkpoint/best/model_best.pth')['model'])
# vgg11
model_vgg = vgg.VGG(layers=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ], batch_norm=False, num_classes=12)
model_vgg.load_state_dict(torch.load('./exp/exp_plantseedlings/test_vgg/2022-05-26-13-10-43/checkpoint/best/model_best.pth')['model'])
# senet18
model_senet = senet.senet(num=[2, 2, 2, 2], classes=12)
model_senet.load_state_dict(torch.load('./exp/exp_plantseedlings/test_senet/2022-05-26-14-21-05/checkpoint/best/model_best.pth')['model'])


labels = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen',
          6: 'Loose Silky-bent', 7: 'Maize', 8: 'Scentless Mayweed', 9: 'Shepherds Purse', 10: 'Small-flowered Cranesbill',
          11: 'Sugar beet'}


results = np.array([])
for (image, label) in test_loader:
    outputs = model_senet(image)
    _, predicted = torch.max(outputs.data, 1)
    results = np.append(results, predicted.detach().numpy().astype(int))

cats = []
for res in results:
    cats.append(labels[int(res)])
# print(images)
# print(cats)

pd_all = pd.DataFrame({'file': images, 'species': cats})
pd_all.to_csv(os.path.join(filepath, 'senet18.csv'), header=True, index=False, sep=',', encoding='UTF-8')

