import pandas as pd
import os
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import src.model.resnet as resnet

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model_path = "./model_best.pth"
# device = torch.device('cuda')
model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], 12)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
classes = np.arange(0, 12)

def prediect(img_path):
    # model = model.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    # img_ = img.to(device)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :' , classes[predicted[0]])


if __name__ == '__main__':
    prediect('./test/00c47e980.jpg')
