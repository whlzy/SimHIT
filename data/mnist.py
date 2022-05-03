import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms

def get_dataset(path, batch_size):
    train_dataset = datasets.MNIST(root=path,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)

    test_dataset = datasets.MNIST(root=path,
                                train=False,
                                transform=transforms.ToTensor())                                                    

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return train_loader, test_loader