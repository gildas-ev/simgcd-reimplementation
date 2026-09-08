import torch
import torchvision
from torchvision.datasets import CIFAR100

class CIFAR100WithIndex(torchvision.datasets.CIFAR100):
    def __init__(self):
        self.data = CIFAR100(root='./data', train=True, download=True)

    def __getitem__(self, idx):
        return self.data[idx]

