import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR100

class MultiViewTransform:
    """Transform function to generate multiple views of train images"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, img):
        return [self.base_transform(img) for _ in range(self.n_views)]

def build_transforms(image_size=224, crop_pct=0.875):
    """Buils transform functions for the train dataset and the test dataset"""
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762) 
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform

def get_gcd_datasets(train_transform, test_transform,
                     num_old_classes = 80,
                     prop_train_labels = 0.5):
    """Load CIFAR100, splits the data into 4 datasets
    """

    rng = np.random.default_rng(seed=0)

    data_train_transform = CIFAR100(root='./data', train=True, download=True, transform=MultiViewTransform(train_transform, 2))
    data_test_transform = CIFAR100(root='./data', train=True, download=True, transform=test_transform)
    
    labels_old_classes = list(range(num_old_classes))
    indexes_old_classes = [idx for idx, label in enumerate(data_train_transform.targets) if label in labels_old_classes]    
    nb_img_old_classes = len(indexes_old_classes)
    
    indexes_random_old_classes = rng.choice(nb_img_old_classes, size=int(prop_train_labels*nb_img_old_classes), replace=False)
    indexes_complement = np.setdiff1d(np.arange(len(data_train_transform)), indexes_random_old_classes)
   


    train_labelled = Subset(data_train_transform, indexes_random_old_classes)
    train_unlabelled = Subset(data_train_transform, indexes_complement)
    train_unlabelled_eval = Subset(data_test_transform, indexes_complement)
    test = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    return {
        'train_labelled':train_labelled,
        'train_unlabelled':train_unlabelled,
        'train_unlabelled_eval':train_unlabelled_eval,
        'test':test
    }

train_transform, test_transform = build_transforms()
result = get_gcd_datasets(train_transform, test_transform)
for key, dataset in result.items():
    print(key, len(dataset), dataset.__getitem__(0))
