import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader
from torchvision.datasets import CIFAR100

class MultiViewTransform:
    """Transform function to generate multiple views of train images"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, img):
        return [self.base_transform(img) for _ in range(self.n_views)]

class WrapperCIFAR(torch.utils.data.Dataset):
    """Merges train_labelled and train_unlabelled into one dataset

    Furthermore, __getitem__ returns an extra bool for labelled or
    unlabelled data"""
    def __init__(self, train_labelled, train_unlabelled):
        self.train_labelled = train_labelled
        self.train_unlabelled = train_unlabelled

    def __len__(self):
        return len(self.train_labelled) + len(self.train_unlabelled)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError()
        if idx < len(self.train_labelled):
            x, y = self.train_labelled[idx]
            return (x, y, True)
        else: # len(self.train_labelled) <= idx < len(self)
            x, y = self.train_unlabelled[idx-len(self.train_labelled)]
            return (x, y, False)

def _subsample(indices, n, rng):
    if n is None or n >= len(indices):
        return indices
    return np.sort(rng.choice(indices, size=n, replace=False))

def build_transforms(img_size_encoder=224, crop_pct=0.875):
    """Builds transform functions for the train dataset and the test dataset

    Returns two callables"""
    mean = [0.485, 0.456, 0.406] # hardcode encoder normalize
    std = [0.229, 0.224, 0.225]

    resize_size = int(img_size_encoder/crop_pct)
    
    train_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=3),
        transforms.RandomCrop(img_size_encoder),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=3),
        transforms.CenterCrop(img_size_encoder),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform

def build_sampler(wrapper_cifar):
    """Builds the sampler to give to the DataLoader
    to balance the labelled and unlabelled items in a batch"""
    weights = np.zeros(len(wrapper_cifar))

    idxs_labelled = np.arange(len(wrapper_cifar.train_labelled))
    idxs_unlabelled = np.arange(len(wrapper_cifar.train_labelled),
                                len(wrapper_cifar))

    weights[idxs_labelled] = 1
    weights[idxs_unlabelled] = len(wrapper_cifar.train_labelled)/len(wrapper_cifar.train_unlabelled)

    n = len(wrapper_cifar.train_labelled)
    assert np.isclose(weights[:n].sum(), weights[n:].sum())

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(wrapper_cifar),
        replacement=True
    )

def get_cifar100_datasets(train_transform, test_transform, config):
    """Load CIFAR100, returns a dict with 4 datasets, respectively
    train_labelled, train_unlabelled, train_unlabelled_eval, test"""

    rng = np.random.default_rng(seed=config.seed)

    data_train_transform = CIFAR100(root=config.path_dataset, train=True, download=config.download, transform=MultiViewTransform(train_transform, config.n_views))
    data_test_transform = CIFAR100(root=config.path_dataset, train=True, download=config.download, transform=test_transform)
    
    labels_old_classes = list(range(config.num_old_classes))
    indexes_old_classes = [idx for idx, label in enumerate(data_train_transform.targets) if label in labels_old_classes]    
    nb_img_old_classes = len(indexes_old_classes)
    
    indexes_random_old_classes = rng.choice(indexes_old_classes, size=int(config.prop_train_labels*nb_img_old_classes), replace=False)
    indexes_complement = np.setdiff1d(np.arange(len(data_train_transform)), indexes_random_old_classes)
   
    targets = np.array(data_train_transform.targets)
    assert set(targets[indexes_random_old_classes].tolist()) == set(range(config.num_old_classes))
    assert (targets[indexes_random_old_classes] < config.num_old_classes).sum() == 50000*config.num_old_classes/100*config.prop_train_labels
    assert (targets[indexes_complement] >= config.num_old_classes).sum() == 50000*(100-config.num_old_classes)/100

    if config.max_train_samples is not None:
        r = config.max_train_samples / len(data_train_transform)
        indexes_random_old_classes = _subsample(
            indexes_random_old_classes,
            int(r*len(indexes_random_old_classes)),
            rng
        )
        indexes_complement = _subsample(
            indexes_complement,
            int(r*len(indexes_complement)),
            rng
        )

    indexes_eval = _subsample(indexes_complement, config.max_eval_samples, rng)

    train_labelled = Subset(data_train_transform, indexes_random_old_classes)
    train_unlabelled = Subset(data_train_transform, indexes_complement)
    train_unlabelled_eval = Subset(data_test_transform, indexes_eval)
    test = CIFAR100(root=config.path_dataset, train=False, download=config.download, transform=test_transform)

    return {
        'train_labelled':train_labelled,
        'train_unlabelled':train_unlabelled,
        'train_unlabelled_eval':train_unlabelled_eval,
        'test':test
    }
