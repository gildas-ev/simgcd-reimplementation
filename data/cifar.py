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

def build_transforms():
    """Builds transform functions for the train dataset and the test dataset

    Returns two callables"""
    mean = [0.485, 0.456, 0.406] # hardcode encoder normalize
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
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

def get_gcd_datasets(train_transform, test_transform,
                     path_dataset,
                     num_old_classes = 80,
                     prop_train_labels = 0.5):
    """Load CIFAR100, returns a dict with 4 datasets, respectively
    train_labelled, train_unlabelled, train_unlabelled_eval, test"""

    rng = np.random.default_rng(seed=0)

    data_train_transform = CIFAR100(root=path_dataset, train=True, download=True, transform=MultiViewTransform(train_transform, 2))
    data_test_transform = CIFAR100(root=path_dataset, train=True, download=True, transform=test_transform)
    
    labels_old_classes = list(range(num_old_classes))
    indexes_old_classes = [idx for idx, label in enumerate(data_train_transform.targets) if label in labels_old_classes]    
    nb_img_old_classes = len(indexes_old_classes)
    
    indexes_random_old_classes = rng.choice(indexes_old_classes, size=int(prop_train_labels*nb_img_old_classes), replace=False)
    indexes_complement = np.setdiff1d(np.arange(len(data_train_transform)), indexes_random_old_classes)
   
    targets = np.array(data_train_transform.targets)
    assert set(targets[indexes_random_old_classes].tolist()) == set(range(80))
    assert (targets[indexes_random_old_classes] < 80).sum() == 20000
    assert (targets[indexes_complement] >= 80).sum() == 10000

    train_labelled = Subset(data_train_transform, indexes_random_old_classes)
    train_unlabelled = Subset(data_train_transform, indexes_complement)
    train_unlabelled_eval = Subset(data_test_transform, indexes_complement)
    test = CIFAR100(root=path_dataset, train=False, download=True, transform=test_transform)

    return {
        'train_labelled':train_labelled,
        'train_unlabelled':train_unlabelled,
        'train_unlabelled_eval':train_unlabelled_eval,
        'test':test
    }

if __name__ == "__main__":
    ### Exemple de ce qui serait dans le main et quelques tests
    train_transform, test_transform = build_transforms()
    result = get_gcd_datasets(train_transform, test_transform, '~/research/simgcd-reimplementation/datasets')
   
    assert len(result['train_labelled']) == 20000
    assert len(result['train_unlabelled']) == 30000
    assert len(result['test']) == 10000


    wrapper_cifar = WrapperCIFAR(result['train_labelled'], result['train_unlabelled'])
    sampler = build_sampler(wrapper_cifar)
    
    train_dataloader = DataLoader(
        dataset=wrapper_cifar,
        batch_size=128,
        drop_last=True,
        sampler=sampler
    )

    eval_dataloader = DataLoader(
        dataset=result['train_unlabelled_eval'],
        batch_size=256,
        drop_last=False
    )

    batch = next(iter(train_dataloader))
    images = batch[0]
    labels = batch[1]
    mask = batch[2]

    views1 = images[0]
    views2 = images[1]

    print("Views shapes", views1.shape, views2.shape, views1.dtype)
    print("Labels shape", labels.shape, labels.dtype)
    print("Mask shape", mask.shape, mask.dtype)
    print("pct labelled", torch.sum(batch[2]).item()/128*100)
