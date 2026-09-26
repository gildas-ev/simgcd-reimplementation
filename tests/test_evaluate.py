from config import CONFIG
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from evaluate import acc_metrics, evaluate
from data.cifar import build_transforms, get_cifar100_datasets, WrapperCIFAR
from models.backbone import build_backbone
from models.head import Head
from models.model import Model

device = torch.device("cpu")

def test_perfect():
    y_true = np.random.randint(100, size=256)
    is_old = (y_true < 80)

    perm = np.arange(100)
    np.random.shuffle(perm)

    y_pred = perm[y_true]
    tot, old, new = acc_metrics(y_pred, y_true, is_old)
    assert np.isclose([tot, old, new], [1, 1, 1]).all()

def test_perfect_old_only():
    y_true = np.random.randint(100, size=30000)
    is_old = (y_true < 80)

    y_pred = y_true.copy()
    y_pred[~is_old] = 0 # wrong cluster

    tot, old, new = acc_metrics(y_pred, y_true, is_old)
    assert np.isclose([tot, old, new], [0.8, 1, 0], atol=0.065).all()

def test_hardcoded():
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    y_true = np.array([1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
    is_old = (y_true < 2)

    # w = [[0, 5, 1],
    #      [4, 0, 0],
    #      [1, 0, 6]]

    tot, old, new = acc_metrics(y_pred, y_true, is_old)
    assert np.isclose([tot, old, new], [15/17, 9/10, 6/7]).all()

def test_random():
    y_pred = np.random.randint(100, size=30000)
    y_true = np.random.randint(100, size=30000)
    is_old = (y_true < 80)

    tot, old, new = acc_metrics(y_pred, y_true, is_old)
    
    assert 0.02 < tot and tot < 0.03 # the tot should land in that range because of the optimal matching 

def test_evaluate():
    train_transform, test_transform = build_transforms(CONFIG.img_size_encoder, CONFIG.crop_pct)
    result = get_cifar100_datasets(train_transform, test_transform,
            CONFIG.path_dataset,
            num_old_classes=CONFIG.num_old_classes,
            prop_train_labels=CONFIG.prop_train_labels,
            n_views=CONFIG.n_views,
            download=CONFIG.download,
            seed=CONFIG.seed)

    subset_eval = Subset(
            result['train_unlabelled_eval'],
            list(range(16)))
    eval_dataloader = DataLoader(
        dataset=subset_eval,
        batch_size=4,
        drop_last=False
    )

    backbone = build_backbone(CONFIG.encoder_name)
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)
    model = Model(backbone, head)

    result = evaluate(model, eval_dataloader, CONFIG.num_old_classes, device)
    assert len(result) == 3
    assert model.training == True
