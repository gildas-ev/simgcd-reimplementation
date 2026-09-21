from config import CONFIG
from models.backbone import build_backbone, unfreeze_vit_from
from data.cifar import build_transforms, get_cifar100_datasets

import torch
from torch.utils.data import DataLoader

def test_encoder_contract():
    backbone = build_backbone(CONFIG.encoder_name)
    backbone.eval()

    x = torch.randn(2, 3, CONFIG.img_size_encoder, CONFIG.img_size_encoder)
    with torch.no_grad():
        y = backbone(x)
    assert y.shape == (2, CONFIG.features_dim)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    
    x = torch.randn(4, 3, CONFIG.img_size_encoder, CONFIG.img_size_encoder)
    with torch.no_grad():
        y = backbone(x)
    assert y.shape == (4, CONFIG.features_dim)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()

def test_backbone_numel():
    backbone = build_backbone(CONFIG.encoder_name)
    unfreeze_vit_from(backbone, 11) # unfreeze only last block for the test
    nb_params_tot = 0
    nb_params_trainable = 0
    for name, param in backbone.named_parameters():
        nb_params_tot += param.numel()
        if param.requires_grad:
            nb_params_trainable += param.numel()

    assert nb_params_tot == 85798656
    assert nb_params_trainable == 7087872
