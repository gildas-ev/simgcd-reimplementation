from config import CONFIG
from models.head import Head

import torch

def test_contract():
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)

    for B in [1, 2, 4, 128]:
        batch = torch.randn(B, CONFIG.features_dim)
        x_proj, logits = head(batch)
        assert x_proj.shape == (B, CONFIG.bottleneck_dim)
        assert x_proj.dtype == torch.float32 and torch.isfinite(x_proj).all()
        assert logits.shape == (B, CONFIG.out_dim)
        assert logits.dtype == torch.float32 and torch.isfinite(x_proj).all()

def test_bounded():
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)

    batch = torch.randn(256, CONFIG.features_dim)
    _, logits = head(batch)
    assert -1 - 1e-6 <= torch.min(logits) and torch.max(logits) <= 1 + 1e-6

def test_scale():
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)

    batch = torch.randn(256, CONFIG.features_dim)
    scale_batch = 4.2*batch
    
    proj, logits = head(batch)
    scale_proj, scale_logits = head(scale_batch)
    dist = torch.linalg.norm(logits-scale_logits, float("inf")).item()

    assert dist <= 1e-5
    assert torch.isclose(proj, scale_proj).sum() <= 5

def test_head_numel():
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)

    nb_params_tot = 0
    for param in head.parameters():
        nb_params_tot += param.numel()
    
    assert nb_params_tot == 6372608
