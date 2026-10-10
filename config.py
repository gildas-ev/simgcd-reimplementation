import dataclasses
from dataclasses import dataclass
from pathlib import Path
import os

import torch

_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    # data
    path_dataset: Path = _ROOT / "datasets"
    download: bool = True
    num_old_classes: int = 80
    prop_train_labels: float = 0.5
    crop_pct: float = 0.875
    max_train_samples: int = None
    max_eval_samples: int = None

    # batchs
    batch_size_train: int = 128
    batch_size_eval: int = 256
    n_views: int = 2

    # encoder
    encoder_name: tuple = ('facebookresearch/dino:main', 'dino_vitb16')
    unfreeze_from_num: int = 11
    img_size_encoder: int = 224
    features_dim: int = 768

    # head
    bottleneck_dim: int = 256
    out_dim: int = 100
    mlp_dims: tuple = (features_dim, 2048, 2048, bottleneck_dim)

    # loss
    weight_lab: float = 0.35
    
    # loss/classification
    tau_s: float = 0.1
    tau_t: float = 0.04
    warmup_tau_t: float = 0.07
    warmup_epochs: int = 30
    epsilon: float = 4.0
    
    # loss/representation
    tau_u: float = 1.0
    tau_c: float = 0.07

    # train
    seed: int = 42
    num_epochs: int = 200
    stop_after_epochs: int = None
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp32" if device != "cuda" else ("bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16") 
    num_workers: int = 4

    # logs
    path_log: Path = _ROOT / "runs"
    print_freq: int = 50
    eval_freq: int = 5

PROFILES = {
    "none": {},
    "local":
    {
        "batch_size_train": 8,
        "batch_size_eval": 8,
        "num_workers": 0,
        "max_train_samples": 64,
        "max_eval_samples": 64,
        "stop_after_epochs": 3,
        "print_freq": 1
    },
    "kaggle_smoke": {
        "path_dataset": Path("/kaggle/temp/datasets"),
        "path_log": Path("/kaggle/working/runs"),
        "num_workers": os.cpu_count(),
        "max_train_samples": 2560,
        "max_eval_samples": 1000,
        "stop_after_epochs": 2,
        "print_freq": 5,
        "eval_freq": 1,
    },
    "kaggle": {
        "path_dataset": Path("/kaggle/temp/datasets"),
        "path_log": Path("/kaggle/working/runs"),
        "num_workers": os.cpu_count(),
        "stop_after_epochs": 20,
        "print_freq": 50,
        "eval_freq": 5,
    },
    "vastai": {
        "path_dataset": Path("/workspace/datasets"),
        "path_log": Path("/workspace/runs"),
        "num_workers": 12,
        "print_freq": 50,
        "eval_freq": 5,
    }
}

def load_config(profile="none"):
    config = dataclasses.replace(Config(), **PROFILES[profile])
    return config
