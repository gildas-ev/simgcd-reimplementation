from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    # data
    path_dataset: Path = _ROOT / "datasets"
    download: bool = True
    num_old_classes: int = 80
    prop_train_labels: float = 0.5
    seed: int = 0
    crop_pct: float = 0.875

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

CONFIG = Config()
