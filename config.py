from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    # data
    path_dataset = _ROOT / "datasets"
    download = True
    num_old_classes = 80
    prop_train_labels = 0.5
    seed = 0
    crop_pct = 0.875

    # batchs
    batch_size_train = 128
    batch_size_eval = 256
    n_views = 2

    # encoder
    img_size_encoder = 224


CONFIG = Config()
