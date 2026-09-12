from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    # datasets
    path_dataset: Path = _ROOT / "datasets"
    download: bool = True

CONFIG = Config()
