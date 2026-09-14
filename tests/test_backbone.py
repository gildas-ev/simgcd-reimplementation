from config import CONFIG
from models.backbone import build_backbone
from data.cifar import build_transforms, get_cifar100_datasets

import torch
from torch.utils.data import DataLoader

def test_encoder_contract():
    backbone = build_backbone()
    
    train_transform, test_transform = build_transforms(CONFIG.img_size_encoder, CONFIG.crop_pct)
    result = get_cifar100_datasets(train_transform, test_transform,             CONFIG.path_dataset,                                                    num_old_classes=CONFIG.num_old_classes,                                 prop_train_labels=CONFIG.prop_train_labels,                             n_views=CONFIG.n_views,                                                 download=CONFIG.download,                                               seed=CONFIG.seed)

    eval_dataloader = DataLoader(
        dataset=result['train_unlabelled_eval'],
        batch_size=CONFIG.batch_size_eval,
        drop_last=False
    )

    batch = next(iter(eval_dataloader))
    images = batch[0]

    features = backbone(images)

    assert features.shape == (CONFIG.batch_size_eval, CONFIG.features_dim)
    assert features.dtype == torch.float32
