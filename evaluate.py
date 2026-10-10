import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

def acc_metrics(y_pred, y_true, is_old):
    """Giving
    y_pred: np.array (N,) int64 the predictions of the model
    y_true: np.array (N,) int64 the ground truth
    is_old: np.array (N,) bool if the ground truth is an old class

    returns the all, old, new accuracies
    
    Note : returns nan if no instances"""
    assert len(y_pred) == len(y_true) and len(y_true) == len(is_old)

    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for pred, target in zip(y_pred, y_true):
        w[pred][target] += 1
    assert np.sum(w) == len(y_pred)

    row, col = linear_sum_assignment(w, maximize=True)
    cluster_to_class = col

    y_pred_classes = cluster_to_class[y_pred]
    correct = (y_pred_classes == y_true)

    nb_all, correct_all = len(y_pred), sum(correct)
    nb_old, correct_old = sum(is_old), sum(correct & is_old)
    nb_new, correct_new = sum(~is_old), sum(correct & ~is_old)
    
    return correct_all/nb_all, correct_old/nb_old, correct_new/nb_new

def evaluate(model, loader, num_old, device, amp_dtype, use_amp):
    """Runs the model with the eval loader and
    returns all, old, new accuracies"""
    try:
        model.eval()

        y_pred, y_true = [], []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for _, batch in enumerate(loader):
                images, labels = batch
            
                _, logits = model(images.to(device, non_blocking=True)) # shape (B, 100)
                y_pred.append(logits.argmax(1).cpu().numpy())
                y_true.append(labels.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        is_old = (y_true < num_old)

        all, old, new = acc_metrics(y_pred, y_true, is_old)
    finally:
        model.train()
    return all, old, new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    if args.checkpoint:
        # lazy import
        from pathlib import Path
        from data.cifar import build_transforms, get_cifar100_datasets
        from models.backbone import build_backbone, unfreeze_vit_from
        from models.head import Head
        from models.model import Model
        from config import load_config

        CONFIG = load_config()
        device = torch.device(CONFIG.device)

        # dataset
        train_transform, test_transform = build_transforms(CONFIG.img_size_encoder, CONFIG.crop_pct)
        result = get_cifar100_datasets(train_transform, test_transform, CONFIG)

        # dataloaders
        eval_dataloader = DataLoader(
            dataset=result['train_unlabelled_eval'],
            batch_size=CONFIG.batch_size_eval,
            drop_last=False,
            num_workers=CONFIG.num_workers,
            pin_memory=(device.type == "cuda")
        )

        # model
        backbone = build_backbone(CONFIG.encoder_name)
        unfreeze_vit_from(backbone, CONFIG.unfreeze_from_num)
        head = Head(CONFIG.out_dim, CONFIG.mlp_dims)
        model = Model(backbone, head).to(device)

        # checkpoint
        ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model-trainable"], strict=False)
        
        amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[CONFIG.precision]
        use_amp = CONFIG.precision != "fp32"

        # evaluation
        all, old, new = evaluate(model, eval_dataloader, CONFIG.num_old_classes, device, amp_dtype, use_amp)

        print(f"Evaluation of {args.checkpoint} : all={all:.3f}, old={old:.3f}, new={new:.3f}")
