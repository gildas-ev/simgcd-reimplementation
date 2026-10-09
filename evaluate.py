import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

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
