import argparse
import dataclasses
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

from data.cifar import WrapperCIFAR, build_transforms, build_sampler, get_cifar100_datasets

from models.backbone import build_backbone, unfreeze_vit_from
from models.head import Head
from models.model import Model

from losses.classification import loss_classification, teacher_temp
from losses.representation import loss_representation

from evaluate import evaluate

from util.utils import Averaging, Logger, git_info, get_param_groups, save_checkpoint, load_checkpoint
from config import load_config, PROFILES

def train(model, train_dataloader, eval_dataloader, optimizer, scheduler, scaler, amp_dtype, use_amp, config, run_dir, first_epoch, global_step):
    last_epoch = min(config.num_epochs, config.stop_after_epochs or config.num_epochs)
    for epoch in range(first_epoch, last_epoch):
        logger.info(f"EPOCH {epoch}")

        train_loss = Averaging()
        tau_t = teacher_temp(epoch, config)

        torch.cuda.synchronize() if config.device == "cuda" else None
        win_start, data_wait, n_win = time.perf_counter(), 0.0, 0
        start_epoch = time.time()
        
        model.train()
        it = iter(train_dataloader)
        batch_i = 0
        while True:
            t0 = time.perf_counter()
            try: batch = next(it)
            except StopIteration: break
            data_wait += time.perf_counter() - t0

            views, labels, mask = batch
            
            views = [v.to(device, non_blocking=True) for v in views]
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            images = torch.cat(views, dim=0)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                proj, logits = model(images)
                
                cls = loss_classification(logits, labels, mask,
                                          config.tau_s, tau_t,
                                          config.epsilon)
                rep = loss_representation(proj, labels, mask,
                                          config.tau_u, config.tau_c)

                w = config.weight_lab
                loss = (
                    w*cls['L_cls_s'] + (1-w)*cls['L_cls_u'] +
                    w*rep['L_rep_s'] + (1-w)*rep['L_rep_u']
                )

                if not torch.isfinite(loss):
                    logger.info(f"Loss NaN/inf epoch {epoch} step {global_step}")
                    raise RuntimeError("Non finite loss")

            train_loss.update(loss.item(), labels.shape[0])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # logs every print_freq iter
            n_win += 1
            if n_win == config.print_freq:
                torch.cuda.synchronize() if config.device == "cuda" else None
                wall = time.perf_counter() - win_start
                t_data = data_wait/n_win
                t_compute = (wall - data_wait)/n_win
                win_start, data_wait, n_win = time.perf_counter(), 0.0, 0

                logger.metrics("iter", global_step, {
                    'loss':loss.item(),
                    'L_rep_s':rep['L_rep_s'].item(),
                    'L_rep_u':rep['L_rep_u'].item(),
                    'L_cls_s':cls['L_cls_s'].item(),
                    'L_cls_u':cls['L_cls_u'].item(),
                    'cls/entropy':cls['logs']['cls/entropy'].item(),
                    'cls/distill':cls['logs']['cls/distill'].item(),
                    'lr':scheduler.get_last_lr()[0],
                    'tau_t':tau_t,
                    't_data':t_data,
                    't_compute':t_compute,
                    'gpu_mem_gb':torch.cuda.max_memory_allocated()/1e9,
                })

            global_step += 1
            batch_i += 1

        # evaluation
        if epoch % config.eval_freq == 0 or epoch == last_epoch-1:
            all, old, new = evaluate(model, eval_dataloader, config.num_old_classes, device, amp_dtype, use_amp)
            logger.metrics("eval", global_step, {
                'all':all,
                'old':old,
                'new':new,
                'proto_norm':model.head.prototypes.norm(dim=1).mean().item()
            })

        # logs of epoch
        logger.metrics("epoch", global_step, {
            'train_loss': train_loss.avg(),
            'lr':scheduler.get_last_lr()[0],
            'total_time':time.time()-start_epoch
        })

        scheduler.step()

        # checkpoint
        logger.info(f"Saving checkpoint")
        save_checkpoint(run_dir, model, optimizer, scheduler, scaler, epoch+1, global_step)

if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES), default="none")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    CONFIG = load_config(args.profile)
    git = git_info()
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if args.resume:
        run_dir = Path(CONFIG.path_log / args.resume)
    else:
        run_dir = Path(CONFIG.path_log / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.profile}")    
        run_dir.mkdir(parents=True)

    logger = Logger(run_dir)
    logger.info(f"Profile:{args.profile}\nCommit:{git['commit']}\nDirty:{git['dirty']}\n" + "\n".join(f"{k}:{v}" for k, v in dataclasses.asdict(CONFIG).items()))

    device = torch.device(CONFIG.device)

    # dataset
    logger.info("Loading transforms and datasets")
    train_transform, test_transform = build_transforms(CONFIG.img_size_encoder, CONFIG.crop_pct)
    result = get_cifar100_datasets(train_transform, test_transform, CONFIG)
    
    wrapper_cifar = WrapperCIFAR(result['train_labelled'], result['train_unlabelled'])
   
    # dataloaders
    logger.info("Building dataloaders")
    sampler = build_sampler(wrapper_cifar)
    train_dataloader = DataLoader(
        dataset=wrapper_cifar,
        batch_size=CONFIG.batch_size_train,
        drop_last=True,
        sampler=sampler,
        num_workers=CONFIG.num_workers,
        pin_memory=(device.type == "cuda")
    )

    eval_dataloader = DataLoader(
        dataset=result['train_unlabelled_eval'],
        batch_size=CONFIG.batch_size_eval,
        drop_last=False,
        num_workers=CONFIG.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # model
    logger.info("Loading and freezing backbone")
    backbone = build_backbone(CONFIG.encoder_name)
    unfreeze_vit_from(backbone, CONFIG.unfreeze_from_num)
    logger.info("Loading head")
    head = Head(CONFIG.out_dim, CONFIG.mlp_dims)
    model = Model(backbone, head).to(device)
    
    # optimizer
    logger.info("Retrieving model parameters groups")
    groups = get_param_groups(model, CONFIG.weight_decay)
    
    logger.info("Building optimizer")
    optimizer = SGD(groups,
                    lr=CONFIG.lr,
                    momentum=CONFIG.momentum)

    # scheduler
    logger.info("Building scheduler")
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.num_epochs,
        eta_min=CONFIG.lr*1e-3
    )
    
    # scaler
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[CONFIG.precision]
    use_amp = CONFIG.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=(CONFIG.precision == "fp16"))

    # load checkpoint
    epoch, global_step = 0, 0
    if args.resume:
        logger.info(f"Loading checkpoint in {run_dir}")
        ckpt = load_checkpoint(run_dir, model, optimizer, scheduler, scaler, device)
        epoch, global_step = ckpt["epoch"], ckpt["global_step"]

    # train
    logger.info("Start training")
    train(model, train_dataloader, eval_dataloader, optimizer, scheduler, scaler, amp_dtype, use_amp, CONFIG, run_dir, epoch, global_step)
