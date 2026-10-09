import sys
import os
import subprocess
import json
import logging
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

class Averaging(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.weighted_sum = 0.0
        self.sum_weights = 0.0

    def update(self, val, weight):
        self.weighted_sum += val*weight
        self.sum_weights += weight

    def avg(self):
        return self.weighted_sum/self.sum_weights

class Logger:
    def __init__(self, path_log, use_wandb):
        self.path_log = path_log
        self.path_log.mkdir(parents=True, exist_ok=True)

        self._log = logging.getLogger("simgcd")
        self._log.setLevel(logging.INFO)
        self._log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout),
                  logging.FileHandler(self.path_log / "train.log", mode="a")):
            h.setFormatter(fmt)
            self._log.addHandler(h)
        self._log.propagate = False
        
        self._json = open(self.path_log / "metrics.jsonl", "a")

    def info(self, msg):
        self._log.info(msg)

    def metrics(self, name, global_step, values):
        record = {
            "name": name,
            "global_step": global_step,
            "time": datetime.now().isoformat(),
            "values": values
        }

        self._json.write(json.dumps(record) + "\n")
        self._json.flush()

        summary = " ".join(f"{k}={v}" for k, v in values.items())
        self.info(f"Metrics [{name}] step={global_step} {summary}")

    def close(self):
        self._json.close()

def git_info():
    repo = Path(__file__).resolve().parent
    try:
        run = lambda *args:subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain"))}
    except:
        return {"commit": "unknow", "dirty": None}

def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

def save_checkpoint(run_dir, model, optimizer, scheduler, scaler, epoch, global_step):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {
        "model-trainable": {k: v for k, v in model.state_dict().items() if k in trainable},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "rng": (np.random.get_state(), torch.get_rng_state())
    }
    tmp = run_dir / "ckpt.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, run_dir / "ckpt.pt")

def load_checkpoint(run_dir, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model-trainable"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    rng_np, rng_torch = ckpt["rng"]
    np.random.set_state(rng_np)
    torch.set_rng_state(rng_torch)

    return ckpt
