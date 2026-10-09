import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config

REFS = {"all": (80.1, 0.9), "old": (81.2, 0.4), "new": (77.8, 2.0)}
COLORS = {"all": "black", "old": "tab:blue", "new": "tab:orange"}

def load_metrics(run_dir):
    with open(run_dir / "metrics.jsonl") as f:
        rows = [{"name": r["name"], "global_step": r["global_step"], **r["values"]}
                for r in map(json.loads, f)]
    df = pd.DataFrame(rows).drop_duplicates(["name", "global_step"], keep="last")
    iters_per_epoch = df.loc[df["name"] == "epoch", "global_step"].min()
    df["epoch"] = df["global_step"] / iters_per_epoch
    return {name: g.sort_values("epoch") for name, g in df.groupby("name")}

def plot_accuracy(ev, cfg, path):
    fig, (ax, ax_norm) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=[3, 1])
    for k, (ref, _) in REFS.items():
        ax.plot(ev["epoch"], 100 * ev[k], marker="o", ms=3, color=COLORS[k], label=k.capitalize())
        ax.axhline(ref, ls="--", lw=1, color=COLORS[k])
    ax.axvline(cfg.warmup_epochs, color="grey", ls=":", label="end of τ_t warmup")
    ax.set_ylabel("accuracy (%)")
    ax.legend(title="dashed = paper")

    ax_norm.plot(ev["epoch"], ev["proto_norm"], marker="o", ms=3, color="tab:green")
    ax_norm.axvline(cfg.warmup_epochs, color="grey", ls=":")
    ax_norm.set(xlabel="epoch", ylabel="mean prototype norm")

    for a in (ax, ax_norm):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_losses(it, cfg, path):
    h_max = math.log(cfg.out_dim)
    offset = (1 - cfg.weight_lab) * cfg.epsilon * h_max
    window = max(1, round(len(it) / it["epoch"].max()))

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, color in ((ax_h, "cls/entropy", "tab:purple"), (ax_l, "loss", "tab:red")):
        ax.plot(it["epoch"], it[col].rolling(window, min_periods=1).median(), color=color,
                label="rolling median (~1 epoch)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    ax_h.axhline(h_max, ls="--", color="grey", label=f"max = ln {cfg.out_dim} = {h_max:.3f}")
    ax_h.set(ylabel="entropy H of mean prediction (natural log)", title=f"H ∈ [0, ln {cfg.out_dim}]")
    ax_h.secondary_yaxis("right", functions=(np.exp, np.log)).set_ylabel("effective classes e^H")
    ax_h.legend()

    ax_l.axhline(-offset, ls="--", color="grey",
                 label=f"lower bound −(1−λ)·ε·ln K = {-offset:.2f}")
    ax_l.set(ylabel="total loss (my implementation)", title="total loss")
    ax_l.secondary_yaxis("right", functions=(lambda y: y + offset, lambda y: y - offset)) \
        .set_ylabel(f"official-code equivalent (+{offset:.2f})")
    ax_l.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(ev, it, ep, path):
    last = ev.iloc[-1]
    path.write_text(f"""# Summary

| | All | Old | New |
|---|---|---|---|
| this run, after {last['epoch']:.0f} epochs | {100 * last['all']:.1f} | {100 * last['old']:.1f} | {100 * last['new']:.1f} |
| paper (3 runs) | 80.1 ± 0.9 | 81.2 ± 0.4 | 77.8 ± 2.0 |

| | value |
|---|---|
| total time | {ep['total_time'].sum() / 3600:.2f} h |
| median t_data | {it['t_data'].median():.3f} s/iter |
| median t_compute | {it['t_compute'].median():.3f} s/iter |
| median GPU memory | {it['gpu_mem_gb'].median():.2f} GB |
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    m = load_metrics(run_dir)
    cfg = Config()

    plot_accuracy(m["eval"], cfg, out / "accuracy.png")
    plot_losses(m["iter"], cfg, out / "losses.png")
    write_summary(m["eval"], m["iter"], m["epoch"], out / "summary.md")
    print(f"Saved to {out}")
