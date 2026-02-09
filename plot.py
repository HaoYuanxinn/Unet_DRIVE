# plot_training_summary.py
# 用法：
#   1) 指定文件：python plot_training_summary.py --results results20260208-153000.txt --out training_summary.png
#   2) 自动选最新：python plot_training_summary.py --results_dir . --out training_summary.png

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"\[epoch:\s*(\d+)\]")
LOSS_RE  = re.compile(r"train_loss:\s*([0-9]*\.?[0-9]+)")
LR_RE    = re.compile(r"lr:\s*([0-9]*\.?[0-9]+)")
DICE_RE  = re.compile(r"dice coefficient:\s*([0-9]*\.?[0-9]+)")

def find_latest_results(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("results*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No results*.txt found in: {results_dir}")
    return candidates[0]

def parse_results_file(fp: Path) -> pd.DataFrame:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    rows = []
    for b in blocks:
        m_epoch = EPOCH_RE.search(b)
        m_loss  = LOSS_RE.search(b)
        m_lr    = LR_RE.search(b)
        m_dice  = DICE_RE.search(b)
        if not (m_epoch and m_loss and m_lr and m_dice):
            continue
        rows.append({
            "epoch": int(m_epoch.group(1)),
            "train_loss": float(m_loss.group(1)),
            "lr": float(m_lr.group(1)),
            "dice": float(m_dice.group(1)),
        })

    if not rows:
        raise ValueError(
            f"Parsed 0 rows from {fp}. "
            f"Expected patterns like: [epoch: k], train_loss:, lr:, dice coefficient:"
        )

    df = (pd.DataFrame(rows)
            .sort_values("epoch")
            .drop_duplicates(subset=["epoch"], keep="last")
            .reset_index(drop=True))
    return df

def style_ax(ax, title, ylabel):
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(direction="in")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="", help="Path to results*.txt")
    parser.add_argument("--results_dir", type=str, default=".", help="Directory to search latest results*.txt")
    parser.add_argument("--out", type=str, default="training_summary.png", help="Output image path (png recommended)")
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi for png")
    args = parser.parse_args()

    if args.results:
        results_fp = Path(args.results)
        if not results_fp.exists():
            raise FileNotFoundError(f"--results not found: {results_fp}")
    else:
        results_fp = find_latest_results(Path(args.results_dir))

    df = parse_results_file(results_fp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.6), sharex=True)

    # 1) Loss
    axes[0].plot(df["epoch"], df["train_loss"], linewidth=1.7)
    style_ax(axes[0], "Training Loss", "Loss")

    # 2) Dice
    axes[1].plot(df["epoch"], df["dice"], linewidth=1.7)
    style_ax(axes[1], "Validation Dice", "Dice")
    axes[1].set_ylim(0, 1.0)

    # 3) LR（跨度大时用对数更清晰）
    axes[2].plot(df["epoch"], df["lr"], linewidth=1.7)
    style_ax(axes[2], "Learning Rate", "LR")
    axes[2].set_xlabel("Epoch")
    if df["lr"].min() > 0 and (df["lr"].max() / df["lr"].min() >= 50):
        axes[2].set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # 保存 PNG
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    main()
