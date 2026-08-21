"""Render training curves (val loss / AUROC / sensitivity) for Devpost.

Usage:
    python src/plot_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = [("runs/baseline", "Baseline (11 epochs)", "#2563eb"),
        ("runs/v3", "v3 fine-tune (10 epochs)", "#16a34a")]


def load(run: str) -> pd.DataFrame:
    path = Path(run) / "training_log.csv"
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df = pd.read_csv(path, header=None,
                         names=["epoch", "train_loss", "val_loss",
                                "val_auc", "val_sens_at_95spec"])
    return df


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for run, name, color in RUNS:
        df = load(run)
        axes[0].plot(df["epoch"], df["val_loss"], color=color, marker="o", ms=3,
                     label=f"{name} val loss")
        axes[1].plot(df["epoch"], df["val_auc"], color=color, marker="o", ms=3,
                     label=f"{name} val AUROC")
        axes[2].plot(df["epoch"], df["val_sens_at_95spec"], color=color, marker="o", ms=3,
                     label=f"{name} sens@95%spec")

    axes[0].set_title("Validation Loss")
    axes[1].set_title("Validation AUROC")
    axes[2].set_title("Sensitivity @ 95% specificity")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("BCE loss")
    axes[1].set_ylabel("AUROC")
    axes[2].set_ylabel("Sensitivity")
    plt.tight_layout()

    out = Path("outputs/eval/training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()