"""Render a clean architecture diagram for the Devpost submission.

Usage:
    python src/plot_architecture.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, color="#eff6ff", edge="#2563eb", fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                       linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#0f172a")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.6, color="#334155"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box(ax, 0.4, 5.6, 2.4, 0.9, "HAM10000 dataset\n10,015 dermoscopy images\n(7 classes -> benign/malignant)",
        color="#fef3c7", edge="#d97706")
    arrow(ax, 2.8, 6.05, 3.5, 6.05)

    box(ax, 3.5, 5.6, 2.4, 0.9, "Patient-level stratified\n70/15/15 split\n(no leakage)",
        color="#ecfdf5", edge="#059669")
    arrow(ax, 5.9, 6.05, 6.6, 6.05)

    box(ax, 6.6, 5.6, 2.5, 0.9, "EfficientNet-B0\n(ImageNet-pretrained,\nfine-tuned)",
        color="#eff6ff", edge="#2563eb")
    arrow(ax, 9.1, 6.05, 9.8, 6.05)

    box(ax, 9.8, 5.6, 1.9, 0.9, "BCE loss\npos_weight\n+ cosine LR",
        color="#eff6ff", edge="#2563eb")

    # refinement loop arrow back to model
    ax.add_patch(FancyArrowPatch((10.75, 5.6), (7.85, 5.6), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4,
                                 color="#94a3b8", connectionstyle="arc3,rad=0.55"))
    ax.text(9.3, 4.55, "resume / fine-tune\n(low LR, cosine)", ha="center",
            fontsize=8, color="#64748b")

    arrow(ax, 5.9, 5.6, 5.9, 4.4)
    box(ax, 3.5, 3.5, 2.4, 0.9, "Test-set evaluation\n(1,520 patient-safe cases)",
        color="#f0fdf4", edge="#16a34a")
    arrow(ax, 5.9, 3.5, 6.6, 3.5)

    box(ax, 6.6, 3.5, 2.5, 0.9, "TTA + temperature\ncalibration (T=3.72)\nAUROC 0.911",
        color="#f0fdf4", edge="#16a34a")
    arrow(ax, 9.1, 3.95, 9.8, 3.95)

    box(ax, 9.8, 3.5, 1.9, 0.9, "Grad-CAM\nheatmaps +\noperating points",
        color="#f0fdf4", edge="#16a34a")

    arrow(ax, 5.9, 3.5, 5.9, 2.4)
    box(ax, 3.5, 1.5, 2.4, 0.9, "Streamlit web app\nupload -> predict -> explain",
        color="#eef2ff", edge="#4f46e5")

    arrow(ax, 5.9, 1.95, 6.6, 1.95)
    box(ax, 6.6, 1.5, 2.5, 0.9, "Triage result\n(calibrated probability)\n+ Grad-CAM overlay",
        color="#eef2ff", edge="#4f46e5")

    # multimodal side-branch
    ax.add_patch(FancyArrowPatch((7.85, 5.6), (7.85, 5.15), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color="#94a3b8"))
    box(ax, 6.6, 4.2, 2.5, 0.9, "Multimodal ablation\nCNN + age/sex/site\n(test AUROC 0.905)",
        color="#faf5ff", edge="#9333ea", fs=8)

    ax.set_title("Skin Lesion Triage Tool — Architecture & Validation Pipeline",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    out = "outputs/eval/architecture.png"
    plt.savefig(out, dpi=300)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()