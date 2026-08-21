"""Training loop for HAM10000 binary classifier.

Trains a timm backbone with a single-logit head using BCEWithLogitsLoss.
Tracks per-epoch AUROC and sensitivity-at-95%-specificity on the val set,
saves the best checkpoint by val AUROC, and writes a CSV log.

Example:
    python -m src.train --epochs 10 --batch-size 32 --lr 1e-4 \
        --metadata data/raw/HAM10000_metadata.csv \
        --image-dir data/raw/HAM10000_images_part_1 \
        --image-dir data/raw/HAM10000_images_part_2 \
        --output-dir runs/baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data import HAMDataset, get_splits
from src.model import build_model
from src.transforms import eval_transform, train_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, type=str)
    p.add_argument("--image-dir", action="append", required=True, type=str,
                   help="Repeat for each image directory (part_1, part_2).")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--arch", default="efficientnet_b0", type=str)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--min-lr", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=1e-5, type=float)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--use-randaug", action="store_true",
                   help="Use RandAugment instead of the mild ColorJitter pipeline.")
    p.add_argument("--label-smoothing", default=0.0, type=float,
                   help="BCE label smoothing in [0, 1) to reduce overconfidence.")
    p.add_argument("--grad-clip", default=0.0, type=float,
                   help="Max grad norm for clipping; 0 disables.")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                   help="LR schedule over the run (cosine decay to --min-lr).")
    p.add_argument("--early-stop", default=0, type=int,
                   help="Stop after N epochs without val AUROC improvement; 0 disables.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95) -> float:
    """Sensitivity (recall on positives) at a fixed specificity threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) == 0:
        return float("nan")
    idx = np.argmin(np.abs((1 - fpr) - target_spec))
    return float(tpr[idx])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(n, 1), 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_scores, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    sens = sensitivity_at_specificity(labels, scores, target_spec=0.95)
    return total_loss / max(n, 1), auc, sens, scores, labels


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = [Path(p) for p in args.image_dir]

    train_df, val_df, _ = get_splits(args.metadata, seed=args.seed)
    print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")

    pos_weight = torch.tensor([
        (len(train_df) - train_df["target"].sum()) / max(train_df["target"].sum(), 1)
    ], dtype=torch.float32, device=args.device)

    train_ds = HAMDataset(train_df, image_dirs=image_dirs, transform=train_transform(rand_aug=args.use_randaug))
    val_ds = HAMDataset(val_df, image_dirs=image_dirs, transform=eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    model = build_model(arch=args.arch, pretrained=True, freeze_backbone=args.freeze_backbone).to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )

    best_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_sens_at_95spec"])

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, args.device,
                                        grad_clip=args.grad_clip,
                                        label_smoothing=args.label_smoothing)
        val_loss, val_auc, val_sens, _, _ = evaluate(model, val_loader, criterion, args.device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_sens@95%spec={val_sens:.4f}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc, val_sens])

        if scheduler is not None:
            scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "val_auc": val_auc},
                       output_dir / "best.pt")
        else:
            stale_epochs += 1

        if args.early_stop > 0 and stale_epochs >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (no val AUROC gain for {args.early_stop} epochs)")
            break

    print(f"\nBest val AUROC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
