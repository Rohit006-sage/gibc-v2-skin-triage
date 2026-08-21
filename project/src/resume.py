"""Resume training from runs/baseline/best.pt for additional epochs.

Loads the existing checkpoint, rebuilds the model, and runs the same train
loop as src.train but appending new rows to the existing training_log.csv
instead of overwriting it. Skips the seed-derived best-pt overwrite if the
resumed run does not beat the original.

Usage:
    python -m src.resume --epochs 9 --batch-size 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.data import HAMDataset, get_splits
from src.model import build_model
from src.train import evaluate, set_seed, train_one_epoch
from src.transforms import eval_transform, train_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/raw/HAM10000_metadata.csv")
    p.add_argument("--image-dir", action="append", required=True)
    p.add_argument("--output-dir", default="runs/baseline")
    p.add_argument("--checkpoint", default="runs/baseline/best.pt")
    p.add_argument("--epochs", type=int, default=9)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--min-lr", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=1e-5, type=float)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                   help="LR schedule over the resumed run (cosine decay to --min-lr).")
    p.add_argument("--grad-clip", default=0.0, type=float,
                   help="Max grad norm for clipping; 0 disables.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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

    train_ds = HAMDataset(train_df, image_dirs=image_dirs, transform=train_transform())
    val_ds = HAMDataset(val_df, image_dirs=image_dirs, transform=eval_transform())

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b0")
    prev_best_auc = ckpt.get("val_auc", -1.0)
    print(f"Loaded checkpoint arch={arch} prev_best_val_auc={prev_best_auc:.4f}")

    model = build_model(arch=arch, pretrained=False, freeze_backbone=False).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )

    log_path = output_dir / "training_log.csv"
    last_epoch = 0
    if log_path.exists():
        with open(log_path, "r") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            try:
                last_epoch = int(rows[-1][0])
            except ValueError:
                last_epoch = 0
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "val_auc", "val_sens_at_95spec"]
            )

    start_epoch = last_epoch + 1
    best_auc = prev_best_auc
    print(f"Resuming from epoch {start_epoch} for {args.epochs} more epochs")

    for offset in range(1, args.epochs + 1):
        epoch = start_epoch + offset - 1
        train_loss, _ = train_one_epoch(model, train_loader, criterion,
                                        optimizer, args.device,
                                        grad_clip=args.grad_clip)
        val_loss, val_auc, val_sens, _, _ = evaluate(model, val_loader,
                                                     criterion, args.device)
        print(
            f"Epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_sens@95%spec={val_sens:.4f}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc, val_sens])

        if scheduler is not None:
            scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "arch": arch,
                        "val_auc": val_auc},
                       output_dir / "best.pt")
            print(f"  -> new best, saved (val_auc={val_auc:.4f})")

    print(f"\nDone. Best val AUROC across run: {best_auc:.4f}")


if __name__ == "__main__":
    main()
