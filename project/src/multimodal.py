"""Multi-modal model: dermoscopy image (EfficientNet-B0 features) fused with
patient tabular metadata (age, sex, anatomical site) from HAM10000.

The tabular head is a small MLP; its output is concatenated with the CNN
feature vector and passed to a single-logit head. Training mirrors
src/train.py (patient-stratified split, pos_weight BCE, cosine schedule).

Example:
    python -m src.multimodal \
        --metadata data/raw/HAM10000_metadata.csv \
        --image-dir data/raw/HAM10000_images_part_1 \
        --image-dir data/raw/HAM10000_images_part_2 \
        --output-dir runs/v2_multimodal \
        --epochs 16 --batch-size 32 --lr 1e-4 --use-randaug
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from src.data import get_splits
from src.train import evaluate, sensitivity_at_specificity
from src.transforms import eval_transform, train_transform

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required. Install with: pip install timm>=0.9") from e


SEX_CATEGORIES = ["male", "female", "unknown"]
SITE_CATEGORIES = [
    "scalp", "face", "ear", "neck", "chest", "back", "abdomen",
    "arm", "hand", "leg", "foot", "acral", "unknown",
]


class TabularEncoder:
    """Fits normalization stats on the train split; applies to any split."""

    def __init__(self) -> None:
        self.age_mean = 0.0
        self.age_std = 1.0
        self.sex_index = {s: i for i, s in enumerate(SEX_CATEGORIES)}
        self.site_index = {s: i for i, s in enumerate(SITE_CATEGORIES)}

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        ages = pd.to_numeric(df["age"], errors="coerce")
        self.age_mean = float(ages.mean())
        self.age_std = float(ages.std()) or 1.0
        return self

    @property
    def dim(self) -> int:
        return 1 + len(SEX_CATEGORIES) + len(SITE_CATEGORIES)

    def encode(self, row: pd.Series) -> np.ndarray:
        age = pd.to_numeric(row.get("age"), errors="coerce")
        age = self.age_mean if pd.isna(age) else age
        age_norm = (age - self.age_mean) / self.age_std

        sex = str(row.get("sex", "unknown")).lower()
        sex_vec = np.zeros(len(SEX_CATEGORIES))
        if sex not in self.sex_index:
            sex = "unknown"
        sex_vec[self.sex_index[sex]] = 1.0

        site = str(row.get("localization", "unknown")).lower()
        site_vec = np.zeros(len(SITE_CATEGORIES))
        if site not in self.site_index:
            site = "unknown"
        site_vec[self.site_index[site]] = 1.0

        return np.concatenate([[age_norm], sex_vec, site_vec]).astype(np.float32)

    def to_dict(self) -> dict:
        return {"age_mean": self.age_mean, "age_std": self.age_std,
                "sex_index": self.sex_index, "site_index": self.site_index}

    @classmethod
    def from_dict(cls, d: dict) -> "TabularEncoder":
        enc = cls()
        enc.age_mean = d["age_mean"]
        enc.age_std = d["age_std"]
        enc.sex_index = d["sex_index"]
        enc.site_index = d["site_index"]
        return enc


class HAMTabDataset(Dataset):
    """Returns (image, tabular_vector, label, image_id)."""

    def __init__(self, df: pd.DataFrame, image_dirs: list[Path],
                 encoder: TabularEncoder, transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dirs = [Path(d) for d in image_dirs]
        self.encoder = encoder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = int(row["target"])

        path = None
        for d in self.image_dirs:
            cand = d / f"{image_id}.jpg"
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(f"Image {image_id}.jpg not found in any image dir")

        from PIL import Image
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        tab = torch.from_numpy(self.encoder.encode(row))
        return image, tab, label, image_id


class MultiModalModel(nn.Module):
    """CNN backbone (feature extractor) + tabular MLP + fused binary head."""

    def __init__(self, arch: str = "efficientnet_b0", tabular_dim: int = 16,
                 hidden: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        cnn_dim = self.backbone.num_features
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.head = nn.Linear(cnn_dim + hidden, 1)

    def forward(self, images: torch.Tensor, tabs: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(images)
        tab_feat = self.tab_mlp(tabs)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(fused)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--metadata", required=True, type=str)
    p.add_argument("--image-dir", action="append", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--checkpoint", default="", type=str,
                   help="Checkpoint for --mode eval (defaults to <output-dir>/best.pt).")
    p.add_argument("--arch", default="efficientnet_b0", type=str)
    p.add_argument("--epochs", default=16, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--min-lr", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=1e-5, type=float)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--use-randaug", action="store_true")
    p.add_argument("--label-smoothing", default=0.0, type=float)
    p.add_argument("--grad-clip", default=1.0, type=float)
    p.add_argument("--hidden", default=64, type=int,
                   help="Hidden width of the tabular MLP.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_multimodal(model, loader, criterion, device):
    """Mirror of src.train.evaluate for the multimodal data loader
    (4-tuple batches instead of 3-tuple)."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_scores, all_labels = [], []
    for imgs, tabs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        tabs = tabs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(imgs, tabs).squeeze(1)
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


def run_eval(args: argparse.Namespace) -> None:
    """Test-set evaluation for a trained multimodal checkpoint."""
    import json as _json

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = [Path(p) for p in args.image_dir]

    _, _, test_df = get_splits(args.metadata, seed=42)
    if not {"age", "sex", "localization"}.issubset(test_df.columns):
        raise SystemExit("Metadata is missing tabular columns (age/sex/localization).")
    print(f"Test set: {len(test_df)} images")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    encoder = TabularEncoder.from_dict(ckpt["tabular"])
    model = MultiModalModel(arch=ckpt.get("arch", args.arch),
                            tabular_dim=encoder.dim,
                            hidden=ckpt.get("hidden", args.hidden),
                            pretrained=False).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = HAMTabDataset(test_df, image_dirs, encoder, transform=eval_transform())
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    all_scores, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for imgs, tabs, labels, image_ids in test_loader:
            imgs = imgs.to(args.device, non_blocking=True)
            tabs = tabs.to(args.device, non_blocking=True)
            logits = model(imgs, tabs).squeeze(1)
            all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(image_ids)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    from sklearn.metrics import classification_report, confusion_matrix, \
        precision_recall_fscore_support, roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    sens = sensitivity_at_specificity(labels, scores, target_spec=0.95)
    preds = (scores >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics = {
        "test_auc": auc,
        "test_sensitivity_at_95_specificity": sens,
        "threshold": 0.5,
        "per_class": {
            "benign":   {"precision": float(precision[0]), "recall": float(recall[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
            "malignant": {"precision": float(precision[1]), "recall": float(recall[1]),
                          "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(labels)),
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        _json.dump(metrics, f, indent=2)
    np.savez(output_dir / "test_predictions.npz",
             image_ids=np.array(all_ids), scores=scores, labels=labels, preds=preds)

    print("\n=== Multimodal test-set metrics (threshold 0.5) ===")
    print(f"AUROC:                         {auc:.4f}")
    print(f"Sensitivity @ 95% specificity: {sens:.4f}")
    print(classification_report(labels, preds, target_names=["benign", "malignant"], digits=4))
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
    print(f"Saved: {output_dir / 'test_metrics.json'}")


def main() -> None:
    args = parse_args()
    if args.mode == "eval":
        run_eval(args)
        return
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = [Path(p) for p in args.image_dir]

    train_df, val_df, _ = get_splits(args.metadata, seed=args.seed)
    if not {"age", "sex", "localization"}.issubset(train_df.columns):
        raise SystemExit(
            "Metadata is missing tabular columns. HAM10000_metadata.csv must "
            "contain 'age', 'sex' and 'localization' for the multimodal model."
        )
    print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")

    encoder = TabularEncoder().fit(train_df)
    tabular_dim = encoder.dim
    print(f"Tabular features: {tabular_dim} dims "
          f"(age z-score + {len(SEX_CATEGORIES)} sex + {len(SITE_CATEGORIES)} site)")

    pos_weight = torch.tensor([
        (len(train_df) - train_df["target"].sum()) / max(train_df["target"].sum(), 1)
    ], dtype=torch.float32, device=args.device)

    train_ds = HAMTabDataset(train_df, image_dirs, encoder,
                             transform=train_transform(rand_aug=args.use_randaug))
    val_ds = HAMTabDataset(val_df, image_dirs, encoder, transform=eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    model = MultiModalModel(arch=args.arch, tabular_dim=tabular_dim,
                            hidden=args.hidden, pretrained=True).to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    best_auc = -1.0
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_sens_at_95spec"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for imgs, tabs, labels, _ in train_loader:
            imgs = imgs.to(args.device, non_blocking=True)
            tabs = tabs.to(args.device, non_blocking=True)
            labels = labels.float().to(args.device, non_blocking=True)
            if args.label_smoothing > 0:
                labels = labels * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            optimizer.zero_grad()
            logits = model(imgs, tabs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = total_loss / max(n, 1)
        scheduler.step()

        val_loss, val_auc, val_sens, _, _ = evaluate_multimodal(
            model, val_loader, criterion, args.device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_sens@95%spec={val_sens:.4f}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc, val_sens])

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "hidden": args.hidden,
                "val_auc": val_auc,
                "tabular": encoder.to_dict(),
            }, output_dir / "best.pt")

    with open(output_dir / "config.json", "w") as f:
        json.dump({"arch": args.arch, "hidden": args.hidden,
                   "tabular_dim": tabular_dim, "encoder": encoder.to_dict()}, f, indent=2)

    print(f"\nBest val AUROC: {best_auc:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()