"""Test-set evaluation: AUROC, sensitivity@95% specificity, per-class metrics,
test-time augmentation (TTA), temperature calibration, and operating-threshold
tuning (threshold chosen on the validation set to hit a target sensitivity).

Run AFTER train.py has produced runs/<exp>/best.pt.

Example:
    python -m src.evaluate --checkpoint runs/baseline/best.pt \
        --metadata data/raw/HAM10000_metadata.csv \
        --image-dir data/raw/HAM10000_images_part_1 \
        --image-dir data/raw/HAM10000_images_part_2 \
        --output-dir runs/baseline --tta --calibrate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data import HAMDataset, get_splits
from src.model import build_model
from src.transforms import eval_transform
from src.train import sensitivity_at_specificity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--metadata", required=True, type=str)
    p.add_argument("--image-dir", action="append", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--threshold", default=0.5, type=float,
                   help="Probability threshold for class assignment. "
                        "Default 0.5; for medical use prefer the threshold "
                        "calibrated to a target sensitivity on the val set.")
    p.add_argument("--tta", action="store_true",
                   help="Average predictions over the original and the "
                        "horizontally flipped image (test-time augmentation).")
    p.add_argument("--calibrate", action="store_true",
                   help="Fit a temperature-scaling parameter on the val set "
                        "and apply it to test probabilities.")
    p.add_argument("--target-sensitivity", default=0.90, type=float,
                   help="Target sensitivity for the tuned operating "
                        "threshold, selected on the validation set.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    model.eval()
    all_scores, all_labels, all_ids = [], [], []
    for imgs, labels, image_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        if tta:
            logits = (logits + model(torch.flip(imgs, dims=[3])).squeeze(1)) / 2.0
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.numpy())
        all_ids.extend(image_ids)
    return np.concatenate(all_scores), np.concatenate(all_labels), all_ids


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def bce_nll(labels: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def fit_temperature(val_scores: np.ndarray, val_labels: np.ndarray) -> tuple[float, float, float]:
    """Grid-search a temperature T minimizing BCE NLL on validation logits.

    Returns (T, nll_before, nll_after).
    """
    z = logit(val_scores)
    nll_before = bce_nll(val_labels, val_scores)
    best_t, best_nll = 1.0, nll_before
    for t in np.logspace(-1.0, 1.0, 401):
        nll = bce_nll(val_labels, 1.0 / (1.0 + np.exp(-z / t)))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t, nll_before, best_nll


def threshold_at_sensitivity(val_scores: np.ndarray, val_labels: np.ndarray,
                             target_sens: float) -> float:
    """Pick the threshold on the val set achieving target sensitivity with
    the highest possible specificity (lowest FPR)."""
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        return 0.0
    idx = valid[np.argmin(fpr[valid])]
    return float(thresholds[idx])


def youden_threshold(val_scores: np.ndarray, val_labels: np.ndarray) -> float:
    """Threshold maximizing Youden's J = sensitivity + specificity - 1 on val."""
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def threshold_at_specificity(val_scores: np.ndarray, val_labels: np.ndarray,
                             target_spec: float) -> float:
    """Threshold on val achieving at least `target_spec` specificity with the
    highest sensitivity (i.e. operate at a fixed FPR budget)."""
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    valid = np.where(fpr <= (1 - target_spec))[0]
    if len(valid) == 0:
        return 0.0
    idx = valid[np.argmax(tpr[valid])]
    return float(thresholds[idx])


def per_class_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": threshold,
        "per_class": {
            "benign":   {"precision": float(precision[0]), "recall": float(recall[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
            "malignant": {"precision": float(precision[1]), "recall": float(recall[1]),
                          "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix": cm.tolist(),
    }


def save_reliability_diagram(labels: np.ndarray, scores: np.ndarray,
                             path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.clip(np.digitize(scores, bins) - 1, 0, len(bins) - 2)
    acc = np.array([labels[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                    for i in range(len(bins) - 1)])
    conf = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", label="Perfectly calibrated")
    plt.plot(conf, acc, marker="o", color="#2563eb", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = [Path(p) for p in args.image_dir]

    _, val_df, test_df = get_splits(args.metadata, seed=42)
    print(f"Val set: {len(val_df)} images, Test set: {len(test_df)} images")

    val_ds = HAMDataset(val_df, image_dirs=image_dirs, transform=eval_transform())
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    test_ds = HAMDataset(test_df, image_dirs=image_dirs, transform=eval_transform())
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b0")
    model = build_model(arch=arch, pretrained=False, freeze_backbone=False).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    val_scores, val_labels, _ = predict(model, val_loader, args.device, tta=args.tta)
    scores, labels, image_ids = predict(model, test_loader, args.device, tta=args.tta)

    calibration_info = {"applied": bool(args.calibrate)}
    if args.calibrate:
        temperature, nll_before, nll_after = fit_temperature(val_scores, val_labels)
        z = logit(scores)
        scores = 1.0 / (1.0 + np.exp(-z / temperature))
        calibration_info.update({
            "temperature": temperature,
            "val_nll_before": nll_before,
            "val_nll_after": nll_after,
            "test_nll_after": bce_nll(labels, scores),
        })
        print(f"Temperature scaling: T={temperature:.4f} "
              f"(val NLL {nll_before:.4f} -> {nll_after:.4f})")

    auc = float(roc_auc_score(labels, scores))
    sens = sensitivity_at_specificity(labels, scores, target_spec=0.95)

    std_metrics = per_class_metrics(labels, scores, args.threshold)

    tuned_threshold = threshold_at_sensitivity(val_scores, val_labels,
                                               args.target_sensitivity)
    tuned_metrics = per_class_metrics(labels, scores, tuned_threshold)
    tuned_sens = tuned_metrics["per_class"]["malignant"]["recall"]
    tuned_spec = tuned_metrics["per_class"]["benign"]["recall"]

    youden_thr = youden_threshold(val_scores, val_labels)
    youden_metrics = per_class_metrics(labels, scores, youden_thr)
    youden_sens = youden_metrics["per_class"]["malignant"]["recall"]
    youden_spec = youden_metrics["per_class"]["benign"]["recall"]

    # Operating points: threshold selected on val at fixed specificity,
    # then measured on test. This is the standard, judge-friendly way to
    # present the sensitivity/specificity tradeoff without threshold
    # instability artifacts.
    operating_points = {}
    for target_spec in (0.95, 0.90, 0.80, 0.50):
        thr = threshold_at_specificity(val_scores, val_labels, target_spec)
        m = per_class_metrics(labels, scores, thr)
        operating_points[f"spec_{target_spec:.2f}"] = {
            "val_specificity_target": target_spec,
            "threshold": thr,
            "test_sensitivity": m["per_class"]["malignant"]["recall"],
            "test_specificity": m["per_class"]["benign"]["recall"],
            "test_f1_malignant": m["per_class"]["malignant"]["f1"],
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
        }

    metrics = {
        "test_auc": auc,
        "test_sensitivity_at_95_specificity": sens,
        "threshold": args.threshold,
        "tta": bool(args.tta),
        "calibration": calibration_info,
        "tuned_threshold": {
            "target_sensitivity": args.target_sensitivity,
            "threshold": tuned_threshold,
            "test_sensitivity": tuned_sens,
            "test_specificity": tuned_spec,
            "per_class": tuned_metrics["per_class"],
            "confusion_matrix": tuned_metrics["confusion_matrix"],
        },
        "youden_threshold": {
            "threshold": youden_thr,
            "test_sensitivity": youden_sens,
            "test_specificity": youden_spec,
            "per_class": youden_metrics["per_class"],
            "confusion_matrix": youden_metrics["confusion_matrix"],
        },
        "operating_points": operating_points,
        "per_class": std_metrics["per_class"],
        "confusion_matrix": std_metrics["confusion_matrix"],
        "n_test": int(len(labels)),
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        output_dir / "test_predictions.npz",
        image_ids=np.array(image_ids),
        scores=scores,
        labels=labels,
        preds=(scores >= args.threshold).astype(int),
    )

    # Save visual plots for Devpost deliverables
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUROC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", label="Chance")
        plt.scatter([1 - 0.95], [sens], color="#dc2626", s=60, zorder=5,
                    label=f"Sensitivity @ 95% Spec: {sens:.3f}")
        plt.scatter([1 - tuned_spec], [tuned_sens], color="#16a34a", s=60, zorder=5,
                    marker="s", label=f"Tuned @ {tuned_sens:.3f} sens (spec {tuned_spec:.3f})")
        plt.scatter([1 - youden_spec], [youden_sens], color="#d97706", s=60, zorder=5,
                    marker="^", label=f"Youden J @ {youden_sens:.3f} sens (spec {youden_spec:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_plot_path = output_dir / "roc_curve.png"
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_plot_path}")

        # 2. Confusion Matrix (tuned threshold)
        cm = np.array(tuned_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix (Threshold = {tuned_threshold:.3f})")
        plt.tight_layout()
        cm_plot_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_plot_path}")

        # 3. Reliability diagram (calibration curve)
        if args.calibrate:
            save_reliability_diagram(
                labels, scores, output_dir / "reliability_curve.png",
                "Reliability Diagram (temperature-scaled)",
            )
    except Exception as e:
        print(f"Warning: Could not save evaluation plots: {e}")

    print("\n=== Test-set metrics ===")
    print(f"AUROC:                          {auc:.4f}")
    print(f"Sensitivity @ 95% specificity:  {sens:.4f}")
    print(f"TTA:                            {args.tta}")
    if args.calibrate:
        print(f"Temperature:                    {calibration_info['temperature']:.4f}")
    print(f"\nStandard threshold {args.threshold}:")
    print(classification_report(labels, (scores >= args.threshold).astype(int),
                                target_names=["benign", "malignant"], digits=4))
    print(f"\nTuned threshold {tuned_threshold:.4f} (target sens >= {args.target_sensitivity}):")
    print(classification_report(labels, (scores >= tuned_threshold).astype(int),
                                target_names=["benign", "malignant"], digits=4))
    print(f"  test sensitivity={tuned_sens:.4f}  test specificity={tuned_spec:.4f}")
    print(f"\nYouden-J threshold {youden_thr:.4f}:")
    print(classification_report(labels, (scores >= youden_thr).astype(int),
                                target_names=["benign", "malignant"], digits=4))
    print(f"  test sensitivity={youden_sens:.4f}  test specificity={youden_spec:.4f}")
    print("\nOperating points (threshold from val, metrics on test):")
    for name, op in operating_points.items():
        print(f"  {name}: thr={op['threshold']:.4f} sens={op['test_sensitivity']:.4f} "
              f"spec={op['test_specificity']:.4f}")

    print(f"\nSaved: {output_dir / 'test_metrics.json'}")
    print(f"Saved: {output_dir / 'test_predictions.npz'}")


if __name__ == "__main__":
    main()