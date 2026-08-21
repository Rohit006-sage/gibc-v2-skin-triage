"""Grad-CAM heatmap generator for an already-trained HAM10000 classifier.

Generates overlay PNGs for an arbitrary list of image_ids showing where the
model attended when making its prediction. Used in week 4 to produce
interpretability screenshots and in the error-analysis pass (week 7).

The target layer is the last convolutional / feature block of the chosen
backbone. For EfficientNet-B0, timm exposes this as model.conv_head; for
ResNet50 it's model.layer4; for ConvNeXt-Tiny it's model.stages[-1].

Example:
    python -m src.gradcam --checkpoint runs/baseline/best.pt \
        --metadata data/raw/HAM10000_metadata.csv \
        --image-ids ISIC_0024313 ISIC_0024314 \
        --image-dir data/raw/HAM10000_images_part_1 \
        --image-dir data/raw/HAM10000_images_part_2 \
        --output-dir outputs/gradcam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import HAMDataset
from src.model import build_model, SUPPORTED_ARCHS
from src.transforms import eval_transform


def _target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    if arch == "efficientnet_b0":
        return model.conv_head
    if arch == "resnet50":
        return model.layer4
    if arch == "convnext_tiny":
        return model.stages[-1]
    raise ValueError(f"Unknown target layer for arch {arch!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--metadata", required=True, type=str)
    p.add_argument("--image-dir", action="append", required=True, type=str)
    p.add_argument("--image-ids", nargs="+", default=[],
                   help="Specific ISIC image_ids to visualize. If empty, "
                        "picks one example per class from the metadata.")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--arch", default="efficientnet_b0", type=str)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return p.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor).squeeze(1)
        score = logits.sum()
        self.model.zero_grad()
        score.backward(retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1).cpu().numpy()[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(torch.sigmoid(logits).detach().cpu().numpy()[0])


def overlay(cam: np.ndarray, original: Image.Image, alpha: float = 0.4) -> Image.Image:
    """Return a PIL image with the CAM heatmap overlaid on the original."""
    h, w = original.size[1], original.size[0]
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0

    red = np.zeros_like(cam_arr)
    green = np.zeros_like(cam_arr)
    blue = np.zeros_like(cam_arr)
    red = cam_arr
    blue = 1 - cam_arr
    green = np.clip(2 * cam_arr - 0.5, 0, 1) * (1 - np.abs(cam_arr - 0.5) * 2)
    heat_rgb = np.stack([red, green, blue], axis=-1)
    heat_rgb = np.uint8(heat_rgb * 255)

    base = np.asarray(original.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat_rgb
    return Image.fromarray(np.uint8(blended))


def main() -> None:
    args = parse_args()
    if args.arch not in SUPPORTED_ARCHS:
        raise ValueError(f"--arch must be one of {SUPPORTED_ARCHS}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = [Path(p) for p in args.image_dir]

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch = ckpt.get("arch", args.arch)
    model = build_model(arch=arch, pretrained=False, freeze_backbone=False).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    import pandas as pd
    meta = pd.read_csv(args.metadata)
    from src.data import map_dx_to_binary
    meta["target"] = meta["dx"].map(map_dx_to_binary).astype(int)

    if not args.image_ids:
        sample_pos = meta[meta.target == 1].sample(n=2, random_state=42)["image_id"].tolist()
        sample_neg = meta[meta.target == 0].sample(n=2, random_state=42)["image_id"].tolist()
        args.image_ids = sample_pos + sample_neg

    transform = eval_transform()
    target_layer = _target_layer(model, arch)
    cam_engine = GradCAM(model, target_layer)

    for image_id in args.image_ids:
        try:
            row = meta[meta.image_id == image_id].iloc[0]
        except IndexError:
            print(f"[skip] {image_id} not in metadata")
            continue

        path = None
        for d in image_dirs:
            cand = d / f"{image_id}.jpg"
            if cand.exists():
                path = cand
                break
        if path is None:
            print(f"[skip] {image_id} image file not found")
            continue

        original = Image.open(path).convert("RGB")
        input_tensor = transform(original).unsqueeze(0).to(args.device)
        cam, prob = cam_engine(input_tensor)
        blended = overlay(cam, original)

        out_path = output_dir / f"{image_id}_prob{int(round(prob*100)):02d}.png"
        blended.save(out_path)
        print(f"[ok] {out_path.name} (p_malignant={prob:.3f}, true={int(row.target)})")


if __name__ == "__main__":
    main()
