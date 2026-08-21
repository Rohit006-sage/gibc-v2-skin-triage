"""Model definitions for HAM10000 binary classification.

Default architecture: EfficientNet-B0 pretrained on ImageNet, with the
classifier head replaced by a single-logit binary head. Backbone is
fine-tunable; the default config freezes it for the v1 baseline and
unfreezes it for the v2 (week 3) iteration.

Public API:
    build_model(arch: str = "efficientnet_b0", pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required. Install with: pip install timm>=0.9"
    ) from e


SUPPORTED_ARCHS = ("efficientnet_b0", "resnet50", "convnext_tiny")


def build_model(
    arch: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a binary classifier head on top of a timm backbone.

    Args:
        arch: one of SUPPORTED_ARCHS.
        pretrained: load ImageNet weights if True.
        freeze_backbone: if True, freeze all parameters except the new head.
            Useful for the v1 baseline; turn off for v2 fine-tuning.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}, got {arch!r}")

    model = timm.create_model(arch, pretrained=pretrained, num_classes=1)

    if freeze_backbone:
        head_param_ids = {id(p) for p in model.get_classifier().parameters()}
        for name, p in model.named_parameters():
            if id(p) not in head_param_ids:
                p.requires_grad = False

    return model


if __name__ == "__main__":
    m = build_model(arch="efficientnet_b0", pretrained=False, freeze_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 1), f"Expected (2, 1), got {tuple(y.shape)}"
    print(f"[OK] model output shape: {tuple(y.shape)}")

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"[OK] trainable params: {trainable:,} / {total:,}")
