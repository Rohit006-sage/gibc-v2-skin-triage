"""Augmentation pipelines for train/val/test.

Train pipeline: resize, random horizontal flip, random rotation, color jitter,
ImageNet normalization. Val/test: resize + center-crop to the same size +
normalization, no random augmentation.

Image size is fixed at 224x224 (matches EfficientNet-B0 default input).
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def train_transform(rand_aug: bool = False) -> transforms.Compose:
    ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ]
    if rand_aug:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
