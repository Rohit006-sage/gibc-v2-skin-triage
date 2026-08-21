"""
Kaggle Notebook setup script.

Run this once at the top of a Kaggle Notebook. It:
  1. Verifies GPU is available.
  2. Clones your GitHub repo into /kaggle/working/ (so you can pull updates
     from git instead of re-uploading the dataset each session).
  3. Downloads HAM10000 from Kaggle's own dataset mirror into
     /kaggle/working/project/data/raw/ (no need to upload it yourself).
  4. Sanity-checks the data with `python -m src.data`.

After this cell runs, you can train with:
    !cd /kaggle/working/project && python -m src.train \
        --metadata data/raw/HAM10000_metadata.csv \
        --image-dir data/raw/HAM10000_images_part_1 \
        --image-dir data/raw/HAM10000_images_part_2 \
        --output-dir runs/baseline \
        --arch efficientnet_b0 \
        --epochs 10 \
        --batch-size 32 \
        --lr 1e-4

Drop this entire script into a single Kaggle cell. Edit the constants
below before running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/Rohit006-sage/-gibc-v2-skin-triage.git"
KAGGLE_DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"

WORK = Path("/kaggle/working")
PROJECT_DIR = WORK / "project"
RAW_DIR = PROJECT_DIR / "data" / "raw"


def run(cmd: str, **kwargs) -> None:
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def check_gpu() -> None:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU detected. Training will be slow.")


def install_dependencies() -> None:
    pip_packages = [
        "timm>=0.9",
        "scikit-learn>=1.3",
        "streamlit>=1.30",
        "matplotlib>=3.7",
    ]
    run(f"{sys.executable} -m pip install --quiet " + " ".join(pip_packages))


def clone_repo() -> None:
    if PROJECT_DIR.exists():
        print(f"Project already present at {PROJECT_DIR}, skipping clone.")
        return
    run(f"git clone {GITHUB_REPO_URL} {PROJECT_DIR}")


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "HAM10000_metadata.csv").exists():
        print("Dataset already present, skipping download.")
        return

    print(f"Downloading HAM10000 from Kaggle dataset {KAGGLE_DATASET_SLUG}...")
    run(f"kaggle datasets download -d {KAGGLE_DATASET_SLUG} -p {RAW_DIR} --unzip")
    print("Download complete.")


def verify_data() -> None:
    expected = [
        RAW_DIR / "HAM10000_metadata.csv",
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2",
    ]
    for p in expected:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")
        print(f"[OK] {p}")

    run(f"cd {PROJECT_DIR} && python -m src.data")


if __name__ == "__main__":
    print("=" * 60)
    print("Kaggle Notebook setup for GIBC V2 skin lesion triage")
    print("=" * 60)

    check_gpu()
    install_dependencies()
    clone_repo()
    download_dataset()
    verify_data()

    print("\n" + "=" * 60)
    print("Setup complete. Next: run the training command shown in the docstring.")
    print("=" * 60)
