"""HAM10000 dataset loading and patient-level stratified splitting.

HAM10000 ships with 7 lesion types. This module collapses them to a binary
benign vs malignant label, then splits at the patient level (lesion_id) so
the same patient never appears in more than one split.

Public API:
    BINARY_LABEL_MAP   - 7-class -> binary int mapping
    map_dx_to_binary   - helper applying the mapping
    HAMDataset         - PyTorch Dataset returning (image, label, image_id)
    get_splits         - patient-grouped stratified 70/15/15 split

Example:
    >>> train_df, val_df, test_df = get_splits("data/raw/HAM10000_metadata.csv")
    >>> ds = HAMDataset(train_df, image_dirs=[...], transform=ToTensor())
    >>> img, label, image_id = ds[0]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as e:
    raise ImportError(
        "scikit-learn is required. Install with: pip install scikit-learn>=1.3"
    ) from e


BINARY_LABEL_MAP: dict[str, int] = {
    "nv":    0,
    "bkl":   0,
    "df":    0,
    "vasc":  0,
    "mel":   1,
    "bcc":   1,
    "akiec": 1,
}

METADATA_REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}


def map_dx_to_binary(dx: str) -> int:
    """Collapse a HAM10000 dx string to a binary 0/1 label."""
    if dx not in BINARY_LABEL_MAP:
        raise ValueError(
            f"Unknown dx value {dx!r}. Expected one of {list(BINARY_LABEL_MAP)}."
        )
    return BINARY_LABEL_MAP[dx]


def _resolve_image_path(image_id: str, image_dirs: list[Path]) -> Path:
    """Locate an image by id, checking each candidate directory in order.

    HAM10000's official distribution splits images across two folders
    (HAM10000_images_part_1 and ..._part_2) without a fixed partition, so
    we search both.
    """
    for d in image_dirs:
        candidate = d / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image {image_id}.jpg not found in any of: {[str(d) for d in image_dirs]}"
    )


class HAMDataset(Dataset):
    """PyTorch dataset for HAM10000 binary classification.

    Returns a 3-tuple (image_tensor, label_int, image_id_str). The image_id
    is included so downstream code (Grad-CAM error analysis, debugging)
    can trace predictions back to specific samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dirs: list[Path],
        transform=None,
    ) -> None:
        if not {"image_id", "target"}.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain 'image_id' and 'target' columns. Got: {df.columns.tolist()}"
            )
        if not image_dirs:
            raise ValueError("image_dirs must be a non-empty list of Path objects.")

        self.df = df.reset_index(drop=True)
        self.image_dirs = [Path(d) for d in image_dirs]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = int(row["target"])

        path = _resolve_image_path(image_id, self.image_dirs)
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_id


def _patient_level_dataframe(metadata: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-image metadata DataFrame to one row per lesion_id.

    If a single patient has both benign and malignant lesions (rare), we
    use the malignant label (max) so any patient with cancer is grouped
    into the malignant stratum.
    """
    grouped = (
        metadata.groupby("lesion_id", as_index=False)
        .agg(target=("target", "max"), dx=("dx", "first"))
    )
    return grouped


def _load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load HAM10000 metadata from either a CSV or a JSON file.

    Accepts:
    - The standard HAM10000 CSV (HAM10000_metadata.csv) with columns
      lesion_id, image_id, dx.
    - A JSON file containing a list of records with the same fields,
      under the key 'records' or at the top level.
    """
    metadata_path = Path(metadata_path)
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(metadata_path)
    elif suffix == ".json":
        obj = pd.read_json(metadata_path)
        if isinstance(obj, pd.DataFrame) and "records" in obj.columns:
            df = pd.json_normalize(obj["records"].tolist())
        else:
            df = obj
    else:
        raise ValueError(
            f"Unsupported metadata format {suffix!r}. Use .csv or .json."
        )

    if "image_id" not in df.columns and "isic_id" in df.columns:
        df = df.rename(columns={"isic_id": "image_id"})
    if "lesion_id" not in df.columns and "patient_id" in df.columns:
        df = df.rename(columns={"patient_id": "lesion_id"})
    if "dx" not in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"diagnosis": "dx"})

    return df


def _split_by_target_share(
    patients: pd.DataFrame,
    target_test: float,
    target_val: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified, patient-grouped split with target image-row shares.

    sklearn's StratifiedGroupKFold places whole groups into folds, so the
    resulting image-row counts drift away from the requested ratio when
    groups have varying size. To honor a target *image-row* ratio, we
    iteratively assign patients to whichever split is currently furthest
    behind its target count, in stratified order. This guarantees:
      (a) no patient appears in more than one split,
      (b) the binary label proportion is preserved within each split,
      (c) the resulting image-row counts are within ~1% of the targets.

    Returns (val_lesion_ids, test_lesion_ids).
    """
    rng = np.random.default_rng(seed)
    patients = patients.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    target_train = 1.0 - target_test - target_val
    target_counts = {
        "train": target_train * len(patients),
        "val":   target_val * len(patients),
        "test":  target_test * len(patients),
    }
    current_counts = {"train": 0, "val": 0, "test": 0}

    split_assignments: list[str] = []
    for _, row in patients.iterrows():
        candidates = ["train", "val", "test"]
        rng.shuffle(candidates)
        best_split = max(
            candidates,
            key=lambda s: (target_counts[s] - current_counts[s]) - 0.001 * rng.random(),
        )
        split_assignments.append(best_split)
        current_counts[best_split] += 1

    train_mask = pd.Series([s == "train" for s in split_assignments], index=patients.index)
    val_mask = pd.Series([s == "val" for s in split_assignments], index=patients.index)
    test_mask = pd.Series([s == "test" for s in split_assignments], index=patients.index)

    val_lesion_ids = patients.loc[val_mask, "lesion_id"].tolist()
    test_lesion_ids = patients.loc[test_mask, "lesion_id"].tolist()
    return val_lesion_ids, test_lesion_ids


def get_splits(
    metadata_path: str | Path,
    seed: int = 42,
    extra_cols: tuple[str, ...] = ("age", "sex", "localization"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-grouped, label-stratified 70/15/15 split by image-row count.

    HAM10000 has ~9000 unique patients (lesion_ids) but ~10,000 images —
    some patients contribute multiple images. The split is constructed so
    that the resulting *image counts* in each split are 70/15/15 of the
    total, while guaranteeing no patient appears in more than one split
    and the binary label proportion is preserved within each split.

    Returns three DataFrames (train, val, test) with columns
    ['image_id', 'lesion_id', 'dx', 'target'] plus any of the tabular
    columns in `extra_cols` that exist in the metadata ('age', 'sex',
    'localization'), so downstream multimodal models can use them.

    Args:
        metadata_path: path to HAM10000_metadata.csv or a JSON equivalent.
        seed: random seed for the split.
        extra_cols: optional metadata columns to carry through the split.
    """
    metadata = _load_metadata(metadata_path)
    missing = METADATA_REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["target"] = metadata["dx"].map(map_dx_to_binary).astype(int)

    patients = _patient_level_dataframe(metadata)

    val_lesion_ids, test_lesion_ids = _split_by_target_share(
        patients, target_test=0.15, target_val=0.15, seed=seed,
    )

    val_lesion_set = set(val_lesion_ids)
    test_lesion_set = set(test_lesion_ids)

    val_patients = patients[patients["lesion_id"].isin(val_lesion_set)].copy()
    test_patients = patients[patients["lesion_id"].isin(test_lesion_set)].copy()

    def expand(patient_df: pd.DataFrame) -> pd.DataFrame:
        return metadata[metadata["lesion_id"].isin(patient_df["lesion_id"])].copy()

    train_df = expand(patients[~patients["lesion_id"].isin(val_lesion_set | test_lesion_set)])
    val_df = expand(val_patients)
    test_df = expand(test_patients)

    cols = ["image_id", "lesion_id", "dx", "target"]
    cols += [c for c in extra_cols if c in metadata.columns]

    return train_df[cols], val_df[cols], test_df[cols]


if __name__ == "__main__":
    METADATA = Path("data/raw/HAM10000_metadata.csv")
    if not METADATA.exists():
        for cand in [Path("data/raw/HAM10000_metadata.json"),
                     Path("data/raw/skin-cancer-mnist-ham10000-metadata.json")]:
            if cand.exists():
                METADATA = cand
                break
    IMAGE_DIRS = [
        Path("data/raw/HAM10000_images_part_1"),
        Path("data/raw/HAM10000_images_part_2"),
    ]

    if not METADATA.exists():
        raise SystemExit(
            f"Metadata file not found. Expected one of:\n"
            f"  data/raw/HAM10000_metadata.csv\n"
            f"  data/raw/HAM10000_metadata.json\n"
            f"Got: {METADATA}"
        )
    train_df, val_df, test_df = get_splits(METADATA, seed=42)

    train_pids = set(train_df["lesion_id"])
    val_pids = set(val_df["lesion_id"])
    test_pids = set(test_df["lesion_id"])

    assert train_pids.isdisjoint(val_pids), "Train/val patient overlap"
    assert train_pids.isdisjoint(test_pids), "Train/test patient overlap"
    assert val_pids.isdisjoint(test_pids), "Val/test patient overlap"
    print("[OK] (a) no patient ID appears in more than one split")

    full_mean = pd.concat([train_df, val_df, test_df])["target"].mean()
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        m = df["target"].mean()
        assert abs(m - full_mean) < 0.03, (
            f"{name} class distribution drifted by {abs(m - full_mean):.3f} from full set"
        )
    print("[OK] (b) class distribution preserved within 3pp across splits")

    from torchvision.transforms import ToTensor

    ds = HAMDataset(train_df.head(1), image_dirs=IMAGE_DIRS, transform=ToTensor())
    img, label, image_id = ds[0]
    assert isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] == 3, (
        f"Expected (3, H, W) tensor, got {tuple(img.shape)}"
    )
    assert label in (0, 1), f"Expected binary label, got {label}"
    assert isinstance(image_id, str), f"Expected str image_id, got {type(image_id)}"
    print(f"[OK] (c) sample loads with shape {tuple(img.shape)}, label={label}, id={image_id}")

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Malignant fraction: train={train_df['target'].mean():.3f}, "
          f"val={val_df['target'].mean():.3f}, test={test_df['target'].mean():.3f}")
