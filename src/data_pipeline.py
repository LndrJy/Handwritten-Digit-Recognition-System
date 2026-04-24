"""
Phase 1 — Data Engineering Pipeline
Handwriting Recognition Engine
================================
Loads MNIST + EMNIST ByClass, merges them, applies augmentation,
and produces versioned train/val/test DataLoaders ready for training.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from collections import Counter


# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR       = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
IMG_SIZE       = 28          # MNIST / EMNIST native resolution
BATCH_SIZE     = 128
NUM_WORKERS    = 4
VAL_SPLIT      = 0.10        # 10% of train → validation
TEST_SPLIT     = 0.10        # 10% of train → test
RANDOM_SEED    = 42
EMNIST_SPLIT   = "byclass"   # 62 classes: digits + A-Z + a-z

# EMNIST ByClass class mapping (0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z)
# Some visually similar pairs are merged in ByMerge but NOT in ByClass
# We keep ByClass for maximum class coverage in a backbone engine
NUM_CLASSES_MNIST  = 10
NUM_CLASSES_EMNIST = 62


# ── Label mapping ─────────────────────────────────────────────────────────────

def build_label_map():
    """
    EMNIST ByClass label mapping:
      0-9   → '0'-'9'
      10-35 → 'A'-'Z'
      36-61 → 'a'-'z'
    MNIST labels 0-9 map directly to the same digit slots.
    Returns dict: {int_label: str_character}
    """
    label_map = {}
    for i in range(10):
        label_map[i] = str(i)
    for i in range(26):
        label_map[10 + i] = chr(ord('A') + i)
    for i in range(26):
        label_map[36 + i] = chr(ord('a') + i)
    return label_map


LABEL_MAP = build_label_map()


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Augmentation pipeline for training.
    Designed to simulate real-world handwriting variation:
    - Random rotation: pens tilt when writing
    - Affine shear: writing slant variation
    - Elastic distortion: natural stroke wobble
    - Gaussian noise: scanner/camera noise
    - Normalize to [-1, 1] for faster convergence
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            shear=(-10, 10),
            translate=(0.05, 0.05)
        ),
        transforms.ElasticTransform(alpha=30.0, sigma=4.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=0.3
        ),
    ])


def get_eval_transforms():
    """
    Clean transforms for validation and test — no augmentation.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_mnist(root: Path, train: bool, transform):
    """Load MNIST (digits 0-9). Labels are already compatible with EMNIST ByClass."""
    print(f"  Loading MNIST ({'train' if train else 'test'})...")
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    print(f"    {len(dataset):,} samples, 10 classes (digits 0-9)")
    return dataset


def load_emnist(root: Path, train: bool, transform):
    """
    Load EMNIST ByClass (digits + A-Z + a-z = 62 classes).
    IMPORTANT: EMNIST images are transposed (90° rotated + flipped).
    We apply a fix transform automatically below.
    """
    print(f"  Loading EMNIST ByClass ({'train' if train else 'test'})...")

    # EMNIST quirk: images are stored rotated 90° CCW and mirrored
    # We need to add a fix on top of the user transform
    class EMNISTFixTransform:
        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, img):
            # Fix EMNIST orientation before applying main transforms
            img = TF.rotate(img, -90)
            img = TF.hflip(img)
            return self.base_transform(img)

    fixed_transform = EMNISTFixTransform(transform)

    dataset = datasets.EMNIST(
        root=root,
        split=EMNIST_SPLIT,
        train=train,
        download=True,
        transform=fixed_transform
    )
    print(f"    {len(dataset):,} samples, {len(set(dataset.targets.tolist())):,} classes")
    return dataset


# ── Dataset merging ───────────────────────────────────────────────────────────

def merge_datasets(mnist_dataset, emnist_dataset):
    """
    Merge MNIST and EMNIST into a single ConcatDataset.
    MNIST digits (0-9) overlap with EMNIST digits (0-9) — this is intentional.
    More digit samples = more robust digit recognition.
    Total unique classes: 62 (EMNIST ByClass covers all of them).
    """
    merged = ConcatDataset([mnist_dataset, emnist_dataset])
    print(f"  Merged dataset: {len(merged):,} total samples")
    return merged


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_dataset(dataset, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=RANDOM_SEED):
    """
    Split a dataset into train / val / test subsets.
    Uses a fixed random seed for full reproducibility.
    """
    total = len(dataset)
    n_test  = int(total * test_split)
    n_val   = int(total * val_split)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    print(f"  Split → train: {len(train_set):,} | val: {len(val_set):,} | test: {len(test_set):,}")
    return train_set, val_set, test_set


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(train_set, val_set, test_set,
                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Build PyTorch DataLoaders for all three splits."""
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True         # avoids incomplete last batch during training
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ── Dataset manifest (versioning) ─────────────────────────────────────────────

def save_manifest(train_set, val_set, test_set, output_dir: Path):
    """
    Save a JSON manifest describing the dataset version.
    This is what makes the dataset reproducible across team members.
    Commit this file to git alongside your code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "emnist_split": EMNIST_SPLIT,
        "image_size": IMG_SIZE,
        "num_classes": NUM_CLASSES_EMNIST,
        "splits": {
            "train": len(train_set),
            "val":   len(val_set),
            "test":  len(test_set),
            "total": len(train_set) + len(val_set) + len(test_set),
        },
        "val_split_ratio":  VAL_SPLIT,
        "test_split_ratio": TEST_SPLIT,
        "label_map": LABEL_MAP,
        "augmentations": [
            "RandomRotation(10deg)",
            "RandomAffine(shear=10, translate=0.05)",
            "ElasticTransform(alpha=30, sigma=4)",
            "GaussianBlur(p=0.3)",
            "Normalize(mean=0.5, std=0.5)"
        ],
        "notes": "EMNIST images are orientation-corrected (90deg CCW + hflip fix applied)."
    }

    manifest_path = output_dir / "dataset_manifest_v1.0.0.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest saved → {manifest_path}")
    return manifest


# ── Class distribution check ──────────────────────────────────────────────────

def check_class_distribution(dataset, label_map, name="dataset", top_n=10):
    """
    Print the top-N and bottom-N classes by sample count.
    Important for spotting class imbalance before training.
    """
    print(f"\n  Class distribution check — {name}")

    # Sample up to 50k items to keep this fast
    n_sample = min(50000, len(dataset))
    indices = torch.randperm(len(dataset))[:n_sample]

    labels = []
    for i in indices:
        _, label = dataset[i]
        labels.append(int(label))

    counts = Counter(labels)
    most_common   = counts.most_common(top_n)
    least_common  = counts.most_common()[:-top_n-1:-1]

    print(f"    Top {top_n} most common classes:")
    for label, count in most_common:
        char = label_map.get(label, f"cls_{label}")
        print(f"      '{char}' (label {label:>2}): {count:,}")

    print(f"    Bottom {top_n} least common classes:")
    for label, count in least_common:
        char = label_map.get(label, f"cls_{label}")
        print(f"      '{char}' (label {label:>2}): {count:,}")


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(loader, label_map, num_batches=2):
    """
    Pull a few batches and verify shapes, dtypes, and label ranges.
    Catches transform bugs before a multi-hour training run.
    """
    print("\n  Running sanity checks...")
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        assert images.shape[1] == 1,        f"Expected 1 channel, got {images.shape[1]}"
        assert images.shape[2] == IMG_SIZE,  f"Expected height {IMG_SIZE}, got {images.shape[2]}"
        assert images.shape[3] == IMG_SIZE,  f"Expected width {IMG_SIZE}, got {images.shape[3]}"
        assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
        assert images.min() >= -1.5 and images.max() <= 1.5, \
            f"Pixel range unexpected: [{images.min():.2f}, {images.max():.2f}]"
        assert labels.min() >= 0 and labels.max() < NUM_CLASSES_EMNIST, \
            f"Label out of range: [{labels.min()}, {labels.max()}]"

        sample_labels = [label_map.get(int(l), f"cls_{l}") for l in labels[:8]]
        print(f"    Batch {i+1}: shape={tuple(images.shape)} "
              f"range=[{images.min():.2f}, {images.max():.2f}] "
              f"sample_labels={sample_labels}")

    print("  All sanity checks passed.")


# ── Main entry point ──────────────────────────────────────────────────────────

def build_pipeline(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    verbose=True
):
    """
    Full Phase 1 pipeline. Returns (train_loader, val_loader, test_loader, manifest).
    Call this from your training script.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data_dir      = Path(data_dir)
    processed_dir = Path(processed_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Phase 1 — Data Engineering Pipeline")
    print("="*60)

    # 1. Transforms
    train_tf = get_train_transforms()
    eval_tf  = get_eval_transforms()

    # 2. Load raw datasets
    print("\n[1/5] Loading datasets...")
    mnist_train  = load_mnist(data_dir, train=True,  transform=train_tf)
    mnist_test   = load_mnist(data_dir, train=False, transform=eval_tf)
    emnist_train = load_emnist(data_dir, train=True,  transform=train_tf)
    emnist_test  = load_emnist(data_dir, train=False, transform=eval_tf)

    # 3. Merge
    print("\n[2/5] Merging MNIST + EMNIST...")
    train_full = merge_datasets(mnist_train, emnist_train)
    test_full  = merge_datasets(mnist_test,  emnist_test)

    # 4. Split train → train / val / test
    print("\n[3/5] Splitting into train / val / test...")
    train_set, val_set, _ = split_dataset(train_full)
    # Use the dedicated test sets from both datasets
    _, _, test_from_train  = split_dataset(train_full)
    # Final test set = merged official test data
    test_set = test_full

    # 5. Build loaders
    print("\n[4/5] Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set, batch_size, num_workers
    )
    print(f"  Batches → train: {len(train_loader):,} | "
          f"val: {len(val_loader):,} | test: {len(test_loader):,}")

    # 6. Save manifest
    print("\n[5/5] Saving dataset manifest...")
    manifest = save_manifest(train_set, val_set, test_set, processed_dir)

    # 7. Sanity check
    sanity_check(train_loader, LABEL_MAP)

    print("\n" + "="*60)
    print("  Phase 1 complete. Pipeline ready for training.")
    print("="*60 + "\n")

    return train_loader, val_loader, test_loader, manifest


# ── Run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, manifest = build_pipeline()

    print("Dataset summary:")
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"  Test batches  : {len(test_loader):,}")
    print(f"  Total samples : {manifest['splits']['total']:,}")
    print(f"  Num classes   : {manifest['num_classes']}")
    print(f"  Label map     : {list(manifest['label_map'].values())}")
