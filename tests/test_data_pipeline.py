"""
Tests for Phase 1 — Data Engineering Pipeline
Run with: pytest tests/test_data_pipeline.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data_pipeline import (
    build_label_map,
    get_train_transforms,
    get_eval_transforms,
    split_dataset,
    LABEL_MAP,
    NUM_CLASSES_EMNIST,
    IMG_SIZE,
    RANDOM_SEED,
)
from torch.utils.data import TensorDataset


# ── Label map tests ───────────────────────────────────────────────────────────

def test_label_map_digits():
    lm = build_label_map()
    for i in range(10):
        assert lm[i] == str(i), f"Expected digit '{i}' at label {i}"

def test_label_map_uppercase():
    lm = build_label_map()
    for i in range(26):
        expected = chr(ord('A') + i)
        assert lm[10 + i] == expected, f"Expected '{expected}' at label {10+i}"

def test_label_map_lowercase():
    lm = build_label_map()
    for i in range(26):
        expected = chr(ord('a') + i)
        assert lm[36 + i] == expected, f"Expected '{expected}' at label {36+i}"

def test_label_map_total_classes():
    lm = build_label_map()
    assert len(lm) == NUM_CLASSES_EMNIST, \
        f"Expected {NUM_CLASSES_EMNIST} classes, got {len(lm)}"


# ── Transform tests ───────────────────────────────────────────────────────────

def test_train_transform_output_shape():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
    transform = get_train_transforms()
    tensor = transform(img)
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE), \
        f"Expected (1, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"

def test_eval_transform_output_shape():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
    transform = get_eval_transforms()
    tensor = transform(img)
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE)

def test_transform_normalization_range():
    from PIL import Image
    import numpy as np
    # All-white image → should normalize to ~1.0
    img = Image.fromarray(np.full((28, 28), 255, dtype=np.uint8), mode='L')
    transform = get_eval_transforms()
    tensor = transform(img)
    assert tensor.min() > 0.5, "All-white image should normalize above 0.5"

def test_transform_dtype():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
    transform = get_eval_transforms()
    tensor = transform(img)
    assert tensor.dtype == torch.float32


# ── Split tests ───────────────────────────────────────────────────────────────

def make_dummy_dataset(n=1000):
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 62, (n,))
    return TensorDataset(images, labels)

def test_split_sizes():
    ds = make_dummy_dataset(1000)
    train, val, test = split_dataset(ds, val_split=0.1, test_split=0.1)
    assert len(train) + len(val) + len(test) == 1000

def test_split_no_overlap():
    ds = make_dummy_dataset(1000)
    train, val, test = split_dataset(ds, val_split=0.1, test_split=0.1)
    train_idx = set(train.indices)
    val_idx   = set(val.indices)
    test_idx  = set(test.indices)
    assert train_idx.isdisjoint(val_idx),  "Train and val overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Val and test overlap!"

def test_split_reproducibility():
    ds = make_dummy_dataset(1000)
    train1, val1, test1 = split_dataset(ds, seed=RANDOM_SEED)
    train2, val2, test2 = split_dataset(ds, seed=RANDOM_SEED)
    assert train1.indices == train2.indices, "Splits not reproducible with same seed"
    assert val1.indices   == val2.indices
    assert test1.indices  == test2.indices

def test_split_different_seeds_differ():
    ds = make_dummy_dataset(1000)
    train1, _, _ = split_dataset(ds, seed=42)
    train2, _, _ = split_dataset(ds, seed=99)
    assert train1.indices != train2.indices, "Different seeds should produce different splits"
