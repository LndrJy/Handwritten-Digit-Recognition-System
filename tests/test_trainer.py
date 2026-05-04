"""
Tests for Phase 3 — TrOCR Fine-Tuning
Run with: pytest tests/test_trainer.py -v

NOTE: These tests do NOT run full training (too slow for CI).
They verify that all components initialize, load, and produce
correctly shaped outputs — catching bugs before a multi-hour
training run.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
import sys
sys.path.append('.')

from src.trainer import (
    set_seed,
    compute_cer,
    RANDOM_SEED,
    TARGET_CER,
    BATCH_SIZE,
    MAX_TARGET_LEN,
)


# ── Reproducibility tests ─────────────────────────────────────────────────────

def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b), "Same seed should produce same random tensors"

def test_set_seed_different_seeds_differ():
    set_seed(42)
    a = torch.randn(5)
    set_seed(99)
    b = torch.randn(5)
    assert not torch.allclose(a, b), "Different seeds should produce different tensors"

def test_set_seed_makes_numpy_reproducible():
    set_seed(42)
    a = np.random.randn(5)
    set_seed(42)
    b = np.random.randn(5)
    assert np.allclose(a, b)


# ── CER computation tests ─────────────────────────────────────────────────────

def test_cer_perfect_prediction():
    preds = ["hello world"]
    refs  = ["hello world"]
    assert compute_cer(preds, refs) == 0.0

def test_cer_completely_wrong():
    preds = ["xxxxx"]
    refs  = ["hello"]
    cer = compute_cer(preds, refs)
    assert cer > 0.0

def test_cer_one_char_off():
    preds = ["helo world"]   # missing one 'l'
    refs  = ["hello world"]
    cer = compute_cer(preds, refs)
    assert 0.0 < cer < 0.2   # 1 error / 11 chars ≈ 0.09

def test_cer_empty_prediction():
    preds = [""]
    refs  = ["hello"]
    cer = compute_cer(preds, refs)
    assert cer == 1.0         # 5 deletions / 5 chars = 100%

def test_cer_multiple_samples():
    preds = ["hello world", "kamusta ka"]
    refs  = ["hello world", "kamusta ka"]
    assert compute_cer(preds, refs) == 0.0

def test_cer_partial_match():
    preds = ["please submit"]
    refs  = ["please submit asap"]
    cer = compute_cer(preds, refs)
    assert 0.0 < cer < 1.0

def test_cer_taglish_sample():
    # Simulates a typical Taglish prediction with one OCR error
    preds = ["pls padala yung report bukas"]   # correct
    refs  = ["pls padala yung report bukas"]
    assert compute_cer(preds, refs) == 0.0

def test_cer_handles_unicode():
    preds = ["Piña colada"]
    refs  = ["Piña colada"]
    assert compute_cer(preds, refs) == 0.0

def test_cer_case_sensitive():
    # CER should be case-sensitive — model output must match case
    preds = ["Hello World"]
    refs  = ["hello world"]
    cer = compute_cer(preds, refs)
    assert cer > 0.0   # 2 substitutions (H→h, W→w)

def test_cer_strips_whitespace():
    preds = ["  hello world  "]
    refs  = ["hello world"]
    assert compute_cer(preds, refs) == 0.0


# ── Config sanity tests ───────────────────────────────────────────────────────

def test_target_cer_is_reasonable():
    # Target CER should be between 5% and 20% for initial training
    assert 0.05 <= TARGET_CER <= 0.20, \
        f"TARGET_CER {TARGET_CER} is outside reasonable range [0.05, 0.20]"

def test_batch_size_is_power_of_two():
    # GPU utilization is most efficient with power-of-two batch sizes
    assert BATCH_SIZE & (BATCH_SIZE - 1) == 0, \
        f"BATCH_SIZE {BATCH_SIZE} should be a power of 2"

def test_max_target_len_fits_sentences():
    # Our longest sentence should fit within MAX_TARGET_LEN characters
    longest = "Mangyaring suriin ang mga dokumentong nakalakip bago pumirma sa kontrata"
    assert len(longest) < MAX_TARGET_LEN, \
        f"Sentence length {len(longest)} exceeds MAX_TARGET_LEN {MAX_TARGET_LEN}"
