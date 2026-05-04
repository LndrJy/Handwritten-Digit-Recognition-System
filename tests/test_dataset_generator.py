"""
Tests for Phase 2 — Synthetic Taglish Line Dataset Generator
Run with: pytest tests/test_dataset_generator.py -v

WHAT THESE TESTS DO:
  Verify every function in dataset_generator.py behaves correctly.
  Each test is named to describe what it is checking — the test name
  IS the documentation. A failing test tells you exactly what broke.

WHY TEST PHASE 2?
  The dataset is the foundation of the entire model. A bug here —
  wrong labels, wrong image sizes, wrong splits — will corrupt training
  silently. You won't discover it until the model performs poorly, and
  by then you've wasted days of compute and time.
"""

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.append('.')
from src.dataset_generator import (
    load_corpus,
    load_abbreviations,
    expand_abbreviations,
    render_line,
    augment_line,
    IMG_HEIGHT,
    MAX_WIDTH,
    RANDOM_SEED,
)
from PIL import ImageFont
import random


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_temp_corpus(sentences: list) -> Path:
    """Write a temporary corpus file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      delete=False, encoding="utf-8")
    tmp.write("# Comment line — should be ignored\n")
    tmp.write("\n")
    for s in sentences:
        tmp.write(s + "\n")
    tmp.close()
    return Path(tmp.name)

def make_temp_abbrev(entries: dict) -> Path:
    """Write a temporary abbreviations JSON and return its path."""
    data = {"_metadata": {}, "english_abbreviations": entries}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)

def get_default_font():
    """Return a default PIL font for rendering tests."""
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ── Corpus loading tests ──────────────────────────────────────────────────────

def test_corpus_strips_comments():
    path = make_temp_corpus(["Hello world", "Kamusta ka"])
    sentences = load_corpus(path)
    assert all(not s.startswith("#") for s in sentences)

def test_corpus_strips_blank_lines():
    path = make_temp_corpus(["Hello world", "", "Kamusta ka"])
    sentences = load_corpus(path)
    assert all(len(s) > 0 for s in sentences)

def test_corpus_deduplicates():
    path = make_temp_corpus(["Hello world", "Hello world", "Kamusta ka"])
    sentences = load_corpus(path)
    assert len(sentences) == 2

def test_corpus_preserves_order():
    expected = ["First sentence", "Second sentence", "Third sentence"]
    path = make_temp_corpus(expected)
    sentences = load_corpus(path)
    assert sentences == expected

def test_corpus_handles_unicode():
    path = make_temp_corpus(["Piña colada", "Señor Lopez", "Niño"])
    sentences = load_corpus(path)
    assert "Piña colada" in sentences
    assert "Señor Lopez" in sentences


# ── Abbreviation loading tests ────────────────────────────────────────────────

def test_abbreviations_loads_correctly():
    path = make_temp_abbrev({"pls": "please", "asap": "as soon as possible"})
    abbrev = load_abbreviations(path)
    assert "pls" in abbrev
    assert "asap" in abbrev

def test_abbreviations_skips_metadata():
    path = make_temp_abbrev({"pls": "please"})
    abbrev = load_abbreviations(path)
    assert "_metadata" not in abbrev

def test_abbreviations_flattens_categories():
    data = {
        "_metadata": {},
        "english_abbreviations": {"pls": "please"},
        "tagalog_abbreviations": {"atbp": "at iba pa"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()
    abbrev = load_abbreviations(Path(tmp.name))
    assert "pls" in abbrev
    assert "atbp" in abbrev


# ── Abbreviation expansion tests ──────────────────────────────────────────────

def test_expansion_replaces_known_abbreviation():
    abbrev = {"pls": "please"}
    result = expand_abbreviations("pls submit", abbrev)
    assert "please" in result

def test_expansion_preserves_unknown_words():
    abbrev = {"pls": "please"}
    result = expand_abbreviations("submit the report", abbrev)
    assert result == "submit the report"

def test_expansion_is_case_insensitive():
    abbrev = {"pls": "please"}
    result = expand_abbreviations("PLS submit", abbrev)
    assert "please" in result.lower()

def test_expansion_preserves_trailing_punctuation():
    abbrev = {"pls": "please"}
    result = expand_abbreviations("pls.", abbrev)
    assert result.endswith(".")

def test_expansion_handles_empty_string():
    abbrev = {"pls": "please"}
    result = expand_abbreviations("", abbrev)
    assert result == ""

def test_expansion_handles_multiple_abbreviations():
    abbrev = {"pls": "please", "asap": "as soon as possible"}
    result = expand_abbreviations("pls submit asap", abbrev)
    assert "please" in result
    assert "as soon as possible" in result


# ── Image rendering tests ─────────────────────────────────────────────────────

def test_render_line_output_size():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    img = render_line("Hello world", font)
    assert img.size == (MAX_WIDTH, IMG_HEIGHT)

def test_render_line_is_grayscale():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    img = render_line("Kamusta", font)
    assert img.mode == "L"

def test_render_line_not_all_white():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    import numpy as np
    img = render_line("Test", font)
    arr = np.array(img)
    assert arr.min() < 255, "Rendered image is entirely white — text not drawn"

def test_render_line_handles_long_text():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    long_text = "This is a very long sentence that might overflow the canvas width"
    img = render_line(long_text, font)
    assert img.size == (MAX_WIDTH, IMG_HEIGHT)

def test_render_line_handles_unicode():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    img = render_line("Piña at Señor", font)
    assert img.size == (MAX_WIDTH, IMG_HEIGHT)


# ── Augmentation tests ────────────────────────────────────────────────────────

def test_augment_preserves_size():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    img = render_line("Test", font)
    rng = random.Random(42)
    augmented = augment_line(img, rng)
    assert augmented.size == img.size

def test_augment_preserves_mode():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    img = render_line("Test", font)
    rng = random.Random(42)
    augmented = augment_line(img, rng)
    assert augmented.mode == "L"

def test_augment_produces_different_image():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    import numpy as np
    img = render_line("Test sentence", font)
    rng = random.Random(99)
    augmented = augment_line(img, rng)
    arr_orig = np.array(img)
    arr_aug  = np.array(augmented)
    assert not np.array_equal(arr_orig, arr_aug), \
        "Augmented image is identical to original — augmentation had no effect"

def test_augment_reproducible_with_same_seed():
    font = get_default_font()
    if font is None:
        pytest.skip("No default font available")
    import numpy as np
    img = render_line("Test", font)
    aug1 = augment_line(img.copy(), random.Random(42))
    aug2 = augment_line(img.copy(), random.Random(42))
    assert np.array_equal(np.array(aug1), np.array(aug2)), \
        "Same seed should produce identical augmentation"
