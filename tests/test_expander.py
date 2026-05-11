"""
Tests for Phase 4 — Abbreviation Expansion Engine
Run with: pytest tests/test_expander.py -v

COVERAGE:
  - Language detection (English / Tagalog / Taglish)
  - Dictionary loading and flattening
  - Token expansion — basic, case preservation, punctuation
  - Context rules — blocking over-expansion
  - ExpandedResult dataclass
  - Document-level expansion
  - Edge cases — empty string, unknown tokens, all-caps
"""

import pytest
import sys
sys.path.append('.')

from src.expander import (
    AbbreviationExpander,
    ExpandedResult,
    detect_language,
    load_abbreviations,
    should_block_expansion,
    DO_NOT_EXPAND_PATTERNS,
)
from pathlib import Path
import json
import tempfile


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_expander(extra_entries: dict = None) -> AbbreviationExpander:
    """Create an expander with the built-in fallback dictionary."""
    exp = AbbreviationExpander(abbrev_path=None)
    if extra_entries:
        exp.abbrev_dict.update(extra_entries)
    return exp

def make_abbrev_file(entries: dict) -> Path:
    """Write a temp abbreviation JSON file and return its path."""
    data = {"_metadata": {}, "english_abbreviations": entries}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_detect_language_english():
    text = "Please submit the report by Friday afternoon"
    assert detect_language(text) == "english"

def test_detect_language_tagalog():
    # A heavily Tagalog sentence should be classified as tagalog or taglish
    # (not english). We don't enforce a strict "tagalog" because the
    # threshold depends on how many markers appear in the specific sentence.
    text = "Pakipadala ang ulat bago mag-Biyernes"
    lang = detect_language(text)
    assert lang in ("tagalog", "taglish"), f"Expected tagalog or taglish, got {lang}"

def test_detect_language_taglish():
    text = "Pls submit ang report bago mag-Friday"
    assert detect_language(text) == "taglish"

def test_detect_language_empty():
    assert detect_language("") == "unknown"

def test_detect_language_single_tagalog_marker():
    # Even one Tagalog marker should push it away from pure English
    text = "Please submit ang report"
    lang = detect_language(text)
    assert lang in ("taglish", "tagalog")

def test_detect_language_numbers_only():
    text = "2x daily 500mg 3 times"
    assert detect_language(text) in ("english", "unknown")

def test_detect_language_case_insensitive():
    # "ANG" should still be detected as Tagalog marker
    text = "ANG ulat ay handa na"
    lang = detect_language(text)
    assert lang in ("tagalog", "taglish")


# ══════════════════════════════════════════════════════════════════════════════
# ABBREVIATION LOADING TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_load_abbreviations_flattens_categories():
    path = make_abbrev_file({"pls": "please", "asap": "as soon as possible"})
    d = load_abbreviations(path)
    assert "pls" in d
    assert "asap" in d

def test_load_abbreviations_skips_metadata():
    path = make_abbrev_file({"pls": "please"})
    d = load_abbreviations(path)
    assert "_metadata" not in d

def test_load_abbreviations_lowercases_keys():
    path = make_abbrev_file({"PLS": "please", "ASAP": "as soon as possible"})
    d = load_abbreviations(path)
    assert "pls" in d
    assert "asap" in d
    assert "PLS" not in d

def test_load_abbreviations_unicode():
    path = make_abbrev_file({"atbp": "at iba pa", "pki": "pakiusap"})
    d = load_abbreviations(path)
    assert d["atbp"] == "at iba pa"


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT RULE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_should_block_re_as_prefix():
    # "re-submit" — "re" is a prefix, not "regarding"
    assert should_block_expansion("re", "please re-submit the form") is True

def test_should_not_block_re_as_abbreviation():
    # "re:" at start — classic email abbreviation for "regarding"
    assert should_block_expansion("re", "re: the last meeting") is False

def test_should_block_it_as_pronoun():
    assert should_block_expansion("it", "it is confirmed") is True

def test_should_not_block_it_as_abbreviation():
    # "IT dept" — information technology
    assert should_block_expansion("it", "contact the IT dept") is False

def test_should_block_am_as_verb():
    assert should_block_expansion("am", "i am ready") is True

def test_should_not_block_am_as_time():
    # "9 AM" — time abbreviation
    assert should_block_expansion("am", "meeting at 9 am tomorrow") is False

def test_unknown_token_never_blocked():
    # Token not in DO_NOT_EXPAND_PATTERNS should never be blocked
    assert should_block_expansion("asap", "submit asap") is False
    assert should_block_expansion("pls", "pls submit") is False


# ══════════════════════════════════════════════════════════════════════════════
# EXPANSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_expand_basic():
    exp = make_expander()
    result = exp.expand("pls submit asap")
    assert "please" in result.expanded
    assert "as soon as possible" in result.expanded

def test_expand_preserves_unknowns():
    exp = make_expander()
    result = exp.expand("please submit the kamusta")
    assert "kamusta" in result.expanded

def test_expand_empty_string():
    exp = make_expander()
    result = exp.expand("")
    assert result.expanded == ""
    assert result.confidence == 1.0

def test_expand_no_abbreviations():
    exp = make_expander()
    text = "Please submit the report by Friday"
    result = exp.expand(text)
    assert result.expanded == text
    assert len(result.changes) == 0

def test_expand_records_changes():
    exp = make_expander()
    result = exp.expand("pls submit asap")
    assert len(result.changes) >= 1
    original_tokens = [c[0].lower() for c in result.changes]
    assert "pls" in original_tokens

def test_expand_uppercase_preserved():
    exp = make_expander()
    result = exp.expand("ASAP")
    assert result.expanded.isupper()

def test_expand_titlecase_preserved():
    exp = make_expander()
    result = exp.expand("Asap")
    assert result.expanded[0].isupper()

def test_expand_lowercase_preserved():
    exp = make_expander()
    result = exp.expand("asap")
    assert result.expanded.islower()

def test_expand_with_trailing_punctuation():
    exp = make_expander()
    result = exp.expand("submit pls.")
    assert "please" in result.expanded
    # Trailing period should be preserved
    assert result.expanded.endswith(".")

def test_expand_tagalog_abbreviation():
    exp = make_expander()
    result = exp.expand("atbp")
    assert "at iba pa" in result.expanded

def test_expand_numeric_abbreviation():
    exp = make_expander()
    result = exp.expand("take 2x daily")
    assert "dalawang beses" in result.expanded

def test_expand_mixed_taglish():
    exp = make_expander()
    result = exp.expand("pls submit ang rpt asap")
    assert "please" in result.expanded
    assert "report" in result.expanded
    assert "as soon as possible" in result.expanded
    # Tagalog "ang" should be preserved
    assert "ang" in result.expanded

def test_expand_confidence_is_float():
    exp = make_expander()
    result = exp.expand("pls submit asap")
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

def test_expand_confidence_high_for_no_abbreviations():
    exp = make_expander()
    result = exp.expand("Please submit the report by Friday")
    assert result.confidence >= 0.95

def test_expand_language_detected():
    exp = make_expander()
    result = exp.expand("Please submit the report")
    assert result.language == "english"

def test_expand_taglish_language_detected():
    exp = make_expander()
    result = exp.expand("pls submit ang rpt")
    assert result.language in ("taglish", "tagalog")


# ══════════════════════════════════════════════════════════════════════════════
# EXPANDED RESULT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_expanded_result_to_dict():
    r = ExpandedResult(
        original="pls submit",
        expanded="please submit",
        changes=[("pls", "please")],
        confidence=0.97,
        language="english",
    )
    d = r.to_dict()
    assert d["original"] == "pls submit"
    assert d["expanded"] == "please submit"
    assert d["confidence"] == 0.97
    assert d["language"] == "english"
    assert len(d["changes"]) == 1

def test_expanded_result_default_fields():
    r = ExpandedResult(original="test", expanded="test")
    assert r.changes == []
    assert r.skipped == []
    assert r.confidence == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT-LEVEL EXPANSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_expand_document_returns_list():
    exp = make_expander()
    lines = ["pls submit", "asap please", "thank you"]
    results = exp.expand_document(lines)
    assert isinstance(results, list)
    assert len(results) == 3

def test_expand_document_skips_blank_lines():
    exp = make_expander()
    lines = ["pls submit", "", "   ", "asap"]
    results = exp.expand_document(lines)
    assert len(results) == 2  # blank lines skipped

def test_expand_document_each_line_independent():
    exp = make_expander()
    lines = ["pls submit", "please confirm"]
    results = exp.expand_document(lines)
    # Each result should have the correct original
    assert results[0].original == "pls submit"
    assert results[1].original == "please confirm"

def test_expand_batch_alias():
    exp = make_expander()
    lines = ["pls submit", "asap"]
    r1 = exp.expand_document(lines)
    r2 = exp.expand_batch(lines)
    assert len(r1) == len(r2)
    assert r1[0].expanded == r2[0].expanded
