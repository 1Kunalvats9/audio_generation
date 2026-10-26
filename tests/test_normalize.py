"""Tests for src/text/normalize.py. Run: python -m pytest tests"""

import pytest

from src.text import normalize
from src.text.normalize import (
    clean_text,
    collapse_whitespace,
    expand_abbreviations,
    handle_brackets,
    normalize_punctuation,
    strip_accents,
)


# ---------- one rule at a time ----------

@pytest.mark.parametrize("raw, expected", [
    ("don\u2019t", "don't"),          # curly apostrophe
    ("\u201cYes\u201d", '"Yes"'),     # curly double quotes
    ("a\u2014b", "a - b"),            # long dash
])
def test_normalize_punctuation(raw, expected):
    assert normalize_punctuation(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("caf\u00e9", "cafe"),
    ("W\u00fcrzburg", "Wurzburg"),
    ("\u00e2 \u00e0 \u00ea \u00e8", "a a e e"),
])
def test_strip_accents(raw, expected):
    assert strip_accents(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Mr. Smith", "mister Smith"),
    ("Mrs. Jones", "missus Jones"),          # "mrs" must win over "mr"
    ("MR. LOUD", "mister LOUD"),             # case-insensitive
    ("Dr. and Col. Brown", "doctor and colonel Brown"),
    ("the co said", "the co said"),          # no full stop -> left alone
])
def test_expand_abbreviations(raw, expected):
    assert expand_abbreviations(raw) == expected


def test_round_brackets_keep_words():
    out = collapse_whitespace(handle_brackets("the man (a clerk) said"))
    assert out == "the man a clerk said"


def test_square_brackets_keep_words(monkeypatch):
    monkeypatch.setattr(normalize, "REMOVE_SQUARE_BRACKET_CONTENT", False)
    out = collapse_whitespace(handle_brackets("he said [quietly] no"))
    assert out == "he said quietly no"


def test_square_brackets_removed(monkeypatch):
    monkeypatch.setattr(normalize, "REMOVE_SQUARE_BRACKET_CONTENT", True)
    out = collapse_whitespace(handle_brackets("he said [quietly] no"))
    assert out == "he said no"


@pytest.mark.parametrize("raw, expected", [
    ("too    many   spaces ", "too many spaces"),
    ("end of it .", "end of it."),           # no space before punctuation
    ("a ,b", "a,b"),
])
def test_collapse_whitespace(raw, expected):
    assert collapse_whitespace(raw) == expected


# ---------- the whole pipeline ----------

@pytest.mark.parametrize("raw, expected", [
    ("Mr. Smith\u2019s house", "mister smith's house"),
    ("a caf\u00e9 in W\u00fcrzburg", "a cafe in wurzburg"),
    ("the man (a clerk) said", "the man a clerk said"),
    ("\u201cYes,\u201d said Mrs. Jones.", '"yes," said missus jones.'),
    ("too    many   spaces ", "too many spaces"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_output_is_ascii_and_lowercase():
    out = clean_text("\u00dc \u00c9 \u201cQ\u201d (X) [Y] Mr. Z")
    assert out.isascii()
    assert out == out.lower()


def test_cleaning_twice_changes_nothing():
    text = "\u201cMr. Smith\u2019s caf\u00e9 (old) [new]\u201d"
    once = clean_text(text)
    assert clean_text(once) == once