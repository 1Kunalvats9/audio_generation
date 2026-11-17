"""Phase 2: text -> phonemes for sentences nobody recorded (M1).

These tests never construct g2p_en (it is slow and downloads NLTK data). They
use words present in the corpus lexicon, and monkeypatch the engine where the
fallback path itself is what is under test.
"""

import json

import pytest

from src.config import SIL
from src.dl.g2p import G2P, expand_numbers, load_lexicon
from src.dl.vocab import PhoneVocab


@pytest.fixture
def g2p(tmp_path):
    """A G2P with small, known lexicons and an engine that refuses to be called."""
    oov = tmp_path / "oov.dict"
    oov.write_text("zzqq\tZ IY1 K Y UW1\n")
    g = G2P(vocab=PhoneVocab.load(), lexicon_path=oov)
    g.corpus = {"hello": ["HH", "AH0", "L", "OW1"], "world": ["W", "ER1", "L", "D"],
                "two": ["T", "UW1"], "and": ["AE1", "N", "D"]}
    def _no_engine():
        raise AssertionError("the g2p_en fallback should not have been reached")
    g._g2p_engine = _no_engine
    return g


# ---------------------------------------------------------------------------
# Number expansion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_words", [
    ("in 2026", ["two", "thousand", "and", "twenty", "six"]),
    ("3 cats", ["three"]),
    ("no digits here", []),
])
def test_expand_numbers(text, expected_words):
    out = expand_numbers(text).lower()
    for w in expected_words:
        assert w in out
    assert not any(ch.isdigit() for ch in out)


def test_expansion_leaves_no_hyphens_to_break_lookup():
    """"twenty-six" is not a dictionary word; it must become two tokens."""
    assert "-" not in expand_numbers("26")


# ---------------------------------------------------------------------------
# Pronunciation sources and their priority
# ---------------------------------------------------------------------------


def test_corpus_lexicon_wins_over_the_oov_dictionary(tmp_path):
    oov = tmp_path / "oov.dict"
    oov.write_text("hello\tZ Z Z\n")
    g = G2P(vocab=PhoneVocab.load(), lexicon_path=oov)
    g.corpus = {"hello": ["HH", "AH0", "L", "OW1"]}
    assert g.pronounce("hello") == ["HH", "AH0", "L", "OW1"]


def test_oov_dictionary_is_used_when_the_corpus_has_not_seen_the_word(g2p):
    assert g2p.pronounce("zzqq") == ["Z", "IY1", "K", "Y", "UW1"]


def test_unseen_word_falls_through_to_the_engine(tmp_path):
    g = G2P(vocab=PhoneVocab.load(), lexicon_path=tmp_path / "missing.dict")
    g.corpus = {}
    g._g2p_engine = lambda: (lambda w: ["T", "EH1", "S", "T", " ", "."])
    assert g.pronounce("whatever") == ["T", "EH1", "S", "T"], "non-phone tokens must be dropped"
    assert "whatever" in g.unknown_words


def test_pronounce_is_case_insensitive(g2p):
    assert g2p.pronounce("HELLO") == g2p.pronounce("hello")


def test_load_lexicon_handles_tabs_and_spaces(tmp_path):
    p = tmp_path / "d.dict"
    p.write_text("alpha\tAE1 L F AH0\nbeta B EY1 T AH0\n# a comment\n\n")
    lex = load_lexicon(p)
    assert lex == {"alpha": ["AE1", "L", "F", "AH0"], "beta": ["B", "EY1", "T", "AH0"]}


def test_missing_lexicon_file_is_not_fatal(tmp_path):
    assert load_lexicon(tmp_path / "nope.dict") == {}


# ---------------------------------------------------------------------------
# Sentences and silence
# ---------------------------------------------------------------------------


def test_every_symbol_is_in_the_frozen_vocabulary(g2p):
    known = set(g2p.vocab.symbols)
    assert all(p in known for p in g2p.phonemes("hello world"))


def test_a_sentence_always_ends_in_silence(g2p):
    """100% of training clips end with silence, so synthesis must too."""
    assert g2p.phonemes("hello world")[-1] == SIL
    assert g2p.phonemes("hello world.")[-1] == SIL


def test_leading_silence_is_off_by_default(g2p):
    assert g2p.phonemes("hello world")[0] != SIL
    assert g2p.phonemes("hello world", leading_silence=True)[0] == SIL


def test_internal_punctuation_becomes_silence(g2p):
    phones = g2p.phonemes("hello, world")
    assert SIL in phones[:-1], "the comma should produce a silence before the end"


def test_silences_are_never_doubled(g2p):
    phones = g2p.phonemes("hello, world.")
    assert not any(a == SIL and b == SIL for a, b in zip(phones, phones[1:]))


def test_trailing_silence_can_be_switched_off(g2p):
    assert g2p.phonemes("hello world", trailing_silence=False)[-1] != SIL


def test_numbers_are_spoken(g2p):
    assert g2p.phonemes("two")[:2] == ["T", "UW1"]


def test_empty_text_raises(g2p):
    with pytest.raises(ValueError, match="no pronounceable words"):
        g2p.phonemes("...")


def test_symbols_outside_the_vocabulary_are_rejected(g2p):
    g2p.corpus = {"hello": ["NOT_A_PHONE"]}
    with pytest.raises(ValueError, match="outside the frozen vocabulary"):
        g2p.phonemes("hello")


def test_encode_returns_embedding_ids(g2p):
    ids = g2p.encode("hello world")
    phones = g2p.phonemes("hello world")
    assert ids == g2p.vocab.encode(phones)
    assert all(isinstance(i, int) and i > 0 for i in ids)


def test_punctuation_only_differences_do_not_change_the_words(g2p):
    a = [p for p in g2p.phonemes("hello world") if p != SIL]
    b = [p for p in g2p.phonemes("hello, world!") if p != SIL]
    assert a == b
