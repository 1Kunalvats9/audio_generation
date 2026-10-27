"""Tests for src/data/build_lexicon.py. Run: python -m pytest tests

Guards the hand-written pronunciations: a typo there makes MFA reject the
whole dictionary, or silently align the wrong sounds.
"""

import pytest

from src.data.build_lexicon import ACRONYMS, LETTERS, OOV_DICT, OVERRIDES, read_dict, spell

CONSONANTS = {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
              "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ARPABET = CONSONANTS | {v + s for v in VOWELS for s in "012"}


def is_valid(pron: str) -> bool:
    phones = pron.split()
    return bool(phones) and set(phones) <= ARPABET and any(p[-1].isdigit() for p in phones)


@pytest.mark.parametrize("word, expected", [
    ("fbi", "EH1 F B IY1 AY1"),
    ("uv", "Y UW1 V IY1"),
    ("wdsu-tv", spell("wdsu") + " T IY1 V IY1"),     # the hyphen is not a letter
])
def test_spell(word, expected):
    assert spell(word) == expected


def test_every_letter_and_override_is_valid_arpabet():
    bad = {w: p for w, p in {**LETTERS, **OVERRIDES}.items() if not is_valid(p)}
    assert not bad


def test_overrides_have_primary_stress():
    assert all("1" in p for p in OVERRIDES.values())


def test_acronyms_and_overrides_do_not_overlap():
    assert not set(ACRONYMS) & set(OVERRIDES)


def test_read_dict_handles_both_formats(tmp_path):
    path = tmp_path / "x.dict"
    path.write_text("cia\tS IY1 AY1 EY1\n"
                    "the\t0.99\t0.1\t1.0\t1.0\tDH AH0\n"
                    "the\t0.5\t0.1\t1.0\t1.0\tDH IY0\n")      # second pronunciation ignored
    assert read_dict(path) == {"cia": "S IY1 AY1 EY1", "the": "DH AH0"}


def test_committed_oov_dict():
    prons = read_dict(OOV_DICT)
    assert len(prons) > 300
    bad = {w: p for w, p in prons.items() if not is_valid(p)}
    assert not bad
    for word in ACRONYMS:
        assert prons[word] == spell(word)
    for word, pron in OVERRIDES.items():
        assert prons[word] == pron
