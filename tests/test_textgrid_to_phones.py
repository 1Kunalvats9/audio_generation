"""Tests for src/data/textgrid_to_phones.py. Run: python -m pytest tests

Builds small MFA-shaped TextGrids with praatio, so no dataset or MFA is needed.
"""

import pytest
from praatio import textgrid

from src.config import SIL, frame_to_sec
from src.data.textgrid_to_phones import (
    COLUMNS,
    Excluded,
    assign_words,
    clip_rows,
    fill_silence,
    fix_zero_frames,
    frame_durations,
)


def f(k):
    """Frame index -> seconds, so test boundaries land exactly on frames."""
    return frame_to_sec(k)


def write_textgrid(path, words, phones, xmax, prefix=""):
    """MFA style: silence is left blank, so includeBlankSpaces fills it with ""."""
    tg = textgrid.Textgrid()
    tg.addTier(textgrid.IntervalTier(f"{prefix}words", words, 0, xmax))
    tg.addTier(textgrid.IntervalTier(f"{prefix}phones", phones, 0, xmax))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    return path


# "the cat": 10 frames silence, DH AH0 | K AE1 T, 12 frames silence -> 50 frames
WORDS = [(f(10), f(18), "the"), (f(18), f(38), "cat")]
PHONES = [(f(10), f(13), "DH"), (f(13), f(18), "AH0"),
          (f(18), f(24), "K"), (f(24), f(33), "AE1"), (f(33), f(38), "T")]
XMAX = f(50)


# ---------- silence ----------

def test_fill_silence_fills_gaps_maps_labels_and_merges():
    out = fill_silence([(0.1, 0.2, "DH"), (0.2, 0.3, "AH0"), (0.35, 0.5, "sp"), (0.5, 0.6, "K")],
                       0.0, 0.8)
    assert out == [(0.0, 0.1, SIL), (0.1, 0.2, "DH"), (0.2, 0.3, "AH0"),
                   (0.3, 0.5, SIL), (0.5, 0.6, "K"), (0.6, 0.8, SIL)]


def test_fill_silence_snaps_float_noise():
    out = fill_silence([(0.0, 0.1, "A"), (0.1000001, 0.2, "B")], 0.0, 0.2)
    assert out == [(0.0, 0.1, "A"), (0.1, 0.2, "B")]


# ---------- frames ----------

def test_frame_durations_exact():
    assert frame_durations([f(0), f(3), f(10)], 10) == [3, 7]


@pytest.mark.parametrize("target, last", [(11, 8), (12, 9), (9, 6), (8, 5)])
def test_last_phone_absorbs_up_to_two_frames(target, last):
    dur = frame_durations([f(0), f(3), f(10)], target)
    assert dur == [3, last]
    assert sum(dur) == target


@pytest.mark.parametrize("target", [13, 7])
def test_mismatch_over_two_frames_is_excluded(target):
    with pytest.raises(Excluded, match="length_mismatch"):
        frame_durations([f(0), f(3), f(10)], target)


@pytest.mark.parametrize("dur, expected", [
    ([3, 0, 5], [3, 1, 4]),          # borrows from the longer neighbour
    ([5, 0, 3], [4, 1, 3]),
    ([0, 4], [1, 3]),                # first phone
    ([4, 0], [3, 1]),                # last phone
    ([5, 0, 0, 5], [4, 1, 1, 4]),    # two in a row
])
def test_fix_zero_frames(dur, expected):
    out = fix_zero_frames(dur)
    assert out == expected
    assert sum(out) == sum(dur)


def test_zero_frame_without_a_lender_is_excluded():
    with pytest.raises(Excluded, match="zero_frame"):
        fix_zero_frames([1, 0, 1])


# ---------- words ----------

def test_assign_words():
    phones = fill_silence(PHONES, 0.0, XMAX)
    out = assign_words(phones, WORDS)
    assert out == [(SIL, -1, 0, 1),
                   ("the", 0, 0, 2), ("the", 0, 1, 2),
                   ("cat", 1, 0, 3), ("cat", 1, 1, 3), ("cat", 1, 2, 3),
                   (SIL, -1, 0, 1)]


# ---------- one whole clip ----------

def test_clip_rows(tmp_path):
    path = write_textgrid(tmp_path / "LJ000-0001.TextGrid", WORDS, PHONES, XMAX)
    rows = clip_rows("LJ000-0001", path, target=51)     # n_frames is 1 + n_samples // 256
    as_dicts = [dict(zip(COLUMNS, r)) for r in rows]

    assert [r["phone"] for r in as_dicts] == [SIL, "DH", "AH0", "K", "AE1", "T", SIL]
    assert [r["n_frames"] for r in as_dicts] == [10, 3, 5, 6, 9, 5, 13]
    assert [r["start_frame"] for r in as_dicts] == [0, 10, 13, 18, 24, 33, 38]
    assert [r["idx"] for r in as_dicts] == list(range(7))
    assert sum(r["n_frames"] for r in as_dicts) == 51
    assert as_dicts[4] == dict(clip_id="LJ000-0001", idx=4, phone="AE1", start_frame=24,
                               n_frames=9, word="cat", word_idx=1, idx_in_word=1, word_len=3)


def test_the_word_silence_is_a_word(tmp_path):
    """Regression: "silence" was once treated as a pause label, orphaning its phones."""
    words = [(f(10), f(30), "silence")]
    phones = [(f(10), f(13), "S"), (f(13), f(18), "AY1"), (f(18), f(21), "L"),
              (f(21), f(24), "AH0"), (f(24), f(27), "N"), (f(27), f(30), "S")]
    path = write_textgrid(tmp_path / "d.TextGrid", words, phones, f(40))
    rows = [dict(zip(COLUMNS, r)) for r in clip_rows("d", path, target=40)]
    assert [r["word"] for r in rows] == [SIL] + ["silence"] * 6 + [SIL]
    assert [r["idx_in_word"] for r in rows[1:7]] == list(range(6))


def test_speaker_prefixed_tier_names(tmp_path):
    path = write_textgrid(tmp_path / "a.TextGrid", WORDS, PHONES, XMAX, prefix="LJ - ")
    assert len(clip_rows("a", path, target=50)) == 7


def test_spn_is_excluded(tmp_path):
    words = WORDS[:1] + [(f(18), f(38), "<unk>")]
    phones = PHONES[:2] + [(f(18), f(38), "spn")]
    path = write_textgrid(tmp_path / "b.TextGrid", words, phones, XMAX)
    with pytest.raises(Excluded, match="spn"):
        clip_rows("b", path, target=50)


def test_missing_tier_is_bad_textgrid(tmp_path):
    tg = textgrid.Textgrid()
    tg.addTier(textgrid.IntervalTier("words", WORDS, 0, XMAX))
    tg.save(str(tmp_path / "c.TextGrid"), format="long_textgrid", includeBlankSpaces=True)
    with pytest.raises(Excluded, match="bad_textgrid"):
        clip_rows("c", tmp_path / "c.TextGrid", target=50)
