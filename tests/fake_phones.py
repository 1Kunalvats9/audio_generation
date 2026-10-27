"""A hand-made phones.csv standing in for M2's output, for the M3 unit tests.

Satisfies the same invariants src/data/textgrid_to_phones.py:check() enforces on
the real file: idx runs 0..n-1 per clip, phones tile the clip with no gaps, and
silence rows carry the word_idx=-1 / word_len=1 sentinels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.textgrid_to_phones import COLUMNS

# (phone, n_frames, word, word_idx, idx_in_word, word_len)
_CLIP = [
    ("P",   4, "pril", 0, 0, 4),
    ("R",   3, "pril", 0, 1, 4),
    ("IH1", 6, "pril", 0, 2, 4),
    ("L",   5, "pril", 0, 3, 4),
    ("sil", 10, "sil", -1, 0, 1),
    ("N",   5, "no",  1, 0, 2),
    ("OW1", 9, "no",  1, 1, 2),
    ("S",   7, "see", 2, 0, 2),
    ("IY0", 8, "see", 2, 1, 2),
    ("NG",  6, "ng",  3, 0, 1),
]


def fake_phones(n_clips: int = 2, seed: int = 0) -> pd.DataFrame:
    """n_clips clips built from the same phone sequence with jittered durations."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clips):
        clip_id = f"LJ{c // 100:03d}-{c % 100:04d}"
        start = 0
        for idx, (phone, frames, word, word_idx, idx_in_word, word_len) in enumerate(_CLIP):
            n = int(max(1, frames + rng.integers(-2, 3)))
            rows.append((clip_id, idx, phone, start, n, word, word_idx, idx_in_word, word_len))
            start += n
    return pd.DataFrame(rows, columns=COLUMNS)


def clip_totals(phones: pd.DataFrame) -> dict[str, int]:
    """The n_frames-per-clip map that textgrid_to_phones.check() wants."""
    return phones.groupby("clip_id")["n_frames"].sum().to_dict()
