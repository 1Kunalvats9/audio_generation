"""Features for the phone-duration model  (M3 — step 1).

Turns M2's phones.csv rows into a feature table for the duration models. Every
feature here is derivable from the *text* alone: nothing is computed from the
clip's true total length. That matters twice over —

  * at synthesis time all we have is the phone sequence, so a feature built from
    the real duration would be a leak, and
  * A2 rescales predictions to the clip's true n_frames anyway (see
    adjust_durations in src/audio.py), so a clip-level tempo term would be
    divided straight back out. The model only has to get durations right
    *relative to each other* within a clip.

Silence rows carry sentinels in the word columns, which the derived features
have to work around:

    phone  word  word_idx  idx_in_word  word_len
    sil    sil   -1        0            1

So word_idx is forward-filled into word_idx_filled, n_words counts real words
only (max + 1, not nunique), and is_word_start/is_word_end are left to be
trivially true on silence — is_sil carries that signal instead.

Usage (as a library):
    from src.baseline.duration_features import add_context, build_features
    X, y = build_features(phones)

Requires (from M2):
    data/cache/phones.csv  — columns: clip_id, idx, phone, start_frame, n_frames,
                              word, word_idx, idx_in_word, word_len
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SIL

# Phone classes, matching M2's notebooks/02_alignment.ipynb so the labels here
# line up with the committed outputs/figures/alignment_energy_by_class.png.
FRICATIVES = {"S", "Z", "SH", "ZH", "F", "V", "TH", "DH", "HH"}
STOPS = {"P", "B", "T", "D", "K", "G"}
NASALS = {"M", "N", "NG"}

BOS, EOS = "<bos>", "<eos>"

CATEGORICAL = ["phone", "prev_phone", "next_phone", "cls", "stress"]
NUMERIC = [
    "idx_in_word", "word_len", "rel_in_word", "is_word_start", "is_word_end",
    "word_idx_filled", "n_words", "rel_word",
    "idx", "n_phones", "rel_phone",
    "is_sil", "prev_is_sil", "next_is_sil",
    "dist_sil_left", "dist_sil_right",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "n_frames"


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def phone_class(phone: pd.Series) -> pd.Series:
    """silence / vowel / fricative / stop / nasal / other."""
    base = phone.str.rstrip("012")
    return pd.Series(
        np.select(
            [
                phone.eq(SIL),
                phone.str[-1].str.isdigit(),
                base.isin(FRICATIVES),
                base.isin(STOPS),
                base.isin(NASALS),
            ],
            ["silence", "vowel", "fricative", "stop", "nasal"],
            default="other",
        ),
        index=phone.index,
    )


def _dist_to_sil(df: pd.DataFrame) -> pd.Series:
    """Phones since the previous `sil`, counting the clip start as a boundary.

    A silence row itself is 0; the phone immediately after a silence is 1. A
    clip that does not open with silence still starts at 1, so every non-silence
    phone is >= 1.
    """
    blk = df["phone"].eq(SIL).groupby(df["clip_id"]).cumsum()
    d = df.groupby([df["clip_id"], blk]).cumcount()
    return d.where(blk > 0, d + 1)


def add_context(phones: pd.DataFrame) -> pd.DataFrame:
    """phones.csv rows -> the same rows plus the derived feature columns."""
    df = phones.sort_values(["clip_id", "idx"]).reset_index(drop=True).copy()

    # Neighbours, grouped so context never leaks across a clip boundary.
    g = df.groupby("clip_id")["phone"]
    df["prev_phone"] = g.shift(1).fillna(BOS)
    df["next_phone"] = g.shift(-1).fillna(EOS)

    df["cls"] = phone_class(df["phone"])
    last = df["phone"].str[-1]
    df["stress"] = last.where(last.str.isdigit(), "none")

    df["is_sil"] = df["phone"].eq(SIL).astype(int)
    df["prev_is_sil"] = df["prev_phone"].eq(SIL).astype(int)
    df["next_is_sil"] = df["next_phone"].eq(SIL).astype(int)

    # Position in the word. word_len is 1 on silence, hence the max(.., 1).
    df["rel_in_word"] = df["idx_in_word"] / np.maximum(df["word_len"] - 1, 1)
    df["is_word_start"] = df["idx_in_word"].eq(0).astype(int)
    df["is_word_end"] = df["idx_in_word"].eq(df["word_len"] - 1).astype(int)

    # Position in the clip, working around word_idx == -1 on silence.
    real = df["word_idx"].ge(0)
    df["word_idx_filled"] = (
        df["word_idx"].where(real).groupby(df["clip_id"]).ffill().fillna(0).astype(int)
    )
    df["n_words"] = df.groupby("clip_id")["word_idx"].transform("max") + 1
    df["rel_word"] = df["word_idx_filled"] / np.maximum(df["n_words"] - 1, 1)

    df["n_phones"] = df.groupby("clip_id")["idx"].transform("size")
    df["rel_phone"] = df["idx"] / np.maximum(df["n_phones"] - 1, 1)

    df["dist_sil_left"] = _dist_to_sil(df)
    df["dist_sil_right"] = _dist_to_sil(df.iloc[::-1]).iloc[::-1]

    return df


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------


def features_from_context(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A frame already through add_context -> (X, y)."""
    missing = [c for c in FEATURES if c not in df.columns]
    assert not missing, f"add_context did not produce: {missing}"

    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    for c in NUMERIC:
        X[c] = X[c].astype(np.float64)

    y = df[TARGET].astype(np.float64)
    return X, y


def build_features(phones: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """phones.csv rows -> (X, y). Convenience wrapper over add_context."""
    return features_from_context(add_context(phones))
