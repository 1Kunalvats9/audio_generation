"""M3 step 1: the feature builder."""

import numpy as np
import pandas as pd
import pytest

from src.baseline.duration_features import (
    BOS,
    CATEGORICAL,
    EOS,
    FEATURES,
    NUMERIC,
    TARGET,
    add_context,
    build_features,
    phone_class,
)
from src.config import SIL
from src.data.textgrid_to_phones import check
from tests.fake_phones import clip_totals, fake_phones


def test_the_fake_satisfies_m2s_own_invariants():
    phones = fake_phones(3)
    check(phones, clip_totals(phones))


def test_feature_list_is_the_two_groups():
    assert FEATURES == CATEGORICAL + NUMERIC
    assert TARGET == "n_frames"
    assert not set(CATEGORICAL) & set(NUMERIC)


@pytest.mark.parametrize("phone,expected", [
    (SIL, "silence"),
    ("AH1", "vowel"), ("IY0", "vowel"), ("EY2", "vowel"),
    ("S", "fricative"), ("DH", "fricative"), ("HH", "fricative"),
    ("P", "stop"), ("G", "stop"),
    ("M", "nasal"), ("NG", "nasal"),
    ("L", "other"), ("W", "other"), ("R", "other"),
])
def test_phone_class(phone, expected):
    assert phone_class(pd.Series([phone]))[0] == expected


def test_context_does_not_leak_across_clips():
    df = add_context(fake_phones(4))
    first = df.groupby("clip_id").head(1)
    last = df.groupby("clip_id").tail(1)
    assert (first["prev_phone"] == BOS).all()
    assert (last["next_phone"] == EOS).all()
    # and the neighbour really is the neighbour inside a clip
    one = df[df["clip_id"] == df["clip_id"].iloc[0]].reset_index(drop=True)
    assert list(one["prev_phone"])[1:] == list(one["phone"])[:-1]
    assert list(one["next_phone"])[:-1] == list(one["phone"])[1:]


def test_distance_to_silence():
    df = add_context(fake_phones(1))
    # the fake clip is 4 phones, then sil at idx 4, then 5 more
    assert list(df["dist_sil_left"]) == [1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    assert list(df["dist_sil_right"]) == [4, 3, 2, 1, 0, 5, 4, 3, 2, 1]


def test_silence_is_zero_and_speech_is_at_least_one():
    df = add_context(fake_phones(5))
    sil = df["phone"] == SIL
    assert (df.loc[sil, ["dist_sil_left", "dist_sil_right"]] == 0).all().all()
    assert (df.loc[~sil, ["dist_sil_left", "dist_sil_right"]] >= 1).all().all()


def test_silence_sentinels_do_not_reach_the_features():
    X, _ = build_features(fake_phones(3))
    # word_idx is -1 on silence in phones.csv; the feature must not be negative
    assert (X["word_idx_filled"] >= 0).all()
    assert (X["n_words"] > 0).all()
    assert (X["rel_word"].between(0, 1)).all()


def test_n_words_counts_real_words_not_the_silence_bucket():
    df = add_context(fake_phones(1))
    # the fake clip has words 0..3 plus one silence row
    assert df["n_words"].unique().tolist() == [4]


def test_no_missing_values_and_right_shape():
    phones = fake_phones(6)
    X, y = build_features(phones)
    assert len(X) == len(phones) == len(y)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert (y > 0).all()


def test_numerics_are_numeric_and_categoricals_are_strings():
    X, _ = build_features(fake_phones(2))
    for c in NUMERIC:
        assert X[c].dtype == np.float64, c
    for c in CATEGORICAL:
        assert X[c].map(type).eq(str).all(), c


def test_row_order_is_clip_then_idx():
    shuffled = fake_phones(4).sample(frac=1.0, random_state=1)
    df = add_context(shuffled)
    assert df[["clip_id", "idx"]].equals(
        df[["clip_id", "idx"]].sort_values(["clip_id", "idx"]).reset_index(drop=True)
    )


def test_no_feature_is_built_from_the_clips_true_length():
    """The leak guard: scaling every duration must not change any feature."""
    phones = fake_phones(3)
    X_before, _ = build_features(phones)
    stretched = phones.copy()
    stretched["n_frames"] = stretched["n_frames"] * 3
    stretched["start_frame"] = (
        stretched.groupby("clip_id")["n_frames"].cumsum() - stretched["n_frames"]
    )
    X_after, _ = build_features(stretched)
    pd.testing.assert_frame_equal(X_before, X_after)
