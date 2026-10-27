"""Tests for src/audio.py  (M4).

Covers the four public functions and adjust_durations without needing
the actual dataset (mocked/synthetic data only).

Run: python -m pytest tests/test_audio.py
"""

import numpy as np
import pytest
import torch

from src.audio import adjust_durations, load_wav, save_wav, wav_to_mel
from src.config import HOP_LENGTH, N_MELS, SAMPLE_RATE


# ---------------------------------------------------------------------------
# wav_to_mel
# ---------------------------------------------------------------------------


def test_wav_to_mel_shape():
    """mel should be (80, T) where T == 1 + n_samples // hop_length."""
    n_samples = SAMPLE_RATE  # 1 second
    wav = np.zeros(n_samples, dtype=np.float32)
    mel = wav_to_mel(wav)
    expected_T = 1 + n_samples // HOP_LENGTH
    assert mel.shape == (N_MELS, expected_T), (
        f"expected ({N_MELS}, {expected_T}), got {mel.shape}"
    )


def test_wav_to_mel_dtype():
    wav = np.random.randn(1024).astype(np.float32)
    mel = wav_to_mel(wav)
    assert mel.dtype == torch.float32


def test_wav_to_mel_range():
    """Log-mel values should be finite and in the documented range."""
    wav = np.random.randn(22050).astype(np.float32)
    mel = wav_to_mel(wav)
    assert torch.isfinite(mel).all(), "mel contains inf/nan"
    # TEAM.md: values land in roughly -11.5 … +1.5
    assert mel.min().item() > -20.0, f"suspiciously low min: {mel.min()}"
    assert mel.max().item() < 5.0,   f"suspiciously high max: {mel.max()}"


def test_wav_to_mel_silent():
    """A silent clip gives the log-clamp floor, not -inf."""
    wav = np.zeros(22050, dtype=np.float32)
    mel = wav_to_mel(wav)
    assert torch.isfinite(mel).all()
    # All values should be log(1e-5) ≈ -11.51
    assert mel.max().item() < -11.0


# ---------------------------------------------------------------------------
# load_wav / save_wav round-trip
# ---------------------------------------------------------------------------


def test_load_save_round_trip(tmp_path):
    """save_wav followed by load_wav should recover the same signal.

    save_wav uses PCM_16 (signed 16-bit) to match the LJSpeech source format.
    The maximum quantisation error is 1 LSB = 2/2**16 ≈ 3.05e-5, so we allow
    atol=4e-5 (one LSB plus a small margin).
    """
    wav_orig = np.random.randn(22050).astype(np.float32) * 0.1
    path = tmp_path / "test.wav"
    save_wav(path, wav_orig)
    wav_loaded = load_wav(path)
    np.testing.assert_allclose(wav_orig, wav_loaded, atol=4e-5)


def test_load_wav_creates_parent(tmp_path):
    """save_wav should create missing parent directories."""
    deep_path = tmp_path / "a" / "b" / "c" / "out.wav"
    wav = np.zeros(1024, dtype=np.float32)
    save_wav(deep_path, wav)
    assert deep_path.exists()


# ---------------------------------------------------------------------------
# adjust_durations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preds, target", [
    ([5.4, 3.1, 2.9], 12),
    ([1.0, 1.0, 1.0], 5),   # needs rounding up
    ([10.0],         10),
    ([0.3, 0.3, 0.3], 3),   # all would clamp to 1
    # Over-prediction by more than the longest phone can give up, so the
    # deficit has to be spread. Shapes taken from the clips that broke A2:
    # LJ005-0225 (108 phones, rounded sum 835 vs target 745).
    ([25.0] + [8.0] * 107, 745),
    ([30.0, 20.0, 15.0] + [6.0] * 40, 120),
])
def test_adjust_durations_sum(preds, target):
    result = adjust_durations(np.array(preds, dtype=np.float32), target)
    assert result.sum() == target, f"sum {result.sum()} != {target}"


def test_adjust_durations_spreads_a_deficit_bigger_than_the_longest_phone():
    """Regression: the residual must be capped at each phone's headroom.

    Subtracting the whole deficit from the longest phone drove it negative, which
    broke 27 of the 299 test clips (8 of the 100 A2 synthesises).
    """
    preds = np.array([25.0] + [8.0] * 107, dtype=np.float32)
    result = adjust_durations(preds, target=745)
    assert result.sum() == 745
    assert (result >= 1).all()
    assert len(result) == len(preds)


@pytest.mark.parametrize("seed", range(25))
def test_adjust_durations_holds_its_contract_for_random_inputs(seed):
    """Any target >= n_phones must come back summing exactly, every phone >= 1."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 120))
    preds = rng.uniform(0.2, 30.0, size=n).astype(np.float32)
    target = int(rng.integers(n, int(preds.sum() * 1.6) + n + 1))
    result = adjust_durations(preds, target)
    assert result.sum() == target
    assert (result >= 1).all()
    assert len(result) == n


def test_adjust_durations_all_positive():
    preds = np.array([0.1, 0.2, 0.3, 100.0], dtype=np.float32)
    result = adjust_durations(preds, target=10)
    assert (result >= 1).all()


def test_adjust_durations_exact_already():
    preds = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    result = adjust_durations(preds, target=12)
    assert result.sum() == 12
    np.testing.assert_array_equal(result, [3, 4, 5])


def test_adjust_durations_dtype():
    preds = np.array([2.7, 3.1], dtype=np.float32)
    result = adjust_durations(preds, target=6)
    assert result.dtype == np.int32
