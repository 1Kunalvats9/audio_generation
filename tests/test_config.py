"""Guards the shared contract in configs/base.yaml.

If someone changes a number here, mels, durations and audio stop lining up.
These tests fail loudly instead of letting the mismatch reach the results.
"""

import pytest

from src.config import (
    HOP_LENGTH,
    LOG_CLAMP_MIN,
    MEL_KWARGS,
    N_MELS,
    SAMPLE_RATE,
    SECONDS_PER_FRAME,
    SIL,
    frame_to_sec,
    n_frames,
    sec_to_frame,
)


def test_frozen_audio_settings():
    assert SAMPLE_RATE == 22050
    assert HOP_LENGTH == 256
    assert N_MELS == 80
    assert LOG_CLAMP_MIN == pytest.approx(1e-5)
    assert SIL == "sil"
    assert SECONDS_PER_FRAME == pytest.approx(0.011610, abs=1e-6)


def test_mel_recipe_matches_the_vocoder():
    """These exact values are what speechbrain/tts-hifigan-ljspeech was trained on."""
    assert MEL_KWARGS["power"] == 1          # amplitude, not power
    assert MEL_KWARGS["norm"] == "slaney"
    assert MEL_KWARGS["mel_scale"] == "slaney"
    assert MEL_KWARGS["center"] is True
    assert MEL_KWARGS["f_min"] == 0
    assert MEL_KWARGS["f_max"] == 8000
    assert MEL_KWARGS["n_fft"] == MEL_KWARGS["win_length"] == 1024


@pytest.mark.parametrize("n_samples, expected", [
    (0, 1),
    (255, 1),
    (256, 2),
    (22050, 87),       # one second
    (212893, 832),     # LJ001-0001
])
def test_n_frames(n_samples, expected):
    assert n_frames(n_samples) == expected


def test_sec_to_frame_round_trips():
    for idx in [0, 1, 85, 86, 500, 869]:
        assert sec_to_frame(frame_to_sec(idx)) == idx


def test_sec_to_frame_is_a_round_not_a_floor():
    # 1.0 s is 86.13 frames -> 86; the floor convention would also give 86,
    # but 0.0122 s is 1.05 frames -> 1, where floor and round differ at 1.5
    assert sec_to_frame(1.0) == 86
    assert sec_to_frame(frame_to_sec(2) - 1e-9) == 2
