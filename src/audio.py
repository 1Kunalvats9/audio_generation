"""Audio I/O and mel spectrogram utilities  (M4).

Four public functions every other module imports:

    from src.audio import load_wav, save_wav, wav_to_mel, mel_to_wav

Plus one duration helper shared with M3:

    from src.audio import adjust_durations

Run from the repo root only.  Never import torchaudio.load — use load_wav.

Vocoder note (verified 2026-10-27):
  speechbrain/tts-hifigan-ljspeech with torch 2.14.1 / torchaudio 2.11.0 /
  speechbrain 1.1.2 loads and vocodes correctly.  No downgrade needed.
  The vocoder adds inference_padding=5, so a mel of T frames returns
  (T + 10) * HOP_LENGTH samples.  mel_to_wav trims to T * HOP_LENGTH.
  Pass n_samples to trim_wav() afterwards if you need the exact original length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import MelSpectrogram

from src.config import HOP_LENGTH, LOG_CLAMP_MIN, MEL_KWARGS, SAMPLE_RATE
from src.config import VOCODER_SOURCE, VOCODER_SAVEDIR

# ---------------------------------------------------------------------------
# Module-level vocoder cache — loaded once, reused for every clip.
# ---------------------------------------------------------------------------

_vocoder = None


def _get_vocoder():
    """Lazy-load HiFi-GAN (downloads ~55 MB on first call)."""
    global _vocoder
    if _vocoder is not None:
        return _vocoder

    # speechbrain 1.1.x moved the import path; try both.
    try:
        from speechbrain.inference.TTS import HIFIGAN
    except ImportError:
        from speechbrain.pretrained import HIFIGAN  # type: ignore[no-redef]

    _vocoder = HIFIGAN.from_hparams(
        source=VOCODER_SOURCE,
        savedir=str(VOCODER_SAVEDIR),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return _vocoder


# ---------------------------------------------------------------------------
# The four public audio functions
# ---------------------------------------------------------------------------


def load_wav(path: Union[str, Path]) -> np.ndarray:
    """Read a wav file as a float32 mono array at 22,050 Hz.

    Uses soundfile (never torchaudio.load) as per team rules.
    Asserts sample rate and mono — raises AssertionError on mismatch.
    """
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, (
        f"{path}: sample rate {sr} Hz, expected {SAMPLE_RATE}"
    )
    assert wav.ndim == 1, (
        f"{path}: {wav.ndim} channels, expected mono"
    )
    return wav


def save_wav(path: Union[str, Path], wav: np.ndarray) -> None:
    """Write a float32 mono array to a wav file at 22,050 Hz.

    Uses PCM_16 subtype (signed 16-bit) to match the LJSpeech source format.
    This introduces a maximum quantisation error of 1 LSB ≈ 3.05e-5 on round-trip
    (2 / 2**16).  Tests that check round-trip accuracy must allow atol >= 4e-5.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, SAMPLE_RATE, subtype="PCM_16")


def wav_to_mel(wav: np.ndarray) -> torch.Tensor:
    """Convert a float32 mono waveform to a log-mel spectrogram.

    Returns a float32 tensor of shape (80, T) — the same recipe used by
    speechbrain/tts-hifigan-ljspeech (verified MAE 0.000000).

        mel = torch.log(MelSpectrogram(**MEL_KWARGS)(wav).clamp(min=1e-5))

    Values land in roughly −11.5 … +1.5.  That is correct.
    """
    t = torch.from_numpy(wav)             # (N,)
    transform = MelSpectrogram(**MEL_KWARGS)
    mel = transform(t)                    # (80, T)
    return torch.log(mel.clamp(min=LOG_CLAMP_MIN))


def mel_to_wav(mel: torch.Tensor) -> np.ndarray:
    """Decode a log-mel spectrogram to a waveform via HiFi-GAN.

    Args:
        mel: float32 tensor, shape (80, T).

    Returns:
        float32 numpy array of length T * HOP_LENGTH (vocoder padding trimmed).

    The vocoder's inference_padding=5 adds 10 extra frames, so the raw output
    is (T + 10) * HOP_LENGTH samples.  We trim here so callers never see junk.
    If you need exactly n_samples from the original clip, call trim_wav() after.
    """
    vocoder = _get_vocoder()
    T = mel.shape[-1]
    device = next(vocoder.parameters()).device if hasattr(vocoder, "parameters") else "cpu"

    # decode_batch expects (batch, n_mels, time) on the vocoder's device.
    mel_batch = mel.unsqueeze(0).to(device)          # (1, 80, T)
    with torch.no_grad():
        wav_batch = vocoder.decode_batch(mel_batch)  # (1, 1, T_out) or (1, T_out)

    wav = wav_batch.squeeze().cpu().numpy()          # (T_out,)

    # Trim the vocoder padding: keep only T * HOP_LENGTH samples.
    expected = T * HOP_LENGTH
    return wav[:expected].astype(np.float32)


def trim_wav(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad a waveform to exactly n_samples.

    Use this after mel_to_wav() when you need the original clip length.
    """
    if len(wav) >= n_samples:
        return wav[:n_samples]
    return np.pad(wav, (0, n_samples - len(wav)))


# ---------------------------------------------------------------------------
# Duration adjustment helper (shared with M3)
# ---------------------------------------------------------------------------


def adjust_durations(
    pred_frames: np.ndarray,
    target: int,
) -> np.ndarray:
    """Round float duration predictions to positive integers summing to target.

    Algorithm (from TEAM.md M3 §5):
      1. Round to nearest integer.
      2. Clamp every phone to ≥ 1 frame.
      3. Fix the rounding error by adding/subtracting the residual from the
         phone with the largest duration (least relative distortion).

    Args:
        pred_frames: float array of predicted frame counts, one per phone.
        target:      the clip's total n_frames (from metadata_clean.csv).

    Returns:
        int32 array, same length as pred_frames, all ≥ 1, sum == target.
    """
    d = np.maximum(1, np.round(pred_frames)).astype(np.int32)
    diff = int(target - d.sum())

    if diff != 0:
        # Apply the residual to the longest phone (minimises relative distortion).
        idx = int(np.argmax(d))
        d[idx] += diff
        # Safety: clamp again in case subtracting made it < 1.
        if d[idx] < 1:
            d[idx] = 1
            # Re-distribute remaining deficit across next-largest phones.
            remaining = target - d.sum()
            sorted_idx = np.argsort(d)[::-1]
            for i in sorted_idx:
                if remaining == 0:
                    break
                add = min(remaining, d[i] - 1) if remaining < 0 else remaining
                d[i] += add
                remaining -= add

    assert d.sum() == target, (
        f"adjust_durations: sum {d.sum()} != target {target}"
    )
    assert (d >= 1).all(), "adjust_durations: phone with 0 frames"
    return d
