"""Contract tests for the phase 2 modules that are not written yet.

Each block is skipped until its module exists, then activates automatically. So
whoever writes the module knows exactly what "done" looks like before they
start, and the pieces fit together the first time they are composed.

  src/dl/duration_predictor.py   M3 — Pushkar + Kunal
  src/dl/decoder.py              M4 — Shaurya
  src/eval/mcd.py                M5 — Chaitanya

Run `pytest tests/test_dl_contracts.py -rs` to see which are still pending.
"""

import importlib.util

import numpy as np
import pytest
import torch

from src.config import N_MELS
from src.dl.config import HIDDEN

have = lambda mod: importlib.util.find_spec(mod) is not None

needs_duration = pytest.mark.skipif(
    not have("src.dl.duration_predictor"), reason="M3 has not written src/dl/duration_predictor.py")
needs_decoder = pytest.mark.skipif(
    not have("src.dl.decoder"), reason="M4 has not written src/dl/decoder.py")
needs_mcd = pytest.mark.skipif(
    not have("src.eval.mcd"), reason="M5 has not written src/eval/mcd.py")


# ---------------------------------------------------------------------------
# M3 — duration predictor
#
#   class DurationPredictor(nn.Module):
#       def forward(self, enc_out: (B, P, H), phone_mask: (B, P)) -> (B, P)
#           returns predicted LOG durations
#   def duration_loss(pred_log: (B, P), target: (B, P), mask: (B, P)) -> scalar
# ---------------------------------------------------------------------------


@needs_duration
def test_duration_predictor_shape_and_masking():
    from src.dl.duration_predictor import DurationPredictor
    model = DurationPredictor(HIDDEN).eval()
    x = torch.randn(2, 5, HIDDEN)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (2, 5), "predict one log-duration per phoneme"
    assert torch.all(out[~mask] == 0), "padded phonemes must predict exactly 0"


@needs_duration
def test_duration_predictor_is_batch_independent():
    """Same reasoning as the encoder: convolutions must not see padding."""
    from src.dl.duration_predictor import DurationPredictor
    torch.manual_seed(0)
    model = DurationPredictor(HIDDEN).eval()
    x = torch.randn(1, 3, HIDDEN)
    short = model(x, torch.ones(1, 3, dtype=torch.bool))
    padded = model(torch.cat([x, torch.randn(1, 4, HIDDEN)], dim=1),
                   torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool))[:, :3]
    assert torch.allclose(short, padded, atol=1e-6)


@needs_duration
def test_duration_loss_ignores_padding_and_is_zero_when_perfect():
    from src.dl.duration_predictor import duration_loss
    target = torch.tensor([[4.0, 9.0, 0.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    perfect = torch.log(target.clamp(min=1e-5))
    assert duration_loss(perfect, target, mask).item() == pytest.approx(0.0, abs=1e-5)
    # junk in the padded slot must not change the loss
    noisy = perfect.clone()
    noisy[0, 2] = 123.0
    assert duration_loss(noisy, target, mask).item() == pytest.approx(0.0, abs=1e-5)


@needs_duration
def test_duration_predictor_gradients_flow():
    from src.dl.duration_predictor import DurationPredictor
    model = DurationPredictor(HIDDEN)
    x = torch.randn(1, 4, HIDDEN, requires_grad=True)
    model(x, torch.ones(1, 4, dtype=torch.bool)).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# M4 — mel decoder (D1)
#
#   class MelDecoder(nn.Module):
#       def forward(self, frames: (B, T, H), frame_mask: (B, T)) -> (B, 80, T)
#   def mel_loss(pred: (B, 80, T), target: (B, 80, T), frame_mask: (B, T)) -> scalar
# ---------------------------------------------------------------------------


@needs_decoder
def test_decoder_returns_a_mel_of_the_right_shape():
    from src.dl.decoder import MelDecoder
    model = MelDecoder(HIDDEN).eval()
    frames = torch.randn(2, 17, HIDDEN)
    mask = torch.ones(2, 17, dtype=torch.bool)
    out = model(frames, mask)
    assert out.shape == (2, N_MELS, 17), "channels-first mel, (B, 80, T)"


@needs_decoder
def test_decoder_zeroes_padded_frames():
    from src.dl.decoder import MelDecoder
    model = MelDecoder(HIDDEN).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    out = model(torch.randn(1, 5, HIDDEN), mask)
    assert torch.all(out[0, :, 3:] == 0)


@needs_decoder
def test_decoder_is_batch_independent():
    from src.dl.decoder import MelDecoder
    torch.manual_seed(0)
    model = MelDecoder(HIDDEN).eval()
    f = torch.randn(1, 6, HIDDEN)
    a = model(f, torch.ones(1, 6, dtype=torch.bool))
    b = model(torch.cat([f, torch.randn(1, 5, HIDDEN)], dim=1),
              torch.tensor([[1] * 6 + [0] * 5], dtype=torch.bool))[:, :, :6]
    assert torch.allclose(a, b, atol=1e-6)


@needs_decoder
def test_mel_loss_is_zero_when_perfect_and_ignores_padding():
    from src.dl.decoder import mel_loss
    target = torch.randn(1, N_MELS, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    pred = target.clone()
    assert mel_loss(pred, target, mask).item() == pytest.approx(0.0, abs=1e-6)
    pred[:, :, 3:] += 50.0                      # garbage in padded frames
    assert mel_loss(pred, target, mask).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# M5 — mel cepstral distortion
#
#   def mel_cepstral_distortion(pred: (80, T), true: (80, T)) -> float   # dB
# ---------------------------------------------------------------------------


@needs_mcd
def test_mcd_of_identical_mels_is_zero():
    from src.eval.mcd import mel_cepstral_distortion
    mel = np.random.default_rng(0).normal(-5, 1.5, size=(N_MELS, 40)).astype(np.float32)
    assert mel_cepstral_distortion(mel, mel) == pytest.approx(0.0, abs=1e-6)


@needs_mcd
def test_mcd_is_positive_and_symmetric():
    from src.eval.mcd import mel_cepstral_distortion
    rng = np.random.default_rng(1)
    a = rng.normal(-5, 1.5, size=(N_MELS, 40)).astype(np.float32)
    b = rng.normal(-5, 1.5, size=(N_MELS, 40)).astype(np.float32)
    assert mel_cepstral_distortion(a, b) > 0
    assert mel_cepstral_distortion(a, b) == pytest.approx(mel_cepstral_distortion(b, a), rel=1e-6)


@needs_mcd
def test_mcd_rejects_a_length_mismatch():
    from src.eval.mcd import mel_cepstral_distortion
    a = np.zeros((N_MELS, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        mel_cepstral_distortion(a, np.zeros((N_MELS, 11), dtype=np.float32))
