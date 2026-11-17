"""D1 padding, loss scaling and optimisation regressions."""

import pytest
import torch

from src.config import N_MELS
from src.dl.decoder import MelDecoder, mel_loss


def test_loss_averages_only_real_elements():
    pred = torch.zeros(2, N_MELS, 4, requires_grad=True)
    target = torch.ones_like(pred)
    mask = torch.tensor([[True] * 4, [True, False, False, False]])
    target[1, :, 1:] = float("nan")
    loss = mel_loss(pred, target, mask)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[1, :, 1:] == 0).all()


def test_all_padding_has_zero_differentiable_loss():
    pred = torch.randn(1, N_MELS, 3, requires_grad=True)
    loss = mel_loss(pred, pred.detach(), torch.zeros(1, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert (pred.grad == 0).all()


def test_decoder_learns_and_backpropagates_to_conditioning():
    torch.manual_seed(42)
    model = MelDecoder(hidden=16, layers=2, kernel=3, dropout=0)
    frames = torch.randn(2, 8, 16, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    target = torch.randn(2, N_MELS, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    before = mel_loss(model(frames, mask), target, mask).item()
    for _ in range(40):
        opt.zero_grad()
        loss = mel_loss(model(frames, mask), target, mask)
        loss.backward()
        opt.step()
    assert loss.item() < before * 0.5
    assert frames.grad.abs().sum() > 0


def test_decoder_ignores_nonfinite_padded_inputs():
    model = MelDecoder(hidden=16, layers=2, kernel=3, dropout=0).eval()
    frames = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, False, False]])
    frames[:, 2:] = float("nan")
    assert torch.isfinite(model(frames, mask)).all()
