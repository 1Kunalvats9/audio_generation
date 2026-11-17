"""Cosine schedule, masked denoising, gradients and DDIM sampling."""
import pytest
import torch
from torch.nn import functional as F

from src.config import N_MELS
from src.dl.diffusion import MelDiffusion, cosine_betas
from src.dl.train import AcousticModel
from src.dl.dataset import collate


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_schedule_is_valid_and_decreasing_signal():
    betas = cosine_betas(1000)
    assert ((betas > 0) & (betas < 1)).all()
    alpha = (1 - betas).cumprod(0)
    assert (alpha[1:] < alpha[:-1]).all()
    assert alpha[-1] < 1e-6


def test_noise_loss_masks_nan_padding_and_has_gradients():
    model = MelDiffusion(hidden=16, train_steps=20, sample_steps=5)
    frames = torch.randn(2, 7, 16, requires_grad=True)
    mel = torch.randn(2, N_MELS, 7)
    mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
    mel[1, :, 3:] = float("nan")
    loss = model.loss(mel, frames, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert frames.grad.abs().sum() > 0
    assert (frames.grad[1, 3:] == 0).all()


def test_denoiser_and_sampler_are_padding_independent_for_odd_lengths():
    torch.manual_seed(42)
    model = MelDiffusion(hidden=16, train_steps=20, sample_steps=5).eval()
    frames = torch.randn(1, 5, 16)
    noise = torch.randn(1, N_MELS, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    padded_frames = F.pad(frames, (0, 0, 0, 6), value=123)
    padded_noise = F.pad(noise, (0, 6), value=float("nan"))
    padded_mask = F.pad(mask, (0, 6), value=False)
    t = torch.tensor([7])
    a = model.denoiser(noise, t, frames, mask)
    b = model.denoiser(padded_noise, t, padded_frames, padded_mask)
    assert torch.allclose(a, b[:, :, :5], atol=1e-6)
    short = model.sample(frames, mask, noise=noise)
    long = model.sample(padded_frames, padded_mask, noise=padded_noise)
    assert torch.allclose(short, long[:, :, :5], atol=1e-4, rtol=1e-5)
    assert (long[:, :, 5:] == 0).all()
    assert torch.isfinite(long).all()


def test_perfect_noise_predictor_recovers_clean_mel(monkeypatch):
    model = MelDiffusion(hidden=16, train_steps=20, sample_steps=5)
    frames = torch.randn(1, 5, 16)
    mel = torch.randn(1, N_MELS, 5)
    noise = torch.randn_like(mel)
    mask = torch.ones(1, 5, dtype=torch.bool)
    t = torch.tensor([10])
    monkeypatch.setattr(model.denoiser, "forward", lambda *args: noise)
    assert model.loss(mel, frames, mask, noise=noise, timesteps=t) == 0
    noisy = model.q_sample(mel, t, noise)
    alpha = model.alpha_bar[t][:, None, None]
    recovered = (noisy - (1 - alpha).sqrt() * noise) / alpha.sqrt()
    assert torch.allclose(recovered, mel, atol=1e-6)


def test_sampler_rejects_invalid_steps():
    model = MelDiffusion(hidden=16, train_steps=20)
    with pytest.raises(ValueError, match="sample steps"):
        model.sample(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool), steps=21)


def test_d2_composes_with_encoder_and_regulator():
    model = AcousticModel(5, system="D2", hidden=16)
    batch = collate([dict(clip_id="fake", phone_ids=torch.tensor([1, 2]),
                          durations=torch.tensor([2, 3]), mel=torch.randn(N_MELS, 5))])
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.embed.weight.grad.abs().sum() > 0
