"""Phase 2: the phoneme encoder and the length regulator (M2)."""

import pytest
import torch

from src.dl.config import HIDDEN, PAD_ID
from src.dl.encoder import (ConvBlock, LengthRegulator, PhonemeEncoder,
                            SinusoidalPositionalEncoding)

VOCAB = 71


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return PhonemeEncoder(VOCAB).eval()       # eval() so dropout is off and runs are comparable


def _ids(rows):
    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_output_shape(encoder):
    ids = _ids([[5, 9, 3, 0, 0], [7, 2, 0, 0, 0]])
    out = encoder(ids, ids != PAD_ID)
    assert out.shape == (2, 5, HIDDEN)


def test_mask_is_derived_when_not_given(encoder):
    ids = _ids([[5, 9, 0, 0]])
    assert torch.allclose(encoder(ids), encoder(ids, ids != PAD_ID))


def test_padded_positions_are_exactly_zero(encoder):
    ids = _ids([[5, 9, 3, 0, 0], [7, 0, 0, 0, 0]])
    out = encoder(ids, ids != PAD_ID)
    assert torch.all(out[ids == PAD_ID] == 0)


def test_padding_does_not_change_real_positions(encoder):
    """THE test for this module.

    A convolution reaches across its kernel. If padding is not masked out, a
    clip's encoding depends on how long the *other* clips in its batch were —
    so the model behaves differently at batch size 1 than at 16, and training
    silently degrades. Here the same real sequence is padded to two different
    lengths and must encode identically.
    """
    short = _ids([[5, 9, 3]])
    long = _ids([[5, 9, 3, 0, 0, 0, 0]])
    a = encoder(short, short != PAD_ID)
    b = encoder(long, long != PAD_ID)[:, :3]
    assert torch.allclose(a, b, atol=1e-6)


def test_a_clip_is_unaffected_by_its_batch_mates(encoder):
    """Same sequence, encoded alone and alongside a much longer neighbour."""
    alone = _ids([[5, 9, 3, 0, 0, 0]])
    together = _ids([[5, 9, 3, 0, 0, 0], [8, 8, 8, 8, 8, 8]])
    a = encoder(alone, alone != PAD_ID)[0, :3]
    b = encoder(together, together != PAD_ID)[0, :3]
    assert torch.allclose(a, b, atol=1e-6)


def test_different_phonemes_give_different_encodings(encoder):
    ids = _ids([[5, 9, 3]])
    out = encoder(ids, ids != PAD_ID)[0]
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_gradients_reach_the_embedding(encoder):
    ids = _ids([[5, 9, 3, 0]])
    encoder.train()
    encoder(ids, ids != PAD_ID).sum().backward()
    grad = encoder.embed.weight.grad
    assert grad is not None
    assert grad[5].abs().sum() > 0, "a used phoneme should receive gradient"
    assert grad[PAD_ID].abs().sum() == 0, "padding_idx must never be updated"


def test_padding_embedding_starts_at_zero():
    enc = PhonemeEncoder(VOCAB)
    assert torch.all(enc.embed.weight[PAD_ID] == 0)


def test_an_even_kernel_is_rejected():
    with pytest.raises(AssertionError, match="odd kernel"):
        ConvBlock(HIDDEN, kernel=4, dropout=0.0)


def test_positional_encoding_is_added_and_bounded():
    pos = SinusoidalPositionalEncoding(HIDDEN)
    x = torch.zeros(1, 6, HIDDEN)
    out = pos(x)
    assert out.shape == x.shape
    assert out.abs().max() <= 1.0 + 1e-6
    assert not torch.allclose(out[0, 0], out[0, 1]), "positions must differ"


def test_positional_encoding_rejects_too_long_a_sequence():
    pos = SinusoidalPositionalEncoding(HIDDEN, max_len=8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pos(torch.zeros(1, 9, HIDDEN))


# ---------------------------------------------------------------------------
# Length regulator
# ---------------------------------------------------------------------------


def test_expansion_repeats_each_phoneme_by_its_duration():
    lr = LengthRegulator()
    x = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])     # (1, 3, 2)
    dur = torch.tensor([[2, 3, 1]])
    out, mask = lr(x, dur)
    assert out.shape == (1, 6, 2)
    expected = torch.tensor([[1.0, 1.0], [1.0, 1.0],
                             [2.0, 2.0], [2.0, 2.0], [2.0, 2.0],
                             [3.0, 3.0]])
    assert torch.allclose(out[0], expected)
    assert mask.all()


def test_frame_count_equals_the_sum_of_durations():
    lr = LengthRegulator()
    x = torch.randn(3, 7, HIDDEN)
    dur = torch.tensor([[2, 1, 3, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [5, 2, 1, 1, 1, 0, 0]])
    out, mask = lr(x, dur)
    assert out.shape[1] == int(dur.sum(dim=1).max())
    assert torch.equal(mask.sum(dim=1), dur.sum(dim=1))


def test_zero_duration_phonemes_contribute_nothing():
    """Padded phonemes carry duration 0, so they need no special handling."""
    lr = LengthRegulator()
    x = torch.tensor([[[1.0], [99.0], [2.0]]])
    dur = torch.tensor([[1, 0, 1]])
    out, _ = lr(x, dur)
    assert torch.allclose(out[0], torch.tensor([[1.0], [2.0]]))


def test_shorter_rows_are_zero_padded():
    lr = LengthRegulator()
    x = torch.ones(2, 3, 4)
    dur = torch.tensor([[1, 1, 1], [1, 0, 0]])
    out, mask = lr(x, dur)
    assert out.shape == (2, 3, 4)
    assert torch.all(out[1, 1:] == 0)
    assert mask[1].tolist() == [True, False, False]


def test_explicit_max_len_pads_and_truncates():
    lr = LengthRegulator()
    x = torch.ones(1, 2, 3)
    out, mask = lr(x, torch.tensor([[2, 2]]), max_len=6)
    assert out.shape[1] == 6 and mask.sum() == 4
    out, mask = lr(x, torch.tensor([[2, 2]]), max_len=3)
    assert out.shape[1] == 3 and mask.sum() == 3


def test_float_durations_are_rounded():
    """A duration predictor emits floats; the regulator must not silently floor them."""
    lr = LengthRegulator()
    x = torch.ones(1, 2, 2)
    out, _ = lr(x, torch.tensor([[1.6, 2.4]]))
    assert out.shape[1] == 4            # 2 + 2

def test_negative_durations_are_rejected():
    lr = LengthRegulator()
    with pytest.raises(ValueError, match="non-negative"):
        lr(torch.ones(1, 2, 2), torch.tensor([[1, -1]]))


def test_gradients_flow_back_through_the_expansion():
    lr = LengthRegulator()
    x = torch.randn(1, 3, 4, requires_grad=True)
    out, _ = lr(x, torch.tensor([[2, 1, 3]]))
    out.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad[0, :, 0], torch.tensor([2.0, 1.0, 3.0])), \
        "each phoneme's gradient should scale with how many frames it produced"


def test_encoder_and_regulator_compose(encoder):
    ids = _ids([[5, 9, 3, 0], [7, 2, 0, 0]])
    dur = torch.tensor([[3, 2, 4, 0], [5, 5, 0, 0]])
    enc = encoder(ids, ids != PAD_ID)
    frames, mask = LengthRegulator()(enc, dur)
    assert frames.shape == (2, 10, HIDDEN)
    assert torch.equal(mask.sum(dim=1), dur.sum(dim=1))
