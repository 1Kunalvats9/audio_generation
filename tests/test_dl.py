"""Phase 2 skeleton: vocabulary, mel statistics, dataset and the batch contract.

Everything here runs on synthetic data in tmp_path, so the tests do not need the
1.1 GB mel cache and run anywhere in seconds.
"""

import json

import numpy as np
import pytest
import torch

from src.config import N_MELS
from src.dl.config import PAD_ID, PAD_SYMBOL
from src.dl.dataset import LengthBucketSampler, PhonemeMelDataset, collate
from src.dl.mel_stats import MelStats, compute
from src.dl.vocab import PhoneVocab
from tests.fake_phones import fake_phones


@pytest.fixture
def corpus(tmp_path):
    """A tiny corpus: fake phones.csv plus a matching mel per clip."""
    phones = fake_phones(6, seed=3)
    csv = tmp_path / "phones.csv"
    phones.to_csv(csv, index=False)

    mel_dir = tmp_path / "mels"
    mel_dir.mkdir()
    rng = np.random.default_rng(0)
    for clip_id, g in phones.groupby("clip_id"):
        T = int(g["n_frames"].sum())
        mel = (rng.normal(-5.0, 1.5, size=(N_MELS, T))).astype(np.float32)
        np.save(mel_dir / f"{clip_id}.npy", mel)

    vocab = PhoneVocab.build(phones)
    stats = compute(sorted(phones.clip_id.unique()), mel_dir=mel_dir)
    ids = sorted(phones.clip_id.unique())
    return dict(phones=phones, csv=csv, mel_dir=mel_dir, vocab=vocab, stats=stats, ids=ids)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_padding_is_id_zero(corpus):
    v = corpus["vocab"]
    assert v.symbols[PAD_ID] == PAD_SYMBOL
    assert PAD_ID == 0, "nn.Embedding(padding_idx=0) and the `ids != 0` mask both assume this"


def test_every_real_phone_gets_a_nonzero_id(corpus):
    v = corpus["vocab"]
    real = sorted(corpus["phones"].phone.unique())
    assert all(i > 0 for i in v.encode(real))
    assert v.size == len(real) + 1


def test_encode_decode_round_trip(corpus):
    v = corpus["vocab"]
    seq = ["sil", "P", "IH1", "sil"]
    assert v.decode(v.encode(seq)) == seq


def test_unknown_phone_raises_rather_than_silently_padding(corpus):
    with pytest.raises(KeyError, match="not in the frozen vocabulary"):
        corpus["vocab"].encode(["P", "NOT_A_PHONE"])


def test_vocab_survives_save_and_load(corpus, tmp_path):
    path = tmp_path / "vocab.json"
    corpus["vocab"].save(path)
    assert PhoneVocab.load(path) == corpus["vocab"]


def test_vocab_is_sorted_so_it_rebuilds_identically(corpus):
    syms = corpus["vocab"].symbols
    assert list(syms[1:]) == sorted(syms[1:]), "unsorted ids would differ between machines"


# ---------------------------------------------------------------------------
# Mel statistics
# ---------------------------------------------------------------------------


def test_stats_have_one_value_per_mel_band(corpus):
    st = corpus["stats"]
    assert st.mean.shape == (N_MELS,) and st.std.shape == (N_MELS,)
    assert (st.std > 0).all()


def test_normalising_gives_zero_mean_unit_std(corpus):
    st = corpus["stats"]
    mels = [np.load(corpus["mel_dir"] / f"{c}.npy") for c in corpus["ids"]]
    allm = np.concatenate([st.normalise(m) for m in mels], axis=1)
    assert abs(allm.mean()) < 0.05
    assert abs(allm.std() - 1.0) < 0.05


@pytest.mark.parametrize("to_torch", [False, True])
def test_denormalise_inverts_normalise(corpus, to_torch):
    """Regression: numpy 2.x arrays also have .device, so the library test must
    not duck-type on that attribute."""
    st = corpus["stats"]
    mel = np.load(corpus["mel_dir"] / f"{corpus['ids'][0]}.npy")
    x = torch.from_numpy(mel) if to_torch else mel
    back = st.denormalise(st.normalise(x))
    back = back.numpy() if to_torch else back
    assert np.allclose(back, mel, atol=1e-4)


def test_stats_survive_save_and_load(corpus, tmp_path):
    path = tmp_path / "stats.json"
    corpus["stats"].save(path)
    loaded = MelStats.load(path)
    assert np.allclose(loaded.mean, corpus["stats"].mean)
    assert np.allclose(loaded.std, corpus["stats"].std)


# ---------------------------------------------------------------------------
# Dataset and the batch contract
# ---------------------------------------------------------------------------


def _dataset(corpus, **kw):
    return PhonemeMelDataset(corpus["ids"], vocab=corpus["vocab"], stats=corpus["stats"],
                             mel_dir=corpus["mel_dir"], phones_csv=corpus["csv"], **kw)


def test_dataset_loads_every_clip(corpus):
    ds = _dataset(corpus)
    assert len(ds) == len(corpus["ids"])


def test_item_shapes_and_durations_match_the_mel(corpus):
    ds = _dataset(corpus)
    item = ds[0]
    assert item["mel"].shape[0] == N_MELS
    assert item["durations"].sum().item() == item["mel"].shape[1]
    assert item["phone_ids"].numel() == item["durations"].numel()


def test_long_clips_are_skipped(corpus):
    """An empty dataset is refused, and the error names which filter emptied it."""
    with pytest.raises(RuntimeError, match=r"skipped for exceeding max_frames=10"):
        _dataset(corpus, max_frames=10)


def test_some_long_clips_are_skipped_without_emptying_the_dataset(corpus):
    frames = sorted(int(g.n_frames.sum()) for _, g in corpus["phones"].groupby("clip_id"))
    cutoff = frames[len(frames) // 2]          # keep roughly half
    ds = _dataset(corpus, max_frames=cutoff)
    assert 0 < len(ds) < len(corpus["ids"])
    assert ds.skipped_long == len(corpus["ids"]) - len(ds)
    assert all(ds.frame_count(i) <= cutoff for i in range(len(ds)))


def test_clips_without_a_cached_mel_are_skipped(corpus):
    (corpus["mel_dir"] / f"{corpus['ids'][0]}.npy").unlink()
    ds = _dataset(corpus)
    assert len(ds) == len(corpus["ids"]) - 1 and ds.skipped_missing == 1


def test_collate_matches_the_documented_contract(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[i] for i in range(len(ds))])
    B = len(ds)
    P, T = int(batch["n_phones"].max()), int(batch["n_frames"].max())

    assert batch["phone_ids"].shape == (B, P) and batch["phone_ids"].dtype == torch.int64
    assert batch["durations"].shape == (B, P) and batch["durations"].dtype == torch.int64
    assert batch["mel"].shape == (B, N_MELS, T) and batch["mel"].dtype == torch.float32
    assert batch["phone_mask"].shape == (B, P) and batch["phone_mask"].dtype == torch.bool
    assert batch["frame_mask"].shape == (B, T) and batch["frame_mask"].dtype == torch.bool
    assert len(batch["clip_ids"]) == B


def test_masks_agree_with_the_lengths(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[i] for i in range(len(ds))])
    assert torch.equal(batch["phone_mask"], batch["phone_ids"] != PAD_ID)
    assert torch.equal(batch["phone_mask"].sum(dim=1), batch["n_phones"])
    assert torch.equal(batch["frame_mask"].sum(dim=1), batch["n_frames"])


def test_durations_sum_to_the_frame_count_for_every_item(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[i] for i in range(len(ds))])
    assert torch.equal(batch["durations"].sum(dim=1), batch["n_frames"])


def test_padded_slots_are_zero(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[i] for i in range(len(ds))])
    pad = ~batch["phone_mask"]
    assert (batch["phone_ids"][pad] == PAD_ID).all()
    assert (batch["durations"][pad] == 0).all()


def test_padded_frames_are_zero(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[i] for i in range(len(ds))])
    for i, n in enumerate(batch["n_frames"].tolist()):
        assert torch.all(batch["mel"][i, :, n:] == 0)


def test_a_single_item_batch_still_works(corpus):
    ds = _dataset(corpus)
    batch = collate([ds[0]])
    assert batch["mel"].shape[0] == 1
    assert batch["phone_mask"].all(), "a batch of one needs no padding"


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_bucket_sampler_covers_every_clip_exactly_once(corpus):
    ds = _dataset(corpus)
    seen = [i for b in LengthBucketSampler(ds, batch_size=2, shuffle=False) for i in b]
    assert sorted(seen) == list(range(len(ds)))


def test_bucket_sampler_groups_similar_lengths(corpus):
    ds = _dataset(corpus)
    for b in LengthBucketSampler(ds, batch_size=2, shuffle=False):
        lens = [ds.frame_count(i) for i in b]
        assert max(lens) - min(lens) <= max(lens) * 0.5


def test_bucket_sampler_batch_count(corpus):
    ds = _dataset(corpus)
    s = LengthBucketSampler(ds, batch_size=4, shuffle=False)
    assert len(s) == len(list(iter(s)))
