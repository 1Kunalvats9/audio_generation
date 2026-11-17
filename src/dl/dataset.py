"""Training data for the neural acoustic model  (M1 — phase 2).

THE TENSOR CONTRACT — agreed once, so nobody spends a week on transpose bugs.
Every batch is a dict with exactly these keys, batch dimension first:

    phone_ids    (B, P)      int64    phoneme ids; 0 is padding
    durations    (B, P)      int64    frames per phoneme; 0 where padded
    mel          (B, 80, T)  float32  NORMALISED log-mel (see src/dl/mel_stats.py)
    phone_mask   (B, P)      bool     True = a real phoneme
    frame_mask   (B, T)      bool     True = a real frame
    n_phones     (B,)        int64
    n_frames     (B,)        int64
    clip_ids     list[str]            length B, for debugging and synthesis

Invariants the collate function asserts, so a violation fails here rather than
somewhere deep in a model:

    durations[b].sum() == n_frames[b]           for every b
    phone_mask == (phone_ids != 0)
    T == max(n_frames),  P == max(n_phones)

Two things to remember:
  * `mel` is NORMALISED. Denormalise before the vocoder (MelStats.denormalise).
  * Durations come from MFA, so at training time the model is given the true
    alignment. The duration predictor (M3) is trained against them, and only at
    synthesis time are its own predictions used.

Usage:
    python -m src.dl.dataset                    # self-check, writes the debug subset
    from src.dl.dataset import make_loader
    loader = make_loader("train", batch_size=16)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.config import CACHE_DIR, MEL_DIR, N_MELS, ROOT
from src.dl.config import (BATCH_SIZE, BUCKET_BATCHES, DEBUG_CLIPS, MAX_FRAMES,
                           NUM_WORKERS, PAD_ID, SEED)
from src.dl.mel_stats import MelStats
from src.dl.vocab import PhoneVocab
from src.utils import read_filelist, write_filelist

PHONES_CSV = CACHE_DIR / "phones.csv"
DEBUG_FILELIST = "debug_200"


class PhonemeMelDataset(Dataset):
    """One item = one clip: its phoneme ids, their durations, and its mel."""

    def __init__(self, clip_ids, vocab=None, stats=None, mel_dir=MEL_DIR,
                 phones_csv=PHONES_CSV, max_frames=MAX_FRAMES):
        self.vocab = vocab or PhoneVocab.load()
        self.stats = stats or MelStats.load()
        self.mel_dir = mel_dir

        phones = pd.read_csv(phones_csv, dtype={"clip_id": str, "phone": str})
        wanted = set(clip_ids)
        phones = phones[phones["clip_id"].isin(wanted)].sort_values(["clip_id", "idx"])

        self.items: list[tuple[str, np.ndarray, np.ndarray]] = []
        skipped_long = skipped_missing = 0
        for clip_id, g in phones.groupby("clip_id", sort=True):
            total = int(g["n_frames"].sum())
            if max_frames and total > max_frames:
                skipped_long += 1
                continue
            if not (self.mel_dir / f"{clip_id}.npy").exists():
                skipped_missing += 1
                continue
            ids = np.asarray(self.vocab.encode(g["phone"].tolist()), dtype=np.int64)
            # .copy(): pandas can return a read-only view, which torch.from_numpy warns about
            dur = g["n_frames"].to_numpy(dtype=np.int64).copy()
            self.items.append((clip_id, ids, dur))

        self.skipped_long = skipped_long
        self.skipped_missing = skipped_missing
        if not self.items:
            raise RuntimeError(
                f"dataset is empty: {len(wanted)} clips requested, "
                f"{skipped_long} skipped for exceeding max_frames={max_frames}, "
                f"{skipped_missing} skipped for having no cached mel. "
                "Check the filelist, raise max_frames, or run: make mels")

    def __len__(self) -> int:
        return len(self.items)

    def frame_count(self, i: int) -> int:
        """Used by the bucketing sampler without touching the disk."""
        return int(self.items[i][2].sum())

    def __getitem__(self, i: int) -> dict:
        clip_id, ids, dur = self.items[i]
        mel = np.load(self.mel_dir / f"{clip_id}.npy").astype(np.float32)   # (80, T) raw
        assert mel.shape[0] == N_MELS, f"{clip_id}: mel has {mel.shape[0]} bands, expected {N_MELS}"
        assert mel.shape[1] == dur.sum(), (
            f"{clip_id}: durations sum to {dur.sum()} but the mel has {mel.shape[1]} frames")
        return {
            "clip_id": clip_id,
            "phone_ids": torch.from_numpy(ids),
            "durations": torch.from_numpy(dur),
            "mel": torch.from_numpy(self.stats.normalise(mel)),
        }


def collate(batch: list[dict]) -> dict:
    """Pad a list of items into the batch described in this module's docstring."""
    B = len(batch)
    n_phones = torch.tensor([b["phone_ids"].numel() for b in batch], dtype=torch.int64)
    n_frames = torch.tensor([b["mel"].shape[1] for b in batch], dtype=torch.int64)
    P, T = int(n_phones.max()), int(n_frames.max())

    phone_ids = torch.full((B, P), PAD_ID, dtype=torch.int64)
    durations = torch.zeros((B, P), dtype=torch.int64)
    mel = torch.zeros((B, N_MELS, T), dtype=torch.float32)

    for i, item in enumerate(batch):
        p, t = int(n_phones[i]), int(n_frames[i])
        phone_ids[i, :p] = item["phone_ids"]
        durations[i, :p] = item["durations"]
        mel[i, :, :t] = item["mel"]

    phone_mask = phone_ids != PAD_ID
    frame_mask = torch.arange(T)[None, :] < n_frames[:, None]

    assert torch.equal(durations.sum(dim=1), n_frames), "durations do not sum to the frame count"
    return {
        "clip_ids": [b["clip_id"] for b in batch],
        "phone_ids": phone_ids, "durations": durations, "mel": mel,
        "phone_mask": phone_mask, "frame_mask": frame_mask,
        "n_phones": n_phones, "n_frames": n_frames,
    }


class LengthBucketSampler(Sampler):
    """Batches of similar-length clips, so a batch is not mostly padding.

    Sorts by frame count, cuts into batches, then shuffles the batch order —
    the model still sees batches in a random order each epoch, but each batch
    is internally homogeneous.
    """

    def __init__(self, dataset: PhonemeMelDataset, batch_size: int, shuffle: bool = True,
                 seed: int = SEED):
        self.dataset, self.batch_size, self.shuffle, self.seed = dataset, batch_size, shuffle, seed
        self.epoch = 0

    def __iter__(self):
        order = np.argsort([self.dataset.frame_count(i) for i in range(len(self.dataset))])
        batches = [order[i:i + self.batch_size].tolist()
                   for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_loader(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = True,
                num_workers: int = NUM_WORKERS, bucket: bool = BUCKET_BATCHES,
                **dataset_kwargs) -> DataLoader:
    """A DataLoader over one filelist ("train", "val", "debug_200", ...)."""
    ds = PhonemeMelDataset(read_filelist(split), **dataset_kwargs)
    if bucket:
        return DataLoader(ds, batch_sampler=LengthBucketSampler(ds, batch_size, shuffle),
                          collate_fn=collate, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers, drop_last=False)


def write_debug_subset(n: int = DEBUG_CLIPS) -> list[str]:
    """A laptop-sized slice of train, so everyone's code runs in seconds."""
    rng = np.random.default_rng(SEED)
    train = read_filelist("train")
    ids = sorted(rng.choice(train, size=min(n, len(train)), replace=False).tolist())
    write_filelist(DEBUG_FILELIST, ids)
    return ids


def main() -> int:
    for path, how in ((PHONES_CSV, "git pull (it is committed)"), (MEL_DIR, "make mels")):
        if not path.exists():
            print(f"missing {path}\nRun: {how}", file=sys.stderr)
            return 1

    ids = write_debug_subset()
    print(f"wrote filelists/{DEBUG_FILELIST}.txt ({len(ids)} clips)")

    loader = make_loader(DEBUG_FILELIST, batch_size=4, num_workers=0)
    ds = loader.dataset
    print(f"dataset: {len(ds)} clips  (skipped {ds.skipped_long} over {MAX_FRAMES} frames, "
          f"{ds.skipped_missing} without a cached mel)")

    batch = next(iter(loader))
    print("\none batch:")
    for k in ("phone_ids", "durations", "mel", "phone_mask", "frame_mask", "n_phones", "n_frames"):
        v = batch[k]
        print(f"  {k:11s} {tuple(v.shape)}  {v.dtype}")
    print(f"  clip_ids    {batch['clip_ids']}")
    pad_frac = 1 - batch["phone_mask"].float().mean().item()
    print(f"\npadding in this batch: {pad_frac:.1%} of phoneme slots")
    print(f"mel (normalised) mean {batch['mel'][batch['frame_mask'].unsqueeze(1).expand_as(batch['mel'])].mean():+.3f} "
          f"std {batch['mel'][batch['frame_mask'].unsqueeze(1).expand_as(batch['mel'])].std():.3f}  (expect ~0 / ~1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
