"""Mel normalisation statistics  (M1 — phase 2).

Per-band mean and standard deviation of the log-mel spectrograms, over the
TRAINING SPLIT ONLY. Using all 13,100 clips would leak validation and test
statistics into training.

Why normalise at all: raw log-mels run about -11.5 to +1.5 and each of the 80
bands has a different typical level. Normalising per band gives every band a
comparable scale, which is what a network's weight initialisation assumes.

    normalised = (mel - mean[:, None]) / std[:, None]

THE TRAP WORTH KNOWING: HiFi-GAN was trained on RAW log-mels, not normalised
ones. Anything we feed the vocoder must be denormalised first. Use denormalise()
rather than doing it by hand in three different files.

Usage:
    python -m src.dl.mel_stats
    from src.dl.mel_stats import MelStats
    stats = MelStats.load()
    x = stats.normalise(mel)        # for the model
    mel = stats.denormalise(x)      # before the vocoder

Produces:
    data/cache/mel_stats.json  — 80 means, 80 stds, and the clip count they came from
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.config import MEL_DIR, N_MELS, ROOT
from src.dl.config import MEL_STATS_PATH
from src.utils import read_filelist


@dataclass(frozen=True)
class MelStats:
    mean: np.ndarray        # (80,) float32
    std: np.ndarray         # (80,) float32
    n_clips: int
    n_frames: int

    def _as(self, x):
        """Return (mean, std) shaped to broadcast over x, matching its library.

        Note: numpy 2.x arrays also have a .device attribute, so test the type's
        module rather than duck-typing on .device.
        """
        if type(x).__module__.startswith("torch"):
            import torch
            m = torch.as_tensor(self.mean, device=x.device).to(x.dtype)
            s = torch.as_tensor(self.std, device=x.device).to(x.dtype)
            return m[:, None], s[:, None]
        return self.mean[:, None], self.std[:, None]

    def normalise(self, mel):
        """(80, T) raw log-mel -> (80, T) normalised. Works for numpy or torch."""
        m, s = self._as(mel)
        return (mel - m) / s

    def denormalise(self, x):
        """Inverse of normalise(). ALWAYS call this before the vocoder."""
        m, s = self._as(x)
        return x * s + m

    def save(self, path=MEL_STATS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(),
                       "n_clips": self.n_clips, "n_frames": self.n_frames}, f)

    @classmethod
    def load(cls, path=MEL_STATS_PATH) -> "MelStats":
        if not path.exists():
            raise FileNotFoundError(f"missing {path}\nRun: python -m src.dl.mel_stats")
        with open(path) as f:
            d = json.load(f)
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32),
                   n_clips=d["n_clips"], n_frames=d["n_frames"])


def compute(clip_ids, mel_dir=MEL_DIR) -> MelStats:
    """Streaming per-band mean/std, so we never hold 1.1 GB of mels at once."""
    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    frames = 0
    used = 0

    for clip_id in tqdm(clip_ids, desc="mel stats", unit="clip"):
        path = mel_dir / f"{clip_id}.npy"
        if not path.exists():
            continue
        mel = np.load(path).astype(np.float64)       # (80, T)
        total += mel.sum(axis=1)
        total_sq += (mel ** 2).sum(axis=1)
        frames += mel.shape[1]
        used += 1

    if frames == 0:
        raise RuntimeError(f"no mels found in {mel_dir}. Run: make mels")

    mean = total / frames
    var = np.maximum(total_sq / frames - mean ** 2, 1e-8)
    return MelStats(mean=mean.astype(np.float32), std=np.sqrt(var).astype(np.float32),
                    n_clips=used, n_frames=frames)


def main() -> int:
    if not MEL_DIR.exists() or not any(MEL_DIR.glob("*.npy")):
        print(f"no mels in {MEL_DIR}\nRun: make mels", file=sys.stderr)
        return 1

    train = read_filelist("train")
    stats = compute(train)
    stats.save()

    print(f"\n{stats.n_clips:,} training clips, {stats.n_frames:,} frames")
    print(f"  mean  {stats.mean.min():+.3f} .. {stats.mean.max():+.3f}")
    print(f"  std    {stats.std.min():.3f} .. {stats.std.max():.3f}")
    print(f"\nwrote {MEL_STATS_PATH.relative_to(ROOT)}")
    print("Remember: denormalise() before the vocoder — HiFi-GAN expects raw log-mels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
