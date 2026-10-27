"""Extract and cache mel spectrograms for all clips  (M4 — step 2).

Reads every wav from data/raw/LJSpeech-1.1/wavs/, converts to a log-mel
spectrogram with wav_to_mel(), and saves as float16 .npy in data/cache/mels/.

Run from the repo root:

    python -m src.data.extract_mels          # all 13,100 clips
    python -m src.data.extract_mels --split test_100   # one filelist only

Shape contract (FROZEN, M2/M3/M5 rely on it):
    mel.npy  →  float16, shape (80, T)   where T == n_frames(n_samples)

Each mel file is asserted at save time.  Any mismatch raises immediately
rather than silently producing a broken cache.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio import load_wav, wav_to_mel
from src.config import MEL_DIR, METADATA_CLEAN, WAV_DIR, n_frames
from src.utils import read_filelist


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def extract_one(clip_id: str, expected_n_frames: int) -> np.ndarray:
    """Load wav, compute mel, assert shape, return float16 array (80, T)."""
    wav = load_wav(WAV_DIR / f"{clip_id}.wav")
    mel = wav_to_mel(wav)                       # float32 tensor (80, T)

    T = mel.shape[1]
    assert T == expected_n_frames, (
        f"{clip_id}: mel has {T} frames, expected {expected_n_frames} "
        f"(n_samples={len(wav)} → n_frames={n_frames(len(wav))})"
    )

    return mel.numpy().astype(np.float16)       # (80, T) float16


def extract_all(ids: list[str], meta: pd.DataFrame) -> None:
    """Extract mels for the given clip IDs, skipping already-cached files."""
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    frames_by_id = dict(zip(meta["id"], meta["n_frames"]))

    skipped = 0
    errors: list[str] = []

    for clip_id in tqdm(ids, desc="extracting mels", unit="clip"):
        out_path = MEL_DIR / f"{clip_id}.npy"
        if out_path.exists():
            skipped += 1
            continue

        if clip_id not in frames_by_id:
            errors.append(f"{clip_id}: not in metadata_clean.csv")
            continue

        try:
            mel_f16 = extract_one(clip_id, int(frames_by_id[clip_id]))
            np.save(out_path, mel_f16)
        except Exception as exc:
            errors.append(f"{clip_id}: {exc}")

    print(f"\n  saved   {len(ids) - skipped - len(errors):6d} mels  → {MEL_DIR}")
    print(f"  skipped {skipped:6d} (already cached)")
    if errors:
        print(f"  ERRORS  {len(errors):6d}:")
        for e in errors[:20]:
            print(f"    {e}")


# ---------------------------------------------------------------------------
# Verification helper — check every cached mel matches metadata
# ---------------------------------------------------------------------------


def verify(ids: list[str], meta: pd.DataFrame) -> int:
    """Return the number of mismatches found (0 = all good)."""
    frames_by_id = dict(zip(meta["id"], meta["n_frames"]))
    bad = 0
    for clip_id in tqdm(ids, desc="verifying", unit="clip"):
        path = MEL_DIR / f"{clip_id}.npy"
        if not path.exists():
            print(f"  MISSING {clip_id}")
            bad += 1
            continue
        mel = np.load(path)
        expected = int(frames_by_id[clip_id])
        if mel.shape != (80, expected):
            print(f"  BAD SHAPE {clip_id}: {mel.shape} expected (80, {expected})")
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        default=None,
        metavar="NAME",
        help="process only one filelist (train / val / test / test_100). "
             "Omit to process all 13,100 clips.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check cached mels match metadata (no re-extraction).",
    )
    args = ap.parse_args()

    if not METADATA_CLEAN.exists():
        print(f"missing {METADATA_CLEAN}\nrun: make metadata", file=sys.stderr)
        return 1

    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str})

    if args.split:
        ids = read_filelist(args.split)
        print(f"processing split '{args.split}': {len(ids)} clips")
    else:
        ids = list(meta["id"])
        print(f"processing all {len(ids)} clips")

    if args.verify:
        bad = verify(ids, meta)
        if bad:
            print(f"\nVERIFICATION FAILED: {bad} problem(s)", file=sys.stderr)
            return 1
        print(f"\nAll {len(ids)} mels verified OK")
        return 0

    extract_all(ids, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
