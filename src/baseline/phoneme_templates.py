"""Build phoneme template spectrograms from cached mels  (M4 — step 3).

For each phone type, we collect every occurrence in the TRAINING set and
split it into three equal (or nearly equal) temporal thirds:

    begin  →  frames [ 0 : d//3 ]
    middle →  frames [ d//3 : 2*d//3 ]
    end    →  frames [ 2*d//3 : d ]

The 80-dim average of each state across all occurrences gives one column.
Result per phone: a (80, 3) matrix saved inside data/cache/templates.npz.

Usage:
    python -m src.baseline.phoneme_templates

Requires (from M2):
    data/cache/phones.csv  — columns: clip_id, idx, phone, start_frame, n_frames,
                              word, word_idx, idx_in_word, word_len

Produces:
    data/cache/templates.npz  — keys: phone symbols, values: float32 (80, 3)
    data/cache/template_counts.csv  — occurrence count per phone (diagnostics)

Silences ('sil') are included so synthesis can handle leading/trailing silences.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CACHE_DIR, MEL_DIR, METADATA_CLEAN, SIL
from src.utils import read_filelist

PHONES_CSV = CACHE_DIR / "phones.csv"
TEMPLATES_NPZ = CACHE_DIR / "templates.npz"
COUNTS_CSV = CACHE_DIR / "template_counts.csv"

# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------


def split_thirds(d: int) -> tuple[int, int]:
    """Return the two split points for a duration-d occurrence.

    begin  : [0, p1)
    middle : [p1, p2)
    end    : [p2, d)
    """
    p1 = d // 3
    p2 = 2 * d // 3
    # If p1 == p2 (d <= 2), make at least begin and end non-empty.
    p1 = max(p1, 1)
    p2 = max(p2, p1 + 1)
    p1 = min(p1, d - 1)
    p2 = min(p2, d)
    return p1, p2


def build_templates(
    phones: pd.DataFrame,
    train_ids: set[str],
) -> dict[str, np.ndarray]:
    """Compute (80, 3) template for each phone symbol.

    Args:
        phones: the full phones.csv DataFrame.
        train_ids: set of clip IDs belonging to the training split.

    Returns:
        dict mapping phone → float32 array (80, 3).
    """
    # Filter to training clips only.
    train_phones = phones[phones["clip_id"].isin(train_ids)].copy()
    print(f"  training phoneme occurrences: {len(train_phones):,}")

    # Accumulators: phone → list of (80, 3) arrays
    accum: dict[str, list[np.ndarray]] = defaultdict(list)

    for clip_id, group in tqdm(
        train_phones.groupby("clip_id", sort=False),
        desc="reading mels",
        unit="clip",
    ):
        mel_path = MEL_DIR / f"{clip_id}.npy"
        if not mel_path.exists():
            continue  # clip not yet cached — skip silently

        mel = np.load(mel_path).astype(np.float32)  # (80, T)

        for row in group.itertuples(index=False):
            start = int(row.start_frame)
            dur = int(row.n_frames)
            if dur < 1:
                continue

            seg = mel[:, start : start + dur]       # (80, dur)
            if seg.shape[1] < 1:
                continue

            p1, p2 = split_thirds(seg.shape[1])

            begin  = seg[:, :p1].mean(axis=1)        # (80,)
            middle = seg[:, p1:p2].mean(axis=1)      # (80,)
            end    = seg[:, p2:].mean(axis=1)         # (80,)

            accum[row.phone].append(
                np.stack([begin, middle, end], axis=1)  # (80, 3)
            )

    templates: dict[str, np.ndarray] = {}
    for phone, arrays in accum.items():
        templates[phone] = np.stack(arrays, axis=0).mean(axis=0)  # (80, 3)

    return templates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    if not PHONES_CSV.exists():
        print(
            f"missing {PHONES_CSV}\n"
            "Run make phones (M2) first, or supply a fake phones.csv for testing.",
            file=sys.stderr,
        )
        return 1

    if not MEL_DIR.exists() or not any(MEL_DIR.glob("*.npy")):
        print(
            f"no mels found in {MEL_DIR}\nRun make mels first.",
            file=sys.stderr,
        )
        return 1

    phones = pd.read_csv(
        PHONES_CSV,
        dtype={"clip_id": str, "phone": str},
    )
    required_cols = {"clip_id", "idx", "phone", "start_frame", "n_frames"}
    missing = required_cols - set(phones.columns)
    assert not missing, f"phones.csv missing columns: {missing}"

    train_ids = set(read_filelist("train"))
    print(f"  training clips in filelist: {len(train_ids):,}")

    templates = build_templates(phones, train_ids)

    if not templates:
        print("ERROR: no templates built — check that mels exist for training clips.",
              file=sys.stderr)
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(TEMPLATES_NPZ, **templates)

    # Write diagnostic counts.
    counts = {
        phone: len(arrays)
        for phone, arrays in _count_occurrences(phones, train_ids).items()
    }
    pd.DataFrame(
        sorted(counts.items(), key=lambda x: -x[1]),
        columns=["phone", "count"],
    ).to_csv(COUNTS_CSV, index=False)

    print(f"\n  templates built: {len(templates)} phones")
    print(f"  saved → {TEMPLATES_NPZ}")
    print(f"  counts → {COUNTS_CSV}")
    if SIL in templates:
        print(f"  '{SIL}' included ✓")
    else:
        print(f"  WARNING: silence symbol '{SIL}' not found in phones.csv")
    return 0


def _count_occurrences(phones: pd.DataFrame, train_ids: set[str]) -> dict[str, list]:
    """Helper: return {phone: list_of_rows} for count reporting."""
    train = phones[phones["clip_id"].isin(train_ids)]
    return {p: list(g.itertuples()) for p, g in train.groupby("phone")}


if __name__ == "__main__":
    raise SystemExit(main())
