"""Baseline synthesis: R1, A1, A2  (M4 — steps 4 & 5).

Produces outputs/audio/{R1,A1,A2}/<id>.wav for every clip in test_100.

Systems:
  R1  — copy synthesis: real cached mel → HiFi-GAN vocoder.
         Proves the mel recipe and vocoder; expected WER ≈ R0.
  A1  — phoneme templates + TRUE MFA durations (from phones.csv, M2).
         Tests the template quality with perfect timing.
  A2  — phoneme templates + PREDICTED durations (pred_best from M3).
         Adds duration prediction error on top of A1.

Synthesis algorithm for A1/A2:
  For each phone in a clip with duration d frames:
    1. Split d into three nearly equal parts (p1, p2 = d//3, 2*d//3).
    2. Tile the matching template state column:
         begin  → template[:, 0] repeated p1      times
         middle → template[:, 1] repeated p2-p1   times
         end    → template[:, 2] repeated d-p2     times
    3. Concatenate across all phones → (80, T) mel.
    4. mel_to_wav → wav trimmed to T * HOP_LENGTH samples.

Run:
    python -m src.baseline.synthesize_baseline --system all
    python -m src.baseline.synthesize_baseline --system R1
    python -m src.baseline.synthesize_baseline --system A1
    python -m src.baseline.synthesize_baseline --system A2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio import adjust_durations, load_wav, mel_to_wav, save_wav, trim_wav, wav_to_mel
from src.baseline.phoneme_templates import TEMPLATES_NPZ, split_thirds
from src.config import (
    CACHE_DIR,
    HOP_LENGTH,
    MEL_DIR,
    METADATA_CLEAN,
    OUTPUTS,
    SIL,
)
from src.utils import read_filelist

PHONES_CSV    = CACHE_DIR / "phones.csv"
TEST_PRED_CSV = OUTPUTS / "durations" / "test_pred.csv"
AUDIO_DIR     = OUTPUTS / "audio"

SYSTEMS = ("R1", "A1", "A2")

# ---------------------------------------------------------------------------
# Template synthesis helper
# ---------------------------------------------------------------------------


def synthesize_from_templates(
    phones_clip: pd.DataFrame,
    templates: dict[str, np.ndarray],
    fallback_phone: str = SIL,
) -> np.ndarray:
    """Build an (80, T) mel from phone templates and per-phone durations.

    Args:
        phones_clip: rows from phones.csv for ONE clip, in order.
                     Must have 'phone' and 'n_frames' columns.
        templates:   dict phone → (80, 3) float32 array.
        fallback_phone: phone to use when a symbol has no template
                        (defaults to 'sil' which always has one).

    Returns:
        float32 numpy array of shape (80, T).
    """
    columns: list[np.ndarray] = []

    for row in phones_clip.itertuples(index=False):
        phone = row.phone
        dur   = int(row.n_frames)
        if dur < 1:
            continue

        tmpl = templates.get(phone) or templates.get(fallback_phone)
        if tmpl is None:
            # Last resort: silence-like zeros
            tmpl = np.zeros((80, 3), dtype=np.float32)

        p1, p2 = split_thirds(dur)
        n_begin  = p1
        n_middle = p2 - p1
        n_end    = dur - p2

        seg = np.concatenate([
            np.tile(tmpl[:, 0:1], (1, max(n_begin,  1))),
            np.tile(tmpl[:, 1:2], (1, max(n_middle, 1))),
            np.tile(tmpl[:, 2:3], (1, max(n_end,    1))),
        ], axis=1)                      # (80, ~dur)

        # Exact length: trim or repeat the last column if rounding drifts.
        if seg.shape[1] != dur:
            if seg.shape[1] > dur:
                seg = seg[:, :dur]
            else:
                pad_col = seg[:, -1:]
                seg = np.concatenate(
                    [seg, np.tile(pad_col, (1, dur - seg.shape[1]))], axis=1
                )
        columns.append(seg)

    if not columns:
        raise ValueError("synthesize_from_templates: empty phone sequence")

    return np.concatenate(columns, axis=1).astype(np.float32)  # (80, T)


# ---------------------------------------------------------------------------
# Per-system synthesis
# ---------------------------------------------------------------------------


def run_R1(
    ids: list[str],
    meta_by_id: dict[str, dict],
) -> None:
    """R1: real mel → vocoder → wav."""
    out_dir = AUDIO_DIR / "R1"
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip_id in tqdm(ids, desc="R1", unit="clip"):
        out_path = out_dir / f"{clip_id}.wav"
        if out_path.exists():
            continue

        mel_path = MEL_DIR / f"{clip_id}.npy"
        if not mel_path.exists():
            print(f"  R1: skipping {clip_id} — mel not cached", flush=True)
            continue

        import torch
        mel_f16 = np.load(mel_path)
        mel = torch.from_numpy(mel_f16.astype(np.float32))  # (80, T)

        wav = mel_to_wav(mel)
        n_samples = meta_by_id[clip_id]["n_samples"]
        wav = trim_wav(wav, n_samples)
        save_wav(out_path, wav)


def run_A1(
    ids: list[str],
    meta_by_id: dict[str, dict],
    phones: pd.DataFrame,
    templates: dict[str, np.ndarray],
) -> None:
    """A1: templates + true MFA durations."""
    out_dir = AUDIO_DIR / "A1"
    out_dir.mkdir(parents=True, exist_ok=True)

    phones_by_clip = {cid: grp.sort_values("idx")
                      for cid, grp in phones.groupby("clip_id")}

    for clip_id in tqdm(ids, desc="A1", unit="clip"):
        out_path = out_dir / f"{clip_id}.wav"
        if out_path.exists():
            continue

        if clip_id not in phones_by_clip:
            print(f"  A1: skipping {clip_id} — no phones", flush=True)
            continue

        import torch
        clip_phones = phones_by_clip[clip_id]
        mel_np = synthesize_from_templates(clip_phones, templates)
        mel    = torch.from_numpy(mel_np)

        wav = mel_to_wav(mel)
        n_samples = meta_by_id[clip_id]["n_samples"]
        wav = trim_wav(wav, n_samples)
        save_wav(out_path, wav)


def run_A2(
    ids: list[str],
    meta_by_id: dict[str, dict],
    phones: pd.DataFrame,
    templates: dict[str, np.ndarray],
    test_pred: pd.DataFrame,
) -> None:
    """A2: templates + predicted durations (pred_best from M3)."""
    out_dir = AUDIO_DIR / "A2"
    out_dir.mkdir(parents=True, exist_ok=True)

    phones_by_clip = {cid: grp.sort_values("idx")
                      for cid, grp in phones.groupby("clip_id")}
    pred_by_clip   = {cid: grp.sort_values("idx")
                      for cid, grp in test_pred.groupby("clip_id")}

    for clip_id in tqdm(ids, desc="A2", unit="clip"):
        out_path = out_dir / f"{clip_id}.wav"
        if out_path.exists():
            continue

        if clip_id not in phones_by_clip or clip_id not in pred_by_clip:
            print(f"  A2: skipping {clip_id} — missing phones or predictions", flush=True)
            continue

        import torch
        clip_phones = phones_by_clip[clip_id].copy()
        preds       = pred_by_clip[clip_id]
        target_T    = meta_by_id[clip_id]["n_frames"]

        # Replace true n_frames with adjusted predictions.
        pred_vals = preds["pred_best"].to_numpy(dtype=np.float32)
        adjusted  = adjust_durations(pred_vals, target_T)
        clip_phones = clip_phones.copy()
        clip_phones["n_frames"] = adjusted

        mel_np = synthesize_from_templates(clip_phones, templates)
        mel    = torch.from_numpy(mel_np)

        wav = mel_to_wav(mel)
        n_samples = meta_by_id[clip_id]["n_samples"]
        wav = trim_wav(wav, n_samples)
        save_wav(out_path, wav)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_templates() -> dict[str, np.ndarray]:
    if not TEMPLATES_NPZ.exists():
        raise FileNotFoundError(
            f"missing {TEMPLATES_NPZ}\nRun: make templates"
        )
    data = np.load(TEMPLATES_NPZ)
    return {k: data[k].astype(np.float32) for k in data.files}


def _load_phones() -> pd.DataFrame:
    if not PHONES_CSV.exists():
        raise FileNotFoundError(
            f"missing {PHONES_CSV}\nRun: make phones (M2)"
        )
    return pd.read_csv(PHONES_CSV, dtype={"clip_id": str, "phone": str})


def _load_test_pred() -> pd.DataFrame:
    if not TEST_PRED_CSV.exists():
        raise FileNotFoundError(
            f"missing {TEST_PRED_CSV}\nRun: make durations (M3)"
        )
    return pd.read_csv(TEST_PRED_CSV, dtype={"clip_id": str})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        choices=["R1", "A1", "A2", "all"],
        default="all",
        help="which system to synthesise (default: all)",
    )
    args = ap.parse_args()

    systems = list(SYSTEMS) if args.system == "all" else [args.system]

    # Always need metadata.
    if not METADATA_CLEAN.exists():
        print(f"missing {METADATA_CLEAN}\nrun: make metadata", file=sys.stderr)
        return 1

    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str})
    meta_by_id = {
        row.id: {"n_samples": row.n_samples, "n_frames": row.n_frames}
        for row in meta.itertuples(index=False)
    }

    ids = read_filelist("test_100")
    print(f"synthesising {len(ids)} clips: {systems}")

    # ---- R1 ----------------------------------------------------------------
    if "R1" in systems:
        run_R1(ids, meta_by_id)

    # ---- A1 / A2 -----------------------------------------------------------
    if "A1" in systems or "A2" in systems:
        try:
            phones    = _load_phones()
            templates = _load_templates()
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        if "A1" in systems:
            run_A1(ids, meta_by_id, phones, templates)

        if "A2" in systems:
            try:
                test_pred = _load_test_pred()
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            run_A2(ids, meta_by_id, phones, templates, test_pred)

    print("\nDone. Audio written to:", AUDIO_DIR)
    for s in systems:
        d = AUDIO_DIR / s
        if d.exists():
            n = sum(1 for _ in d.glob("*.wav"))
            print(f"  {s}: {n} wav files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
