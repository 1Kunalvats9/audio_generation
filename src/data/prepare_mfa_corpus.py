"""Build the MFA corpus folders for M2  (M1 -> M2).

MFA wants a speaker folder holding matched pairs:

    data/mfa_<name>/LJ/<id>.wav
    data/mfa_<name>/LJ/<id>.lab     <- the cleaned transcript, one line

"LJ" is the speaker name (LJSpeech is a single speaker, so one folder).
The wavs are HARD LINKED, not copied: same bytes on disk, no extra gigabytes,
and MFA still sees ordinary files.

Run from the repo root, pilot first:

    python -m src.data.prepare_mfa_corpus pilot     # 200 train clips, hand to M2
    python -m src.data.prepare_mfa_corpus core      # test + val + 4,000 train
    python -m src.data.prepare_mfa_corpus rest      # every remaining train clip
    python -m src.data.prepare_mfa_corpus all
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CORE_TRAIN_CLIPS, PILOT_CLIPS, ROOT, SEED, WAV_DIR, METADATA_CLEAN
from src.utils import read_filelist

SPEAKER = "LJ"


def make_corpus(ids, out_dir, text_by_id) -> int:
    """Write <id>.wav (hard link) + <id>.lab (cleaned text) for every id."""
    speaker_dir = out_dir / SPEAKER
    speaker_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for clip_id in tqdm(sorted(ids), desc=out_dir.name):
        src = WAV_DIR / f"{clip_id}.wav"
        if not src.exists():
            raise FileNotFoundError(src)

        dst = speaker_dir / f"{clip_id}.wav"
        if not dst.exists():
            try:
                os.link(src, dst)          # hard link: no extra disk space
            except OSError:
                shutil.copy2(src, dst)     # different filesystem -> copy

        text = text_by_id[clip_id].strip()
        assert text, f"{clip_id}: empty transcript"
        (speaker_dir / f"{clip_id}.lab").write_text(text + "\n", encoding="utf-8")
        written += 1

    return written


def select(which: str, df: pd.DataFrame) -> list[str]:
    """Which clip IDs go in each corpus folder."""
    rng = np.random.default_rng(SEED)
    train = read_filelist("train")

    if which == "pilot":
        return sorted(rng.choice(train, size=PILOT_CLIPS, replace=False).tolist())

    # the pilot is drawn first, so core/rest stay consistent run to run
    pilot = set(rng.choice(train, size=PILOT_CLIPS, replace=False).tolist())
    remaining = sorted(set(train) - pilot)
    core_train = set(rng.choice(
        remaining, size=min(CORE_TRAIN_CLIPS, len(remaining)), replace=False).tolist())

    if which == "core":
        return sorted(set(read_filelist("test")) | set(read_filelist("val")) | core_train)
    if which == "rest":
        return sorted(set(remaining) - core_train)
    raise ValueError(which)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=["pilot", "core", "rest", "all"])
    args = ap.parse_args()

    if not METADATA_CLEAN.exists():
        print("run: make metadata", file=sys.stderr)
        return 1

    df = pd.read_csv(METADATA_CLEAN, dtype={"id": str})
    text_by_id = dict(zip(df["id"], df["text_clean"]))

    targets = ["pilot", "core", "rest"] if args.which == "all" else [args.which]
    for which in targets:
        ids = select(which, df)
        out_dir = ROOT / "data" / f"mfa_{which}"
        n = make_corpus(ids, out_dir, text_by_id)
        print(f"  {out_dir.relative_to(ROOT)}  {n} clips "
              f"({n * 2} files in {out_dir.name}/{SPEAKER}/)")

    print("\nHand to M2. Align with:")
    print("  make align-pilot      # then align-core, align-rest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
