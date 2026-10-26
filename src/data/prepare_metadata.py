"""metadata.csv -> data/metadata_clean.csv  (M1)

Reads LJSpeech's pipe-separated metadata, cleans the normalised transcript,
and measures every wav. Run from the repo root:

    python -m src.data.prepare_metadata

Output columns (FROZEN - M2/M3/M4/M5 read these):
    id, text_raw, text_clean, n_samples, n_frames, seconds
"""

import sys

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import METADATA_CLEAN, METADATA_RAW, SAMPLE_RATE, WAV_DIR, n_frames
from src.text.normalize import clean_text

COLUMNS = ["id", "text_raw", "text_clean", "n_samples", "n_frames", "seconds"]


def load_metadata() -> pd.DataFrame:
    """Read metadata.csv: 3 pipe-separated fields, no header, no quoting.

    Column 3 (the normalised text, numbers spelled out) is what we clean,
    because the reader spoke that version.
    """
    df = pd.read_csv(
        METADATA_RAW,
        sep="|",
        header=None,
        names=["id", "text_raw", "text_norm"],
        quoting=3,            # QUOTE_NONE: transcripts contain bare " characters
        dtype=str,
        keep_default_na=False,
    )
    assert len(df) == 13100, f"expected 13,100 rows, got {len(df)}"
    assert df["id"].is_unique, "duplicate clip IDs"
    assert (df["text_norm"].str.len() > 0).all(), "empty normalised text"
    df["text_clean"] = df["text_norm"].map(clean_text)
    assert (df["text_clean"].str.len() > 0).all(), "cleaning emptied a transcript"
    return df


def add_audio_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add n_samples / n_frames / seconds, asserting 22,050 Hz mono throughout.

    soundfile.info only reads the header, so this never decodes the audio.
    """
    n_samples = []
    for clip_id in tqdm(df["id"], desc="reading wav headers"):
        info = sf.info(WAV_DIR / f"{clip_id}.wav")
        if info.samplerate != SAMPLE_RATE:
            raise AssertionError(f"{clip_id}: {info.samplerate} Hz, expected {SAMPLE_RATE}")
        if info.channels != 1:
            raise AssertionError(f"{clip_id}: {info.channels} channels, expected mono")
        n_samples.append(info.frames)

    df = df.copy()
    df["n_samples"] = n_samples
    df["n_frames"] = df["n_samples"].map(n_frames)
    df["seconds"] = df["n_samples"] / SAMPLE_RATE
    return df


def main() -> int:
    if not METADATA_RAW.exists():
        print(f"missing {METADATA_RAW}\nrun: make data", file=sys.stderr)
        return 1

    df = add_audio_info(load_metadata())[COLUMNS]

    METADATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_CLEAN, index=False)

    total_h = df["seconds"].sum() / 3600
    print(f"\nwrote {METADATA_CLEAN}  ({len(df)} rows)")
    print(f"  total audio   {total_h:.2f} h")
    print(f"  clip seconds  min {df.seconds.min():.2f}  "
          f"median {df.seconds.median():.2f}  max {df.seconds.max():.2f}")
    print(f"  frames        min {df.n_frames.min()}  max {df.n_frames.max()}")
    print(f"  chars         min {df.text_clean.str.len().min()}  "
          f"max {df.text_clean.str.len().max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
