"""Whisper ASR Intelligibility & WER/CER Evaluation Harness (M5).

Transcribes audio using OpenAI Whisper (small.en) and scores against
LJSpeech cleaned transcripts using jiwer.

Both reference and hypothesis transcripts are normalized with Whisper's
EnglishTextNormalizer to measure speech intelligibility rather than punctuation differences.

Usage:
    python -m src.eval.whisper_wer --system R0          # evaluate real audio baseline
    python -m src.eval.whisper_wer --all                # evaluate all available systems
    python -m src.eval.whisper_wer --system R0 --limit 5 # quick sanity check on 5 clips
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
import whisper
from whisper.normalizers import EnglishTextNormalizer

from src.config import METADATA_CLEAN, OUTPUTS, ROOT, WAV_DIR, WHISPER_MODEL
from src.utils import read_filelist

SYSTEM_DIRS = {
    "R0": WAV_DIR,
    "R1": OUTPUTS / "audio" / "R1",
    "A1": OUTPUTS / "audio" / "A1",
    "A2": OUTPUTS / "audio" / "A2",
}

RESULTS_DIR = OUTPUTS / "results"
WHISPER_DOWNLOAD_ROOT = ROOT / "pretrained_models" / "whisper"


def get_normalizer() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


def compute_metrics(ref: str, hyp: str) -> tuple[float, float]:
    """Compute WER and CER between normalized reference and hypothesis."""
    if not ref.strip():
        return (0.0, 0.0) if not hyp.strip() else (1.0, 1.0)
    w = jiwer.wer(ref, hyp)
    c = jiwer.cer(ref, hyp)
    return float(w), float(c)


def evaluate_system(
    system: str,
    audio_dir: Path,
    clip_ids: list[str],
    metadata_df: pd.DataFrame,
    model: whisper.Whisper,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate one system on a list of clip IDs."""
    if limit is not None and limit > 0:
        clip_ids = clip_ids[:limit]

    normalizer = get_normalizer()
    text_map = dict(zip(metadata_df["id"], metadata_df["text_clean"]))

    results = []
    missing_files = []

    for clip_id in tqdm(clip_ids, desc=f"Evaluating {system}"):
        wav_path = audio_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            missing_files.append(clip_id)
            continue

        raw_ref = text_map.get(clip_id, "")
        norm_ref = normalizer(raw_ref)

        # Load audio with soundfile and resample to 16kHz for Whisper (avoids external ffmpeg requirement)
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        # Transcribe with Whisper
        use_fp16 = (device == "cuda" and torch.cuda.is_available())
        transcription = model.transcribe(
            audio_16k,
            language="en",
            fp16=use_fp16,
        )
        raw_hyp = transcription.get("text", "").strip()
        norm_hyp = normalizer(raw_hyp)

        wer, cer = compute_metrics(norm_ref, norm_hyp)

        results.append({
            "clip_id": clip_id,
            "system": system,
            "ref": norm_ref,
            "hyp": norm_hyp,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
        })

    if missing_files:
        print(f"[{system}] Warning: {len(missing_files)} / {len(clip_ids)} wav files were missing.", file=sys.stderr)

    df_res = pd.DataFrame(results, columns=["clip_id", "system", "ref", "hyp", "wer", "cer"])
    return df_res


def run_evaluation(
    systems: list[str],
    filelist: str = "test_100",
    audio_dir_override: Optional[Path] = None,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Run Whisper evaluation across requested systems and save CSVs."""
    if not METADATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {METADATA_CLEAN}. Run: make metadata")

    meta_df = pd.read_csv(METADATA_CLEAN, dtype={"id": str})
    clip_ids = read_filelist(filelist)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Loading Whisper model '{WHISPER_MODEL}' on {device} (download_root: {WHISPER_DOWNLOAD_ROOT})...")
    model = whisper.load_model(WHISPER_MODEL, device=device, download_root=str(WHISPER_DOWNLOAD_ROOT))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dfs = {}

    for sys_name in systems:
        audio_dir = audio_dir_override if audio_dir_override else SYSTEM_DIRS.get(sys_name)
        if audio_dir is None or not audio_dir.exists():
            print(f"Skipping {sys_name}: directory {audio_dir} does not exist.")
            continue

        df_sys = evaluate_system(
            system=sys_name,
            audio_dir=audio_dir,
            clip_ids=clip_ids,
            metadata_df=meta_df,
            model=model,
            device=device,
            limit=limit,
        )

        if not df_sys.empty:
            out_path = RESULTS_DIR / f"wer_{sys_name}.csv"
            df_sys.to_csv(out_path, index=False)
            mean_wer = df_sys["wer"].mean()
            mean_cer = df_sys["cer"].mean()
            print(f"\n[{sys_name}] {len(df_sys)} clips evaluated -> {out_path}")
            print(f"  Mean WER: {mean_wer * 100:.2f}%")
            print(f"  Mean CER: {mean_cer * 100:.2f}%")
            out_dfs[sys_name] = df_sys
        else:
            print(f"[{sys_name}] No clips evaluated.")

    return out_dfs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["R0", "R1", "A1", "A2"], help="Evaluate a specific system")
    parser.add_argument("--all", action="store_true", help="Evaluate all available systems")
    parser.add_argument("--filelist", default="test_100", help="Clip list to evaluate (default: test_100)")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Custom audio directory for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips evaluated (for fast test)")
    parser.add_argument("--device", default=None, help="Device to run on ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.all:
        systems = ["R0", "R1", "A1", "A2"]
    elif args.system:
        systems = [args.system]
    else:
        # Default to R0 if nothing specified
        systems = ["R0"]

    run_evaluation(
        systems=systems,
        filelist=args.filelist,
        audio_dir_override=args.audio_dir,
        limit=args.limit,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
