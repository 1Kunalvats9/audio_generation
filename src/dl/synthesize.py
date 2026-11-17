"""Synthesize D1/D2 on test_100 with the frozen mel statistics and vocoder.

Default durations come from M3's phase-1 GBT predictions. --durations mfa is
an oracle ablation; --durations neural requires a jointly trained predictor.
No cached target mel is used at inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CACHE_DIR, METADATA_CLEAN, OUTPUTS
from src.dl.config import device
from src.dl.mel_stats import MelStats
from src.dl.train import load_checkpoint
from src.dl.vocab import PhoneVocab
from src.utils import read_filelist


@torch.no_grad()
def predict_mel(model, phone_ids, durations, sample_steps=None):
    """Return a normalised mel; input durations must already be positive integers."""
    mask = phone_ids != 0
    enc = model.encoder(phone_ids, mask)
    frames, frame_mask = model.regulator(enc, durations)
    if model.spec["system"] == "D1":
        return model.generator(frames, frame_mask)
    return model.generator.sample(frames, frame_mask, steps=sample_steps)


def prepare_clips(clip_ids, vocab, duration_source, *, phones_csv=CACHE_DIR / "phones.csv",
                  metadata_csv=METADATA_CLEAN, predictions_csv=OUTPUTS / "durations/test_pred.csv"):
    """Validate every input before generating the first clip; preserve phone order."""
    from src.audio import adjust_durations
    phones = pd.read_csv(phones_csv).sort_values(["clip_id", "idx"])
    groups = dict(tuple(phones.groupby("clip_id")))
    metadata = pd.read_csv(metadata_csv).set_index("id")
    if not metadata.index.is_unique:
        raise ValueError("metadata contains duplicate clip IDs")
    predictions = None
    if duration_source == "gbt":
        predictions = pd.read_csv(predictions_csv)
        if predictions.duplicated(["clip_id", "idx"]).any():
            raise ValueError("duration predictions contain duplicate phone indices")
    clips = []
    for clip_id in clip_ids:
        if clip_id not in groups or clip_id not in metadata.index:
            raise ValueError(f"missing alignment/metadata for {clip_id}")
        group = groups[clip_id]
        ids = vocab.encode(group["phone"].tolist())
        durations = group["n_frames"].to_numpy(dtype=np.int64)
        target = int(metadata.loc[clip_id, "n_frames"])
        if (durations < 1).any() or durations.sum() != target:
            raise ValueError(f"invalid MFA durations for {clip_id}")
        if predictions is not None:
            pred = predictions[predictions["clip_id"] == clip_id].sort_values("idx")
            if (len(pred) != len(group)
                    or not np.array_equal(pred["idx"].to_numpy(), group["idx"].to_numpy())
                    or not np.array_equal(pred["phone"].to_numpy(), group["phone"].to_numpy())):
                raise ValueError(f"duration predictions do not match aligned phones for {clip_id}")
            values = pred["pred_best"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"nonpositive/nonfinite predicted durations for {clip_id}")
            durations = adjust_durations(values, target)
        clips.append(dict(clip_id=clip_id, ids=ids, durations=durations,
                          n_frames=target, n_samples=int(metadata.loc[clip_id, "n_samples"])))
    return clips


@torch.no_grad()
def synthesize(model, clips, stats, output_dir, *, duration_source="gbt",
               target_device="cpu", sample_steps=None):
    from src.audio import adjust_durations, mel_to_wav, save_wav, trim_wav
    model.to(target_device).eval()
    if duration_source == "neural" and model.duration_predictor is None:
        raise ValueError("--durations neural requires a jointly trained duration predictor")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        ids = torch.tensor([clip["ids"]], dtype=torch.long, device=target_device)
        durations = clip["durations"]
        if duration_source == "neural":
            enc = model.encoder(ids, ids != 0)
            log_d = model.duration_predictor(enc, ids != 0)[0]
            # Bound extreme outputs before exponentiating to avoid overflow.
            values = log_d.clamp(max=np.log(clip["n_frames"])).exp().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite neural durations for {clip['clip_id']}")
            durations = adjust_durations(values, clip["n_frames"])
        durations = torch.as_tensor(durations, dtype=torch.long, device=target_device)[None, :]
        mel = predict_mel(model, ids, durations, sample_steps)[0].cpu()
        raw_mel = stats.denormalise(mel)  # HiFi-GAN must receive RAW log-mels.
        wav = trim_wav(mel_to_wav(raw_mel), clip["n_samples"])
        path = output_dir / f"{clip['clip_id']}.wav"
        save_wav(path, wav)
        print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--durations", choices=["gbt", "mfa", "neural"], default="gbt")
    parser.add_argument("--split", default="test_100")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=device())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-steps", type=int)
    args = parser.parse_args()
    model, state = load_checkpoint(args.checkpoint, target_device=args.device)
    if state["metadata"].get("overfit"):
        parser.error("use a full-training checkpoint for held-out evaluation, not the ten-clip sanity model")
    clips = prepare_clips(read_filelist(args.split), PhoneVocab.load(), args.durations)
    # Keep oracle-duration ablations from silently overwriting standard results.
    name = model.spec["system"] + ("_oracle" if args.durations == "mfa" else "")
    synthesize(model, clips, MelStats.load(), args.output_dir or OUTPUTS / "audio" / name,
               duration_source=args.durations, target_device=args.device,
               sample_steps=args.sample_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
