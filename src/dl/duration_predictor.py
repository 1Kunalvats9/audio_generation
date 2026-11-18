"""Neural duration predictor  (M3 — phase 2).

FastSpeech 2's variance adaptor: predict how long each phoneme lasts from the
encoder's output, so synthesis needs no forced alignment.

    (B, P, H) encoder output  ->  (B, P) predicted LOG durations

Phase 1 solved the same task with 21 hand-engineered features and gradient
boosted trees, reaching 19.65 ms MAE on the held-out book. This module is the
learned alternative, and `python -m src.dl.duration_predictor` scores it against
that number on the same split — the comparison is the point of M3's phase 2.

Three choices carried over from phase 1 for the same reasons:

  * Predict LOG duration. Durations are strongly right-skewed (median 7 frames,
    silences to 481), so squared error on raw counts spends the model's capacity
    on a few long pauses.
  * Mask padded phonemes everywhere. A convolution reaches across its kernel, so
    without masking a clip's prediction depends on how long its batch mates
    were. Same pattern as src/dl/encoder.py.
  * Convert to integer frames with `src.audio.adjust_durations`, phase 1's
    function, which rounds, clamps to >= 1 and makes a clip sum to its target.

Usage:
    python -m src.dl.duration_predictor            # score against phase 1
    from src.dl.duration_predictor import DurationPredictor, duration_loss
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.config import HIDDEN

LOG_FLOOR = 1e-5          # a phone is at least 1 frame; this only guards log(0)


def _zero_pad(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask.unsqueeze(-1).to(x.dtype)


class DurationPredictor(nn.Module):
    """(B, P, H) -> (B, P) log-durations, zero on padded phonemes."""

    def __init__(self, hidden: int = HIDDEN, layers: int = 2, kernel: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        assert kernel % 2 == 1, "an odd kernel keeps the sequence length unchanged"
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2))
            self.norms.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, enc_out: torch.Tensor, phone_mask: torch.Tensor) -> torch.Tensor:
        x = _zero_pad(enc_out, phone_mask)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.drop(norm(F.relu(x)))
            x = _zero_pad(x, phone_mask)        # re-mask: the kernel smears into padding
        out = self.proj(x).squeeze(-1)          # (B, P)
        return out * phone_mask.to(out.dtype)


def duration_loss(pred_log: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE between predicted and true log-durations.

    `target` is in FRAMES (as the dataset provides); it is logged here so
    callers cannot accidentally compare log against linear.
    """
    target_log = torch.log(target.to(pred_log.dtype).clamp(min=LOG_FLOOR))
    err = (pred_log - target_log) ** 2 * mask.to(pred_log.dtype)
    denom = mask.sum().clamp(min=1).to(pred_log.dtype)
    return err.sum() / denom


@torch.no_grad()
def predict_frames(model: DurationPredictor, enc_out: torch.Tensor,
                   phone_mask: torch.Tensor) -> torch.Tensor:
    """Log-durations -> float frame counts (not yet rounded or made to sum)."""
    return torch.exp(model(enc_out, phone_mask)) * phone_mask.to(enc_out.dtype)


# ---------------------------------------------------------------------------
# The phase 1 comparison
# ---------------------------------------------------------------------------


def evaluate_against_phase1(checkpoint=None, split: str = "test", device: str = "cpu") -> dict:
    """MAE in ms on `split`, next to phase 1's boosted trees on the same clips."""
    import numpy as np
    import pandas as pd

    from src.audio import adjust_durations
    from src.config import METADATA_CLEAN, OUTPUTS, SECONDS_PER_FRAME
    from src.dl.dataset import make_loader
    from src.dl.encoder import build_encoder
    from src.dl.train import AcousticModel, load_checkpoint
    from src.dl.vocab import PhoneVocab

    ms = SECONDS_PER_FRAME * 1000
    model = AcousticModel(PhoneVocab.load().size, "D1", joint_durations=True).to(device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, target_device=device)
    model.eval()

    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str}).set_index("id")
    errs = []
    for batch in make_loader(split, batch_size=8, shuffle=False, num_workers=0):
        ids = batch["phone_ids"].to(device)
        pmask = batch["phone_mask"].to(device)
        with torch.no_grad():
            enc = model.encoder(ids, pmask)
            pred = predict_frames(model.duration_predictor, enc, pmask).cpu().numpy()
        for b, clip_id in enumerate(batch["clip_ids"]):
            n = int(batch["n_phones"][b])
            target = int(meta.loc[clip_id, "n_frames"])
            adj = adjust_durations(pred[b, :n].astype(np.float32), target)
            errs.append(np.abs(adj - batch["durations"][b, :n].numpy()))

    err = np.concatenate(errs)
    result = {"mae_frames": float(err.mean()), "mae_ms": float(err.mean() * ms),
              "n_phones": int(err.size)}

    table = OUTPUTS / "results" / "duration_models.csv"
    if table.exists():
        df = pd.read_csv(table)
        row = df[(df.split == split) & (df.model == "gbt")]
        if not row.empty:
            result["phase1_gbt_ms"] = float(row.iloc[0]["mae_ms"])
    return result


def main() -> int:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, help="a --joint-durations checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.checkpoint is None:
        print("no --checkpoint given: scoring an untrained predictor, which is "
              "only a wiring check.", file=sys.stderr)

    r = evaluate_against_phase1(args.checkpoint, args.split, args.device)
    print(f"neural duration predictor on {args.split}: "
          f"{r['mae_frames']:.3f} frames = {r['mae_ms']:.2f} ms  ({r['n_phones']:,} phones)")
    if "phase1_gbt_ms" in r:
        delta = (1 - r["mae_ms"] / r["phase1_gbt_ms"]) * 100
        verdict = "better than" if delta > 0 else "worse than"
        print(f"phase 1 boosted trees:        {r['phase1_gbt_ms']:.2f} ms")
        print(f"-> the neural predictor is {abs(delta):.1f}% {verdict} phase 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
