"""Phase 2 settings, loaded from configs/dl.yaml.

Phase 1's src/config.py stays the single source for audio and split settings —
import SAMPLE_RATE, HOP_LENGTH, N_MELS and the paths from there, not here.
This module only adds what the neural model needs.

Usage:
    from src.dl.config import DL, HIDDEN, BATCH_SIZE, VOCAB_PATH
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import ROOT

DL_YAML = ROOT / "configs" / "dl.yaml"
with open(DL_YAML) as f:
    DL = yaml.safe_load(f)

SEED = DL["seed"]

PAD_SYMBOL = DL["vocab"]["pad_symbol"]
PAD_ID = DL["vocab"]["pad_id"]

DEBUG_CLIPS = DL["data"]["debug_clips"]
MAX_FRAMES = DL["data"]["max_frames"]
NUM_WORKERS = DL["data"]["num_workers"]
BUCKET_BATCHES = DL["data"]["bucket_batches"]

HIDDEN = DL["model"]["hidden"]
ENCODER_LAYERS = DL["model"]["encoder_layers"]
ENCODER_KERNEL = DL["model"]["encoder_kernel"]
DECODER_LAYERS = DL["model"]["decoder_layers"]
DECODER_KERNEL = DL["model"]["decoder_kernel"]
DROPOUT = DL["model"]["dropout"]

BATCH_SIZE = DL["train"]["batch_size"]
LR = float(DL["train"]["lr"])
WEIGHT_DECAY = DL["train"]["weight_decay"]
GRAD_CLIP = DL["train"]["grad_clip"]
MAX_STEPS = DL["train"]["max_steps"]
WARMUP_STEPS = DL["train"]["warmup_steps"]

VOCAB_PATH = ROOT / DL["paths"]["vocab"]
MEL_STATS_PATH = ROOT / DL["paths"]["mel_stats"]
CHECKPOINTS = ROOT / DL["paths"]["checkpoints"]
TRAIN_LOG = ROOT / DL["paths"]["train_log"]


def device() -> str:
    """The best device available: cuda, then Apple MPS, then cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _main() -> int:
    import torch
    print(f"config    {DL_YAML.relative_to(ROOT)}")
    print(f"device    {device()}  (torch {torch.__version__})")
    print(f"hidden    {HIDDEN}   batch {BATCH_SIZE}   lr {LR}")
    print(f"vocab     {VOCAB_PATH.relative_to(ROOT)}  exists={VOCAB_PATH.exists()}")
    print(f"melstats  {MEL_STATS_PATH.relative_to(ROOT)}  exists={MEL_STATS_PATH.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
