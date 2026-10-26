"""Shared settings, loaded once from configs/base.yaml.

Every script imports from here instead of hard-coding numbers:

    from src.config import SAMPLE_RATE, HOP_LENGTH, MEL_KWARGS, n_frames

Run scripts from the repo root, e.g. `python -m src.data.prepare_metadata`.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "configs" / "base.yaml") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LJ_DIR = ROOT / CFG["paths"]["raw"]
WAV_DIR = LJ_DIR / "wavs"
METADATA_RAW = LJ_DIR / "metadata.csv"
METADATA_CLEAN = ROOT / CFG["paths"]["metadata_clean"]
FILELISTS = ROOT / "filelists"
CACHE_DIR = ROOT / CFG["paths"]["cache"]
MEL_DIR = CACHE_DIR / "mels"
OUTPUTS = ROOT / CFG["paths"]["outputs"]
FIGURES = OUTPUTS / "figures"

# ---------------------------------------------------------------------------
# Audio / mel
# ---------------------------------------------------------------------------

_A = CFG["audio"]

SAMPLE_RATE = _A["sample_rate"]
N_FFT = _A["n_fft"]
WIN_LENGTH = _A["win_length"]
HOP_LENGTH = _A["hop_length"]
N_MELS = _A["n_mels"]
F_MIN = _A["f_min"]
F_MAX = _A["f_max"]
LOG_CLAMP_MIN = _A["log_clamp_min"]

# Pass straight into torchaudio.transforms.MelSpectrogram(**MEL_KWARGS).
# Keeping it in one dict means M4's mels and the vocoder cannot drift apart.
MEL_KWARGS = dict(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=_A["power"],
    norm=_A["norm"],
    mel_scale=_A["mel_scale"],
    center=_A["center"],
)

SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE   # 0.0116 s

# ---------------------------------------------------------------------------
# Text / phonemes
# ---------------------------------------------------------------------------

SIL = CFG["text"]["silence_symbol"]

# ---------------------------------------------------------------------------
# Splits and MFA
# ---------------------------------------------------------------------------

TEST_PREFIXES = CFG["splits"]["test_prefixes"]
N_VAL = CFG["splits"]["n_val"]
N_TEST_100 = CFG["splits"]["n_test_100"]

PILOT_CLIPS = CFG["mfa"]["pilot_clips"]
CORE_TRAIN_CLIPS = CFG["mfa"]["core_train_clips"]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

VOCODER_SOURCE = CFG["vocoder"]["source"]
VOCODER_SAVEDIR = ROOT / CFG["vocoder"]["savedir"]
WHISPER_MODEL = CFG["asr"]["whisper_model"]

# ---------------------------------------------------------------------------
# The two frame formulas. Import these; do not re-implement them.
# ---------------------------------------------------------------------------


def n_frames(n_samples: int) -> int:
    """Number of mel frames for a clip of n_samples (center=True padding)."""
    return 1 + n_samples // HOP_LENGTH


def sec_to_frame(t: float) -> int:
    """A time in seconds -> the mel frame index it falls in."""
    return round(t * SAMPLE_RATE / HOP_LENGTH)


def frame_to_sec(idx: int) -> float:
    """A mel frame index -> its time in seconds."""
    return idx * HOP_LENGTH / SAMPLE_RATE


if __name__ == "__main__":
    print(f"ROOT            {ROOT}")
    print(f"sample_rate     {SAMPLE_RATE}")
    print(f"hop_length      {HOP_LENGTH}  ({SECONDS_PER_FRAME * 1000:.2f} ms/frame)")
    print(f"n_mels          {N_MELS}")
    print(f"silence symbol  {SIL!r}")
    print(f"LJSpeech found  {METADATA_RAW.exists()}")
    print(f"n_frames(22050) {n_frames(22050)}   sec_to_frame(1.0) {sec_to_frame(1.0)}")
