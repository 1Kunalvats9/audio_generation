"""Small shared helpers. Keep this file boring and dependency-light."""

import random
from pathlib import Path

import numpy as np

from src.config import FILELISTS, SEED


def set_seed(seed: int = SEED) -> None:
    """Make random choices repeatable across all five members' scripts."""
    random.seed(seed)
    np.random.seed(seed)


def read_filelist(name: str) -> list[str]:
    """read_filelist("test_100") -> ["LJ005-0001", ...] (one clip ID per line)."""
    path = FILELISTS / f"{name}.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_filelist(name: str, ids) -> Path:
    """Write one clip ID per line, sorted, with a trailing newline."""
    FILELISTS.mkdir(parents=True, exist_ok=True)
    path = FILELISTS / f"{name}.txt"
    with open(path, "w") as f:
        for i in sorted(ids):
            f.write(f"{i}\n")
    return path


def prefix_of(clip_id: str) -> str:
    """"LJ005-0017" -> "LJ005" (the source text it came from)."""
    return clip_id.split("-")[0]
