"""The phoneme vocabulary  (M1 — phase 2).

Maps MFA's phone symbols to integer ids for the embedding layer. Built once
from data/cache/phones.csv and then FROZEN: every member imports the same file,
so an embedding row always means the same phone in everyone's checkpoints.

Id 0 is always padding, so real phones start at 1. That lets us use
nn.Embedding(padding_idx=0) and makes `ids != 0` a correct padding mask.

Usage:
    python -m src.dl.vocab                 # build and write the json
    from src.dl.vocab import PhoneVocab
    vocab = PhoneVocab.load()
    ids = vocab.encode(["DH", "AH0", "sil"])
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import pandas as pd

from src.config import CACHE_DIR, ROOT
from src.dl.config import PAD_ID, PAD_SYMBOL, VOCAB_PATH

PHONES_CSV = CACHE_DIR / "phones.csv"


@dataclass(frozen=True)
class PhoneVocab:
    """Frozen phone symbol <-> id mapping."""

    symbols: tuple[str, ...]          # index == id, symbols[PAD_ID] is the pad symbol

    @property
    def size(self) -> int:
        return len(self.symbols)

    def encode(self, phones) -> list[int]:
        """["DH", "AH0"] -> [12, 3].  Unknown symbols raise rather than silently pad."""
        index = self._index()
        missing = sorted({p for p in phones if p not in index})
        if missing:
            raise KeyError(f"phone(s) not in the frozen vocabulary: {missing}")
        return [index[p] for p in phones]

    def decode(self, ids) -> list[str]:
        return [self.symbols[i] for i in ids]

    def _index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}

    # -- persistence ---------------------------------------------------------

    def save(self, path=VOCAB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"pad_symbol": PAD_SYMBOL, "pad_id": PAD_ID,
                       "symbols": list(self.symbols)}, f, indent=2)

    @classmethod
    def load(cls, path=VOCAB_PATH) -> "PhoneVocab":
        if not path.exists():
            raise FileNotFoundError(f"missing {path}\nRun: python -m src.dl.vocab")
        with open(path) as f:
            d = json.load(f)
        assert d["pad_id"] == PAD_ID, "pad id in the vocab file disagrees with configs/dl.yaml"
        assert d["symbols"][PAD_ID] == PAD_SYMBOL, "pad symbol is not at pad_id"
        return cls(symbols=tuple(d["symbols"]))

    @classmethod
    def build(cls, phones: pd.DataFrame) -> "PhoneVocab":
        """Every distinct phone in phones.csv, sorted, with padding first."""
        found = sorted(phones["phone"].astype(str).unique())
        assert PAD_SYMBOL not in found, f"{PAD_SYMBOL!r} collides with a real phone symbol"
        return cls(symbols=(PAD_SYMBOL, *found))


def main() -> int:
    if not PHONES_CSV.exists():
        print(f"missing {PHONES_CSV}\nRun: make phones  (or git pull — it is committed)",
              file=sys.stderr)
        return 1

    phones = pd.read_csv(PHONES_CSV, usecols=["phone"])
    vocab = PhoneVocab.build(phones)
    vocab.save()

    print(f"{vocab.size} ids ({vocab.size - 1} phones + padding)")
    print(f"  id {PAD_ID} = {PAD_SYMBOL!r}")
    print(f"  id 1 = {vocab.symbols[1]!r} ... id {vocab.size - 1} = {vocab.symbols[-1]!r}")
    print(f"\nwrote {VOCAB_PATH.relative_to(ROOT)}")
    print("This file is FROZEN. Changing it invalidates every checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
