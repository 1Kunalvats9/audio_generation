"""Automated cross-check of the failure log's guessed causes.

M5 annotates outputs/results/failure_log.csv by ear, using notebooks/03_demo.ipynb. This script
is an independent, reproducible second opinion derived only from measurable signals, written to
outputs/results/failure_log_auto.csv so it never overwrites that hand annotation.

TEAM.md asks for the 10 worst clips per system, each tagged with a guessed cause from:

    flat prosody · wrong duration · slurred transition · misaligned phone · vocoder artefact

Those five describe *synthesis* failures. R0 is the real recording, so by definition none of
them applies to it; its errors come from the reference text or from Whisper itself. We add two
honest labels for that case rather than forcing a clip into a category that cannot be true:

    reference normalisation — ref and hyp mean the same thing but are written differently
                              (e.g. "george the 4th" vs "george iv"), so it is scored wrong
    asr error               — Whisper simply misheard clean audio

Each tag is assigned from measurable evidence, not by ear, and the evidence column records why.

Usage:
    python -m src.eval.tag_failures

Reads:  outputs/results/failure_log.csv, wer_R0.csv, outputs/durations/test_pred.csv,
        data/cache/phones.csv
Writes: outputs/results/failure_log.csv  (guessed_cause + evidence filled in)
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from src.audio import adjust_durations
from src.config import CACHE_DIR, METADATA_CLEAN, OUTPUTS, SECONDS_PER_FRAME

RESULTS = OUTPUTS / "results"
FAILURE_LOG = RESULTS / "failure_log.csv"          # M5's hand-annotated deliverable (read-only here)
AUTO_LOG = RESULTS / "failure_log_auto.csv"        # this script's independent, reproducible pass
TEST_PRED = OUTPUTS / "durations" / "test_pred.csv"
MS = SECONDS_PER_FRAME * 1000

NUMBERISH = re.compile(r"^(\d+(st|nd|rd|th)?|[ivxlc]+)$", re.I)


def number_token_mismatch(ref: str, hyp: str) -> bool:
    """True when the differing tokens are numerals on one side and numerals on the other.

    Catches the normaliser artifact: the reference writes "george the 4th", Whisper hears
    "george iv". Same meaning, scored as two errors.
    """
    r, h = ref.split(), hyp.split()
    only_r = [w for w in r if w not in h]
    only_h = [w for w in h if w not in r]
    if not only_r or not only_h:
        return False
    return any(NUMBERISH.match(w) for w in only_r) and any(NUMBERISH.match(w) for w in only_h)


def per_clip_signals() -> pd.DataFrame:
    """Evidence we can measure for every test clip."""
    phones = pd.read_csv(CACHE_DIR / "phones.csv", dtype={"clip_id": str, "word": str})
    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str}).set_index("id")
    pred = pd.read_csv(TEST_PRED, dtype={"clip_id": str})

    rows = []
    for cid, g in pred.groupby("clip_id"):
        g = g.sort_values("idx")
        true = g["n_frames"].to_numpy(dtype=float)
        adj = adjust_durations(g["pred_best"].to_numpy(np.float32), int(meta.loc[cid, "n_frames"]))
        rows.append({
            "clip_id": cid,
            "dur_mae_ms": float(np.abs(adj - true).mean() * MS),
            "short_true": float((true <= 2).mean()),
            "short_pred": float((adj <= 2).mean()),
            "longest_pred": int(adj.max()),
        })
    sig = pd.DataFrame(rows).set_index("clip_id")

    # how crushed the templates are on this clip, for A1 (which uses the true durations)
    tp = phones[phones["clip_id"].isin(sig.index)]
    sig["short_true_all"] = tp.assign(s=tp.n_frames.le(2)).groupby("clip_id")["s"].mean()
    return sig


def tag(row: pd.Series, sig: pd.DataFrame, r0_failed: set[str], thresholds: dict) -> tuple[str, str]:
    """-> (guessed_cause, evidence)."""
    cid, system = row["clip_id"], row["system"]
    ref, hyp = str(row["ref"]), str(row["hyp"])

    if number_token_mismatch(ref, hyp):
        return "reference normalisation", "differing tokens are numerals on both sides"

    if system == "R0":
        return "asr error", "real audio, no numeral mismatch: Whisper misheard"

    if system == "R1":
        if cid in r0_failed:
            return "asr error", "this clip is also among R0's worst, so not caused by the vocoder"
        return "vocoder artefact", "real mel; only the vocoder differs from R0"

    s = sig.loc[cid] if cid in sig.index else None
    if s is None:
        return "flat prosody", "templates carry no prosody or coarticulation"

    if system == "A2" and s["dur_mae_ms"] >= thresholds["dur_hi"]:
        return "wrong duration", f"clip duration MAE {s['dur_mae_ms']:.0f} ms, top quartile"

    if system == "A2" and s["longest_pred"] >= thresholds["long_hi"]:
        return "wrong duration", f"one predicted phone of {s['longest_pred']} frames after rescaling"

    short = s["short_pred"] if system == "A2" else s["short_true_all"]
    if short >= thresholds["short_hi"]:
        return "slurred transition", f"{short:.0%} of phones under 3 frames, so templates are crushed"

    return "flat prosody", "templates carry no prosody or coarticulation"


def main() -> int:
    if not FAILURE_LOG.exists():
        print(f"missing {FAILURE_LOG}\nRun: python -m src.eval.make_results", file=sys.stderr)
        return 1

    log = pd.read_csv(FAILURE_LOG, dtype={"clip_id": str})
    sig = per_clip_signals()
    r0_failed = set(log.loc[log.system == "R0", "clip_id"])

    thresholds = {
        "dur_hi": float(sig["dur_mae_ms"].quantile(0.75)),
        "long_hi": int(sig["longest_pred"].quantile(0.90)),
        "short_hi": float(sig["short_true_all"].quantile(0.75)),
    }
    print("thresholds from the test split: "
          f"duration MAE p75 = {thresholds['dur_hi']:.0f} ms, "
          f"longest phone p90 = {thresholds['long_hi']} frames, "
          f"short-phone share p75 = {thresholds['short_hi']:.1%}")

    tags = [tag(r, sig, r0_failed, thresholds) for _, r in log.iterrows()]
    log["guessed_cause"] = [t[0] for t in tags]
    log["evidence"] = [t[1] for t in tags]
    log.to_csv(AUTO_LOG, index=False)

    print(f"\ntagged {len(log)} clips -> {AUTO_LOG.relative_to(OUTPUTS.parent)}\n")
    print(pd.crosstab(log["system"], log["guessed_cause"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
