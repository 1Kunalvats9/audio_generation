"""Duration-model metrics  (M3 — step 3).

Scores the predictions written by src/baseline/duration_models.py. Every number
is reported twice, including and excluding silence, because `sil` is only 5.3%
of rows but has mean 14.6 frames against 7.4 for real phones (and runs to 481).
One long pause can move the headline MAE on its own, so the excluding-silence
column is the honest measure of whether the model learned phone durations,
while the including-silence column is what A2's synthesis actually depends on.

Usage:
    python -m src.eval.duration_mae

Requires (from M3):
    outputs/durations/val_pred.csv, outputs/durations/test_pred.csv

Produces:
    outputs/results/duration_models.csv — the model comparison for the report:
        model, split, mae_frames, mae_ms, mae_ms_nosil, rmse_frames, n_phones
    outputs/results/mae.csv — the two rows M5's make_results.py joins on:
        system, model, mae_ms, mae_frames
        A1 uses the true MFA durations, so its duration MAE is 0 by definition;
        A2 uses pred_best, so it carries the real error.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.baseline.duration_models import MODELS, TEST_PRED_CSV, VAL_PRED_CSV
from src.config import OUTPUTS, SECONDS_PER_FRAME, SIL

RESULTS_DIR = OUTPUTS / "results"
MODELS_CSV = RESULTS_DIR / "duration_models.csv"
MAE_CSV = RESULTS_DIR / "mae.csv"
MS_PER_FRAME = SECONDS_PER_FRAME * 1000


def score(pred: np.ndarray, truth: np.ndarray, is_sil: np.ndarray) -> dict:
    """MAE / RMSE for one model on one split."""
    err = np.abs(pred - truth)
    speech = ~is_sil
    return {
        "mae_frames": float(err.mean()),
        "mae_ms": float(err.mean() * MS_PER_FRAME),
        "mae_ms_nosil": float(err[speech].mean() * MS_PER_FRAME),
        "rmse_frames": float(np.sqrt(((pred - truth) ** 2).mean())),
        "n_phones": int(len(err)),
    }


def score_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """-> one row per (model, split)."""
    rows = []
    for split, df in frames.items():
        truth = df["n_frames"].to_numpy(dtype=float)
        is_sil = df["phone"].eq(SIL).to_numpy()
        for name in MODELS:
            rows.append({"model": name, "split": split,
                         **score(df[f"pred_{name}"].to_numpy(dtype=float), truth, is_sil)})
    return pd.DataFrame(rows)


def best_model(table: pd.DataFrame) -> str:
    """The model pred_best was taken from: lowest MAE on val."""
    val = table[table["split"] == "val"]
    return str(val.loc[val["mae_frames"].idxmin(), "model"])


def system_table(table: pd.DataFrame, best: str) -> pd.DataFrame:
    """The A1 / A2 rows M5's make_results.py reads."""
    test = table[(table["split"] == "test") & (table["model"] == best)].iloc[0]
    return pd.DataFrame([
        {"system": "A1", "model": "mfa_true", "mae_ms": 0.0, "mae_frames": 0.0},
        {"system": "A2", "model": best,
         "mae_ms": round(float(test["mae_ms"]), 4),
         "mae_frames": round(float(test["mae_frames"]), 4)},
    ])


def report(table: pd.DataFrame, best: str) -> None:
    print(f"{MS_PER_FRAME:.3f} ms per frame\n")
    print(f"{'model':<7}{'split':<7}{'MAE frames':>12}{'MAE ms':>10}"
          f"{'MAE ms (no sil)':>18}{'RMSE frames':>13}")
    for _, r in table.iterrows():
        print(f"{r['model']:<7}{r['split']:<7}{r['mae_frames']:>12.4f}{r['mae_ms']:>10.2f}"
              f"{r['mae_ms_nosil']:>18.2f}{r['rmse_frames']:>13.4f}")

    base = table[(table["split"] == "test") & (table["model"] == "mean")].iloc[0]["mae_frames"]
    win = table[(table["split"] == "test") & (table["model"] == best)].iloc[0]["mae_frames"]
    print(f"\npred_best = {best} (chosen on val)")
    print(f"test MAE {win:.4f} frames vs {base:.4f} for the per-phone mean "
          f"-> {(1 - win / base) * 100:.1f}% better")


def main() -> int:
    missing = [p for p in (VAL_PRED_CSV, TEST_PRED_CSV) if not p.exists()]
    if missing:
        print("missing " + ", ".join(str(p) for p in missing)
              + "\nRun: python -m src.baseline.duration_models", file=sys.stderr)
        return 1

    frames = {
        "val": pd.read_csv(VAL_PRED_CSV, dtype={"clip_id": str}),
        "test": pd.read_csv(TEST_PRED_CSV, dtype={"clip_id": str}),
    }
    table = score_table(frames)
    best = best_model(table)
    report(table, best)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(MODELS_CSV, index=False)
    system_table(table, best).to_csv(MAE_CSV, index=False)
    print(f"\nwrote {MODELS_CSV.relative_to(OUTPUTS.parent)}")
    print(f"wrote {MAE_CSV.relative_to(OUTPUTS.parent)}  (read by src/eval/make_results.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
