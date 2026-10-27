"""Phone-duration models  (M3 — step 2).

Three models, each predicting one phone's duration in mel frames:

  mean   per-phone mean of log duration. The baseline everything must beat.
  ridge  one-hot categoricals + scaled numerics -> Ridge. alpha picked on val.
  gbt    ordinal categoricals -> HistGradientBoostingRegressor (native splits
         on categories, so no one-hot blow-up).

All three regress log(n_frames) rather than n_frames: durations are strongly
right-skewed (silence runs to 481 frames against a median of 9), and a squared
error on raw frames would spend most of its budget on long pauses. This is also
what FastSpeech 2's duration predictor does, which keeps us comparable to the
paper the project is anchored to.

pred_best is whichever model has the lowest MAE on val — picked on val, never
on test.

Predictions are written as floats and deliberately left unrounded: A2 calls
adjust_durations (src/audio.py) which rounds, clamps to >= 1 frame and makes
each clip sum to its true n_frames.

Usage:
    python -m src.baseline.duration_models

Requires (from M2):
    data/cache/phones.csv

Produces:
    outputs/durations/test_pred.csv  — clip_id, idx, phone, n_frames,
                                       pred_mean, pred_ridge, pred_gbt, pred_best
    outputs/durations/val_pred.csv   — same columns, for model selection and
                                       for src/eval/duration_mae.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.baseline.duration_features import (
    CATEGORICAL,
    NUMERIC,
    add_context,
    features_from_context,
)
from src.config import CACHE_DIR, OUTPUTS, SECONDS_PER_FRAME, SEED
from src.utils import read_filelist, set_seed

PHONES_CSV = CACHE_DIR / "phones.csv"
DURATIONS_DIR = OUTPUTS / "durations"
TEST_PRED_CSV = DURATIONS_DIR / "test_pred.csv"
VAL_PRED_CSV = DURATIONS_DIR / "val_pred.csv"

MODELS = ("mean", "ridge", "gbt")
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ID_COLUMNS = ["clip_id", "idx", "phone", "n_frames"]
MS_PER_FRAME = SECONDS_PER_FRAME * 1000


# ---------------------------------------------------------------------------
# The per-phone mean baseline
# ---------------------------------------------------------------------------


def fit_phone_mean(phone: pd.Series, y: pd.Series) -> dict:
    """Per-phone mean of log duration, with a global fallback for unseen labels."""
    logd = np.log(y.to_numpy())
    table = pd.Series(logd, index=phone.to_numpy()).groupby(level=0).mean()
    return {"table": table, "global": float(logd.mean())}


def predict_phone_mean(model: dict, phone: pd.Series) -> np.ndarray:
    log_pred = phone.map(model["table"]).fillna(model["global"]).to_numpy(dtype=float)
    return np.exp(log_pred)


# ---------------------------------------------------------------------------
# The two learned models
# ---------------------------------------------------------------------------


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])),
        ("model", Ridge(alpha=alpha)),
    ])


def make_gbt() -> Pipeline:
    # unknown_value=nan keeps unseen categories out of the negative integers,
    # which HistGradientBoosting rejects; it treats nan as a missing category.
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                dtype=np.float64,
            ), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ])),
        ("model", HistGradientBoostingRegressor(
            categorical_features=list(range(len(CATEGORICAL))),
            max_iter=1800,   # converges around 1050; early stopping decides, not the budget
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )),
    ])


def mae_frames(pred: np.ndarray, truth: pd.Series) -> float:
    return float(np.abs(pred - truth.to_numpy()).mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_all(
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xva: pd.DataFrame, yva: pd.Series,
) -> tuple[dict, dict]:
    """Fit all three models. -> (fitted models, val MAE in frames per model)."""
    fitted, val_mae = {}, {}

    print("fitting mean  ...", flush=True)
    fitted["mean"] = fit_phone_mean(Xtr["phone"], ytr)
    val_mae["mean"] = mae_frames(predict_phone_mean(fitted["mean"], Xva["phone"]), yva)

    log_ytr = np.log(ytr.to_numpy())

    print("fitting ridge ...", flush=True)
    best = (np.inf, None, None)
    for alpha in RIDGE_ALPHAS:
        pipe = make_ridge(alpha).fit(Xtr, log_ytr)
        m = mae_frames(np.exp(pipe.predict(Xva)), yva)
        print(f"  alpha={alpha:<6} val MAE {m:.4f} frames", flush=True)
        if m < best[0]:
            best = (m, alpha, pipe)
    val_mae["ridge"], alpha_best, fitted["ridge"] = best
    print(f"  chose alpha={alpha_best}", flush=True)

    print("fitting gbt   ...", flush=True)
    pipe = make_gbt().fit(Xtr, log_ytr)
    fitted["gbt"] = pipe
    val_mae["gbt"] = mae_frames(np.exp(pipe.predict(Xva)), yva)
    print(f"  stopped after {pipe['model'].n_iter_} iterations", flush=True)

    return fitted, val_mae


def predict_all(fitted: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "mean": predict_phone_mean(fitted["mean"], X["phone"]),
        "ridge": np.exp(fitted["ridge"].predict(X)),
        "gbt": np.exp(fitted["gbt"].predict(X)),
    }


def prediction_frame(df: pd.DataFrame, preds: dict[str, np.ndarray], best: str) -> pd.DataFrame:
    """The on-disk contract M4 reads: ID columns + one column per model + pred_best."""
    out = df[ID_COLUMNS].copy()
    for name in MODELS:
        out[f"pred_{name}"] = preds[name]
    out["pred_best"] = preds[best]
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    if not PHONES_CSV.exists():
        print(
            f"missing {PHONES_CSV}\n"
            "Get it from the team Drive, or run: make mfa-setup lexicon align-core align-rest phones",
            file=sys.stderr,
        )
        return 1

    set_seed(SEED)
    phones = pd.read_csv(PHONES_CSV, dtype={"clip_id": str, "word": str})
    df = add_context(phones)

    splits = {name: set(read_filelist(name)) for name in ("train", "val", "test")}
    parts = {name: df[df["clip_id"].isin(ids)] for name, ids in splits.items()}
    for name, part in parts.items():
        print(f"{name:6s} {part['clip_id'].nunique():6d} clips  {len(part):8,} phones")

    built = {name: features_from_context(part) for name, part in parts.items()}
    (Xtr, ytr), (Xva, yva), (Xte, yte) = built["train"], built["val"], built["test"]

    print()
    fitted, val_mae = train_all(Xtr, ytr, Xva, yva)

    best = min(MODELS, key=lambda m: val_mae[m])
    print("\nval MAE (frames) — model selection:")
    for name in MODELS:
        mark = "  <- best" if name == best else ""
        print(f"  {name:6s} {val_mae[name]:.4f}  = {val_mae[name] * MS_PER_FRAME:5.2f} ms{mark}")

    DURATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (X, _), path in (
        ("val", built["val"], VAL_PRED_CSV),
        ("test", built["test"], TEST_PRED_CSV),
    ):
        frame = prediction_frame(parts[name], predict_all(fitted, X), best)
        frame.to_csv(path, index=False)
        print(f"\nwrote {path.relative_to(OUTPUTS.parent)}  ({len(frame):,} rows, pred_best = {best})")

    print("\nNext: python -m src.eval.duration_mae")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
