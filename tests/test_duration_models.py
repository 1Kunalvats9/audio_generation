"""M3 steps 2 and 3: the models, the on-disk contract, and the metrics."""

import numpy as np
import pandas as pd
import pytest

from src.baseline.duration_features import build_features
from src.baseline.duration_models import (
    ID_COLUMNS,
    MODELS,
    fit_phone_mean,
    make_gbt,
    make_ridge,
    mae_frames,
    predict_all,
    predict_phone_mean,
    prediction_frame,
)
from src.config import SIL
from src.eval.duration_mae import best_model, score, score_table, system_table
from tests.fake_phones import fake_phones

# The columns TEAM.md promises M4, which synthesize_baseline.py reads.
M4_CONTRACT = ["clip_id", "idx", "phone", "n_frames",
               "pred_mean", "pred_ridge", "pred_gbt", "pred_best"]


# ---------------------------------------------------------------------------
# The per-phone mean baseline
# ---------------------------------------------------------------------------


def test_phone_mean_recovers_a_single_observation():
    phone = pd.Series(["S", "T"])
    y = pd.Series([10.0, 20.0])
    model = fit_phone_mean(phone, y)
    pred = predict_phone_mean(model, phone)
    assert pred == pytest.approx([10.0, 20.0])


def test_phone_mean_is_geometric_because_it_averages_logs():
    phone = pd.Series(["S", "S"])
    y = pd.Series([4.0, 16.0])
    model = fit_phone_mean(phone, y)
    # mean of log(4), log(16) -> exp = 8, not the arithmetic mean of 10
    assert predict_phone_mean(model, phone)[0] == pytest.approx(8.0)


def test_phone_mean_falls_back_for_an_unseen_phone():
    model = fit_phone_mean(pd.Series(["S", "T"]), pd.Series([10.0, 20.0]))
    pred = predict_phone_mean(model, pd.Series(["ZZZ"]))
    assert np.isfinite(pred).all()
    assert pred[0] == pytest.approx(np.exp(np.mean(np.log([10.0, 20.0]))))


def test_phone_mean_predictions_are_positive():
    phones = fake_phones(4)
    X, y = build_features(phones)
    model = fit_phone_mean(X["phone"], y)
    assert (predict_phone_mean(model, X["phone"]) > 0).all()


# ---------------------------------------------------------------------------
# The learned models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: make_ridge(1.0), make_gbt])
def test_pipelines_fit_and_predict_positive_durations(make):
    X, y = build_features(fake_phones(40))
    pipe = make().fit(X, np.log(y.to_numpy()))
    pred = np.exp(pipe.predict(X))
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_gbt_tolerates_an_unseen_category():
    X, y = build_features(fake_phones(40))
    pipe = make_gbt().fit(X, np.log(y.to_numpy()))
    unseen = X.iloc[:1].copy()
    unseen["phone"] = "ZZZ"
    assert np.isfinite(pipe.predict(unseen)).all()


def test_ridge_tolerates_an_unseen_category():
    X, y = build_features(fake_phones(40))
    pipe = make_ridge(1.0).fit(X, np.log(y.to_numpy()))
    unseen = X.iloc[:1].copy()
    unseen["phone"] = "ZZZ"
    assert np.isfinite(pipe.predict(unseen)).all()


def test_mae_frames():
    assert mae_frames(np.array([1.0, 5.0]), pd.Series([2.0, 3.0])) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The contract M4 reads
# ---------------------------------------------------------------------------


def _fitted_and_frame(best="gbt"):
    phones = fake_phones(40)
    X, y = build_features(phones)
    log_y = np.log(y.to_numpy())
    fitted = {
        "mean": fit_phone_mean(X["phone"], y),
        "ridge": make_ridge(1.0).fit(X, log_y),
        "gbt": make_gbt().fit(X, log_y),
    }
    df = phones.sort_values(["clip_id", "idx"]).reset_index(drop=True)
    return prediction_frame(df, predict_all(fitted, X), best)


def test_prediction_frame_matches_the_m4_contract():
    assert list(_fitted_and_frame().columns) == M4_CONTRACT


def test_pred_best_mirrors_the_chosen_model():
    frame = _fitted_and_frame(best="ridge")
    assert frame["pred_best"].equals(frame["pred_ridge"])
    assert not frame["pred_best"].equals(frame["pred_gbt"])


def test_predictions_are_left_unrounded_for_adjust_durations():
    frame = _fitted_and_frame()
    for name in MODELS:
        col = frame[f"pred_{name}"]
        assert col.dtype.kind == "f"
        assert (col > 0).all()
    # floats, not integers: A2's adjust_durations does the rounding
    assert not np.allclose(frame["pred_gbt"], frame["pred_gbt"].round())


def test_id_columns_survive_untouched():
    phones = fake_phones(5)
    X, _ = build_features(phones)
    df = phones.sort_values(["clip_id", "idx"]).reset_index(drop=True)
    model = {"mean": fit_phone_mean(X["phone"], df["n_frames"].astype(float))}
    preds = {m: np.ones(len(df)) for m in MODELS}
    frame = prediction_frame(df, preds, "mean")
    pd.testing.assert_frame_equal(frame[ID_COLUMNS], df[ID_COLUMNS])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_score_separates_silence():
    pred = np.array([1.0, 1.0, 10.0])
    truth = np.array([2.0, 2.0, 2.0])
    is_sil = np.array([False, False, True])
    s = score(pred, truth, is_sil)
    assert s["mae_frames"] == pytest.approx((1 + 1 + 8) / 3)
    assert s["mae_ms_nosil"] == pytest.approx(1.0 * 11.609977324263039)
    assert s["n_phones"] == 3


def test_perfect_predictions_score_zero():
    truth = np.array([3.0, 4.0])
    s = score(truth.copy(), truth, np.array([False, False]))
    assert s["mae_frames"] == 0.0 and s["rmse_frames"] == 0.0


def _score_frames():
    rows = []
    for split in ("val", "test"):
        for i, name in enumerate(MODELS):
            rows.append({"clip_id": "a", "idx": i, "phone": SIL if i == 0 else "S",
                         "n_frames": 5.0, "pred_mean": 9.0, "pred_ridge": 7.0,
                         "pred_gbt": 5.5, "pred_best": 5.5, "split": split})
    df = pd.DataFrame(rows)
    return {s: df[df.split == s].drop(columns="split") for s in ("val", "test")}


def test_score_table_covers_every_model_and_split():
    table = score_table(_score_frames())
    assert len(table) == len(MODELS) * 2
    assert set(table["model"]) == set(MODELS)
    assert set(table["split"]) == {"val", "test"}


def test_best_model_is_chosen_on_val():
    table = score_table(_score_frames())
    assert best_model(table) == "gbt"


def test_system_table_is_what_make_results_joins_on():
    table = score_table(_score_frames())
    sysr = system_table(table, "gbt")
    # M5's load_duration_mae() needs exactly these two column names
    assert {"system", "mae_ms"} <= set(sysr.columns)
    assert list(sysr["system"]) == ["A1", "A2"]
    # A1 uses the true MFA durations, so it has no duration error by definition
    assert sysr.loc[0, "mae_ms"] == 0.0
    assert sysr.loc[1, "mae_ms"] > 0.0
