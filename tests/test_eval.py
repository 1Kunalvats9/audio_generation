"""Tests for M5 evaluation harness (whisper_wer.py and make_results.py).

Run:
    python -m pytest tests/test_eval.py
"""

import pandas as pd
import pytest

from src.eval.make_results import build_failure_log, build_summary_table
from src.eval.whisper_wer import compute_metrics, get_normalizer


def test_normalizer_behavior():
    normalizer = get_normalizer()
    text = "Hello, Mr. Smith! The year is 1844."
    normalized = normalizer(text)
    # Whisper's EnglishTextNormalizer lowercases and expands/cleans standard punctuation
    assert "hello" in normalized
    assert "!" not in normalized


def test_compute_metrics_exact_match():
    wer, cer = compute_metrics("hello world", "hello world")
    assert wer == 0.0
    assert cer == 0.0


def test_compute_metrics_mismatch():
    wer, cer = compute_metrics("the cat sat", "the dog sat")
    assert wer == pytest.approx(1 / 3)
    assert cer > 0.0


def test_compute_metrics_empty():
    wer, cer = compute_metrics("", "")
    assert wer == 0.0
    assert cer == 0.0

    wer_empty_ref, _ = compute_metrics("", "something")
    assert wer_empty_ref == 1.0


def test_build_summary_table():
    df_r0 = pd.DataFrame([
        {"clip_id": "LJ005-0001", "system": "R0", "ref": "sample text", "hyp": "sample text", "wer": 0.0, "cer": 0.0},
        {"clip_id": "LJ005-0002", "system": "R0", "ref": "second sample", "hyp": "second sample", "wer": 0.0, "cer": 0.0},
    ])
    wer_dfs = {"R0": df_r0}
    mae_dict = {"R0": 0.0}

    summary = build_summary_table(wer_dfs, mae_dict)
    assert "System" in summary.columns
    assert "Mean WER (%)" in summary.columns
    assert len(summary) == 4  # R0, R1, A1, A2
    r0_row = summary[summary["System"] == "R0"].iloc[0]
    assert r0_row["Mean WER (%)"] == 0.0
    assert r0_row["Clips"] == 2


def test_build_failure_log():
    df_r0 = pd.DataFrame([
        {"clip_id": "c1", "ref": "a", "hyp": "a", "wer": 0.0, "cer": 0.0},
        {"clip_id": "c2", "ref": "a", "hyp": "b", "wer": 1.0, "cer": 1.0},
        {"clip_id": "c3", "ref": "a", "hyp": "c", "wer": 0.5, "cer": 0.5},
    ])
    wer_dfs = {"R0": df_r0}
    failures = build_failure_log(wer_dfs, top_k=2)
    assert len(failures) == 2
    assert failures.iloc[0]["clip_id"] == "c2"  # Highest WER first
    assert failures.iloc[1]["clip_id"] == "c3"
    assert "guessed_cause" in failures.columns
