"""Compile baseline evaluation results and failure log (M5).

Aggregates:
  - outputs/results/wer_<system>.csv (R0, R1, A1, A2)
  - outputs/results/mae.csv (duration MAE, from M3)

Outputs:
  - outputs/results/summary.csv & printed Markdown table
  - outputs/results/failure_log.csv (10 worst clips per system for manual diagnosis)

Usage:
    python -m src.eval.make_results
"""

import sys
from pathlib import Path
import pandas as pd

from src.config import OUTPUTS

RESULTS_DIR = OUTPUTS / "results"
SYSTEMS = ["R0", "R1", "A1", "A2"]


def load_wer_results() -> dict[str, pd.DataFrame]:
    results = {}
    for sys_name in SYSTEMS:
        path = RESULTS_DIR / f"wer_{sys_name}.csv"
        if path.exists():
            results[sys_name] = pd.read_csv(path)
    return results


def load_duration_mae() -> dict[str, float]:
    """Load duration MAE in ms if available."""
    mae_dict = {}
    mae_file = RESULTS_DIR / "mae.csv"
    if mae_file.exists():
        try:
            df = pd.read_csv(mae_file)
            # Expecting columns like: system, mae_ms or model, mae_ms
            if "system" in df.columns and "mae_ms" in df.columns:
                for _, row in df.iterrows():
                    mae_dict[row["system"]] = float(row["mae_ms"])
        except Exception as e:
            print(f"Warning reading {mae_file}: {e}", file=sys.stderr)
    return mae_dict


def build_summary_table(wer_dfs: dict[str, pd.DataFrame], mae_dict: dict[str, float]) -> pd.DataFrame:
    rows = []
    system_descriptions = {
        "R0": "Real recording (sanity check)",
        "R1": "Real mel -> HiFi-GAN (copy synth)",
        "A1": "Phoneme templates + true MFA durations",
        "A2": "Phoneme templates + predicted durations",
    }

    for sys_name in SYSTEMS:
        desc = system_descriptions.get(sys_name, "")
        if sys_name in wer_dfs:
            df = wer_dfs[sys_name]
            mean_wer = df["wer"].mean() * 100
            mean_cer = df["cer"].mean() * 100
            n_clips = len(df)
            dur_mae = mae_dict.get(sys_name, None)
            dur_str = f"{dur_mae:.2f} ms" if dur_mae is not None else "-"
            rows.append({
                "System": sys_name,
                "Description": desc,
                "Clips": n_clips,
                "Mean WER (%)": round(mean_wer, 2),
                "Mean CER (%)": round(mean_cer, 2),
                "Duration MAE": dur_str,
            })
        else:
            rows.append({
                "System": sys_name,
                "Description": desc,
                "Clips": 0,
                "Mean WER (%)": "-",
                "Mean CER (%)": "-",
                "Duration MAE": "-",
            })

    return pd.DataFrame(rows)


def build_failure_log(wer_dfs: dict[str, pd.DataFrame], top_k: int = 10) -> pd.DataFrame:
    """Identify the top-K worst clips by WER for each system."""
    failure_rows = []
    for sys_name, df in wer_dfs.items():
        if df.empty:
            continue
        # Sort by WER descending, then CER descending
        worst = df.sort_values(by=["wer", "cer"], ascending=[False, False]).head(top_k)
        for rank, (_, row) in enumerate(worst.iterrows(), start=1):
            failure_rows.append({
                "system": sys_name,
                "rank": rank,
                "clip_id": row["clip_id"],
                "wer": row["wer"],
                "cer": row["cer"],
                "ref": row["ref"],
                "hyp": row["hyp"],
                "guessed_cause": "",  # To be filled during error analysis
            })
    return pd.DataFrame(failure_rows)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wer_dfs = load_wer_results()
    if not wer_dfs:
        print("No evaluation CSVs found in outputs/results/. Run: make wer", file=sys.stderr)
        return 1

    mae_dict = load_duration_mae()
    summary_df = build_summary_table(wer_dfs, mae_dict)

    summary_csv = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("TTS BASELINE EVALUATION SUMMARY")
    print("=" * 70)
    try:
        print(summary_df.to_markdown(index=False))
    except (ImportError, Exception):
        print(summary_df.to_string(index=False))
    print(f"\nWrote summary to {summary_csv}")

    # Build Failure Log
    failure_df = build_failure_log(wer_dfs)
    if not failure_df.empty:
        failure_csv = RESULTS_DIR / "failure_log.csv"
        failure_df.to_csv(failure_csv, index=False)
        print(f"Wrote top failure cases to {failure_csv}")
        print("Review the failure log and annotate `guessed_cause`:")
        print("  Options: flat prosody, wrong duration, slurred transition, misaligned phone, vocoder artefact")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
