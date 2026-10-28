"""Figures for the project report  (EDA, feature engineering, results).

Writes vector PDFs (for LaTeX) and PNGs (for the slides and README) into outputs/figures/.

Usage:
    python -m src.eval.make_figures

Produces:
    fig_dataset.{pdf,png}     clip lengths, phones per clip, clips per source text
    fig_durations.{pdf,png}   phone duration by class, and the stress effect
    fig_importance.{pdf,png}  permutation importance of the 21 features
    fig_results.{pdf,png}     duration MAE by model, WER by system, predicted vs true
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.baseline.duration_features import add_context, features_from_context
from src.baseline.duration_models import make_gbt
from src.config import CACHE_DIR, FIGURES, METADATA_CLEAN, OUTPUTS, SECONDS_PER_FRAME, SEED, SIL
from src.utils import prefix_of, read_filelist

MS = SECONDS_PER_FRAME * 1000
INK, GRID = "#21295C", "#E6EDF1"
BLUE, TEAL, GREY, RED = "#065A82", "#02C39A", "#9AB2BF", "#B85042"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": "#8C9BA8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
})


def _tidy(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_dataset(meta, phones):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))

    axes[0].hist(meta.seconds, bins=45, color=BLUE, edgecolor="white", lw=0.3)
    axes[0].axvline(meta.seconds.median(), color=TEAL, lw=1.4,
                    label=f"median {meta.seconds.median():.2f} s")
    axes[0].set_xlabel("clip length (s)"); axes[0].set_ylabel("clips")
    axes[0].set_title("(a) Clip length"); axes[0].legend(frameon=False)
    _tidy(axes[0])

    per_clip = phones.groupby("clip_id").size()
    axes[1].hist(per_clip, bins=45, color=BLUE, edgecolor="white", lw=0.3)
    axes[1].set_xlabel("phones per clip"); axes[1].set_ylabel("clips")
    axes[1].set_title("(b) Phones per clip")
    _tidy(axes[1])

    counts = meta.index.map(prefix_of).value_counts().sort_index()
    test = {p for p in counts.index if p == "LJ005"}
    cols = [RED if p in test else GREY for p in counts.index]
    axes[2].bar(range(len(counts)), counts.values, color=cols, width=0.9)
    axes[2].set_xlabel("source text (50 books)"); axes[2].set_ylabel("clips")
    axes[2].set_title("(c) Clips per source text")
    axes[2].set_xticks([])
    axes[2].annotate("LJ005\n(held out)", xy=(list(counts.index).index("LJ005"), counts["LJ005"]),
                     xytext=(6, 480), fontsize=6.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    _tidy(axes[2])
    save(fig, "fig_dataset")


def fig_durations(phones):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))

    df = add_context(phones)
    order = ["silence", "stop", "fricative", "nasal", "other", "vowel"]
    data = [(df.loc[df.cls == c, "n_frames"] * MS).values for c in order]
    bp = axes[0].boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE); patch.set_alpha(0.75); patch.set_edgecolor("none")
    for med in bp["medians"]:
        med.set_color(TEAL); med.set_linewidth(1.4)
    axes[0].set_xticks(range(1, len(order) + 1)); axes[0].set_xticklabels(order, rotation=20)
    axes[0].set_ylabel("duration (ms)"); axes[0].set_title("(a) Duration by phone class")
    _tidy(axes[0])

    vowels = df[df.cls == "vowel"]
    stress = vowels.phone.str[-1].map({"0": "unstressed", "1": "primary", "2": "secondary"})
    lab = ["unstressed", "secondary", "primary"]
    means = [(vowels.loc[stress == s, "n_frames"] * MS).mean() for s in lab]
    axes[1].bar(lab, means, color=[GREY, BLUE, TEAL], width=0.6)
    for i, v in enumerate(means):
        axes[1].text(i, v + 1.5, f"{v:.0f} ms", ha="center", fontsize=7.5, color=INK)
    axes[1].set_ylabel("mean vowel duration (ms)")
    axes[1].set_title("(b) Stress is a real duration signal")
    axes[1].set_ylim(0, max(means) * 1.18)
    _tidy(axes[1])
    save(fig, "fig_durations")


def fig_importance(phones):
    df = add_context(phones)
    tr = df[df.clip_id.isin(set(read_filelist("train")))]
    va = df[df.clip_id.isin(set(read_filelist("val")))]
    Xtr, ytr = features_from_context(tr)
    Xva, yva = features_from_context(va)

    print("  fitting GBT for permutation importance ...")
    pipe = make_gbt().fit(Xtr, np.log(ytr.to_numpy()))
    r = permutation_importance(pipe, Xva, np.log(yva.to_numpy()),
                               n_repeats=5, random_state=SEED, n_jobs=1,
                               scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=Xva.columns).sort_values()

    fig, ax = plt.subplots(figsize=(3.45, 3.3))
    top = imp.tail(12)
    cols = [TEAL if v == top.max() else BLUE for v in top.values]
    ax.barh(range(len(top)), top.values, color=cols, height=0.72)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.set_xlabel("increase in MAE when shuffled (log-frames)")
    ax.set_title("Permutation importance (validation)")
    _tidy(ax, grid_axis="x")
    save(fig, "fig_importance")
    imp.sort_values(ascending=False).round(5).to_csv(OUTPUTS / "results" / "feature_importance.csv",
                                                     header=["importance"])
    print("  wrote outputs/results/feature_importance.csv")


def fig_results():
    dur = pd.read_csv(OUTPUTS / "results" / "duration_models.csv")
    test = dur[dur.split == "test"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))

    models = ["mean", "ridge", "gbt"]
    names = ["phone mean\n(baseline)", "ridge", "boosted\ntrees"]
    vals = [test.loc[m, "mae_ms"] for m in models]
    axes[0].bar(names, vals, color=[GREY, BLUE, TEAL], width=0.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    axes[0].set_ylabel("duration MAE (ms)"); axes[0].set_title("(a) Duration model")
    axes[0].set_ylim(0, max(vals) * 1.18)
    _tidy(axes[0])

    systems = ["R0", "R1", "A1", "A2"]
    wer = []
    for s in systems:
        w = pd.read_csv(OUTPUTS / "results" / f"wer_{s}.csv")
        wer.append(w.wer.mean() * 100)
    axes[1].bar(systems, wer, color=[GREY, BLUE, RED, TEAL], width=0.6)
    for i, v in enumerate(wer):
        axes[1].text(i, v + 0.7, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    axes[1].set_ylabel("word error rate (%)"); axes[1].set_title("(b) Intelligibility")
    axes[1].set_ylim(0, max(wer) * 1.18)
    _tidy(axes[1])

    pred = pd.read_csv(OUTPUTS / "durations" / "test_pred.csv", dtype={"clip_id": str})
    sub = pred[pred.phone != SIL]
    axes[2].hexbin(sub.n_frames * MS, sub.pred_gbt * MS, gridsize=38, bins="log",
                   cmap="Blues", mincnt=1, linewidths=0)
    lim = 220
    axes[2].plot([0, lim], [0, lim], color=RED, lw=1.0, ls="--", label="perfect")
    axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("true duration (ms)"); axes[2].set_ylabel("predicted (ms)")
    axes[2].set_title("(c) Predicted vs true"); axes[2].legend(frameon=False, loc="upper left")
    _tidy(axes[2], grid_axis="both")
    save(fig, "fig_results")


def main() -> int:
    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str}).set_index("id")
    phones = pd.read_csv(CACHE_DIR / "phones.csv", dtype={"clip_id": str, "word": str})
    print("figures ->", FIGURES)
    fig_dataset(meta, phones)
    fig_durations(phones)
    fig_results()
    fig_importance(phones)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
