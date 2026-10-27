"""MFA TextGrids -> data/cache/phones.csv + data/cache/excluded.csv  (M2)

Run after `make align-pilot` / `align-core` / `align-rest`, from the repo root:

    python -m src.data.textgrid_to_phones              # every corpus aligned so far
    python -m src.data.textgrid_to_phones pilot        # only these corpora

phones.csv: one row per phone, clips in ID order (FROZEN - M3/M4 read these):
    clip_id, idx, phone, start_frame, n_frames, word, word_idx, idx_in_word, word_len

    phone        MFA's ARPAbet label WITH stress (AH0, IY1, ...), or "sil".
                 phone.str.rstrip("012") gives the 39 stress-free classes.
    start_frame  first mel frame of the phone. Phones tile the clip with no gaps:
                 start_frame + n_frames == the next row's start_frame.
    n_frames     >= 1, and a clip's rows sum to its n_frames in metadata_clean.csv
    word         the word the phone belongs to; "sil" for silence
    word_idx     0-based index of that word among the clip's spoken words; -1 for silence
    idx_in_word  0-based position inside the word (0 for silence)
    word_len     phones in the word (1 for silence)

excluded.csv: clip_id, reason - every clip sent to MFA that is not in phones.csv:
    no_textgrid      MFA wrote no TextGrid (it could not align the clip at all)
    spn              MFA heard a word it could not pronounce (out of vocabulary)
    length_mismatch  durations differ from n_frames by more than MAX_MISMATCH frames
    zero_frame       a 0-frame phone whose neighbours are too short to lend a frame
    bad_textgrid     unreadable file, missing tier, or a phone outside every word
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from praatio import textgrid
from tqdm import tqdm

from src.config import CACHE_DIR, METADATA_CLEAN, ROOT, SECONDS_PER_FRAME, SIL, sec_to_frame
from src.utils import read_filelist

CORPORA = ["pilot", "core", "rest"]
TEXTGRID_DIR = ROOT / "data" / "textgrids"     # make align-<corpus> writes here
PHONES_CSV = CACHE_DIR / "phones.csv"
EXCLUDED_CSV = CACHE_DIR / "excluded.csv"

COLUMNS = ["clip_id", "idx", "phone", "start_frame", "n_frames",
           "word", "word_idx", "idx_in_word", "word_len"]

MAX_MISMATCH = 2                  # frames the last phone may absorb before a clip is excluded
SILENCE_LABELS = {"", "sil", "sp", "<eps>"}   # never "silence": it is a word in LJSpeech
UNKNOWN_PHONE = "spn"             # MFA's phone for an out-of-vocabulary word
UNKNOWN_WORD = "<unk>"
GAP_EPS = 1e-4                    # seconds; smaller gaps are float noise, not silence


class Excluded(Exception):
    """A clip that cannot be used. str(e) is the reason written to excluded.csv."""


# ---------------------------------------------------------------------------
# TextGrid -> intervals in seconds
# ---------------------------------------------------------------------------

def fill_silence(intervals, xmin: float, xmax: float) -> list[tuple]:
    """Make the tier gap-free: gaps and silence-like labels become SIL, neighbouring SILs merge.

    MFA leaves silence as empty or missing intervals; this turns them into
    explicit "sil" phones so every frame of the clip belongs to exactly one phone.
    """
    out = []
    t = xmin
    for start, end, label in intervals:
        if start - t > GAP_EPS:
            out.append((t, start, SIL))
        else:
            start = t                  # snap float noise so boundaries stay shared
        label = label.strip()
        out.append((start, end, SIL if label.lower() in SILENCE_LABELS else label))
        t = end
    if xmax - t > GAP_EPS:
        out.append((t, xmax, SIL))

    merged = []
    for iv in out:
        if merged and iv[2] == SIL and merged[-1][2] == SIL:
            merged[-1] = (merged[-1][0], iv[1], SIL)
        else:
            merged.append(iv)
    return merged


def _tier(tg, kind: str):
    """The "phones" or "words" tier. MFA names them "<speaker> - phones" in some versions."""
    for name in tg.tierNames:
        if name == kind or name.endswith(f" - {kind}"):
            return tg.getTier(name)
    raise Excluded("bad_textgrid")


def read_tiers(path: Path) -> tuple[list[tuple], list[tuple]]:
    """-> (phones, words) as (start, end, label) lists.

    phones covers the whole clip with no gaps (silence filled in as SIL);
    words holds only the spoken words, lowercased.
    """
    try:
        tg = textgrid.openTextgrid(str(path), False)
    except Exception as e:
        raise Excluded("bad_textgrid") from e
    phones = fill_silence(_tier(tg, "phones").entries, tg.minTimestamp, tg.maxTimestamp)
    words = [(s, e, w.strip().lower()) for s, e, w in _tier(tg, "words").entries
             if w.strip().lower() not in SILENCE_LABELS]
    return phones, words


# ---------------------------------------------------------------------------
# Seconds -> whole frames
# ---------------------------------------------------------------------------

def fix_zero_frames(dur: list[int]) -> list[int]:
    """Every phone under 1 frame borrows from its longer neighbour."""
    dur = list(dur)
    for i in range(len(dur)):
        if dur[i] >= 1:
            continue
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(dur)]
        if not neighbours:
            raise Excluded("zero_frame")
        j = max(neighbours, key=lambda j: dur[j])
        need = 1 - dur[i]
        if dur[j] - need < 1:
            raise Excluded("zero_frame")
        dur[j] -= need
        dur[i] = 1
    return dur


def frame_durations(bounds: list[float], target: int) -> list[int]:
    """Phone boundaries in seconds -> whole-frame durations that sum to exactly `target`.

    Rounding the boundaries (not the durations) keeps phones contiguous, so the
    total is off from n_frames by at most a frame; the last phone absorbs it.
    """
    frames = [sec_to_frame(t) for t in bounds]
    dur = [b - a for a, b in zip(frames, frames[1:])]
    remainder = target - sum(dur)
    if abs(remainder) > MAX_MISMATCH:
        raise Excluded("length_mismatch")
    dur[-1] += remainder
    return fix_zero_frames(dur)


# ---------------------------------------------------------------------------
# Phones -> words
# ---------------------------------------------------------------------------

def assign_words(phones: list[tuple], words: list[tuple]) -> list[tuple]:
    """-> (word, word_idx, idx_in_word, word_len) per phone.

    A phone belongs to the word whose interval contains the phone's midpoint.
    """
    word_of = []
    for start, end, label in phones:
        if label == SIL:
            word_of.append(-1)
            continue
        mid = (start + end) / 2
        k = next((k for k, (ws, we, _) in enumerate(words) if ws <= mid < we), None)
        if k is None:
            raise Excluded("bad_textgrid")
        word_of.append(k)

    word_len = Counter(k for k in word_of if k >= 0)
    seen = Counter()
    out = []
    for k in word_of:
        if k < 0:
            out.append((SIL, -1, 0, 1))
        else:
            out.append((words[k][2], k, seen[k], word_len[k]))
            seen[k] += 1
    return out


def clip_rows(clip_id: str, path: Path, target: int) -> list[tuple]:
    """One clip's TextGrid -> its phones.csv rows, in COLUMNS order. Raises Excluded."""
    phones, words = read_tiers(path)
    if any(p[2] == UNKNOWN_PHONE for p in phones) or any(w[2] == UNKNOWN_WORD for w in words):
        raise Excluded("spn")

    bounds = [p[0] for p in phones] + [phones[-1][1]]
    dur = frame_durations(bounds, target)
    starts = np.cumsum([0] + dur[:-1])
    info = assign_words(phones, words)

    return [(clip_id, i, phone, int(start), d, *word)
            for i, ((_, _, phone), start, d, word) in enumerate(zip(phones, starts, dur, info))]


# ---------------------------------------------------------------------------
# Whole corpus
# ---------------------------------------------------------------------------

def find_textgrids(corpora) -> dict[str, Path | None]:
    """-> {clip_id: TextGrid path, or None if MFA wrote none} for every aligned corpus.

    A corpus counts as aligned once data/textgrids/<name>/ exists. The clips
    expected there are the .lab files M1 put in data/mfa_<name>/.
    """
    clips = {}
    for name in corpora:
        tg_dir = TEXTGRID_DIR / name
        if not tg_dir.exists():
            continue
        paths = {p.stem: p for p in tg_dir.rglob("*.TextGrid")}
        lab_dir = ROOT / "data" / f"mfa_{name}"
        expected = {p.stem for p in lab_dir.rglob("*.lab")} if lab_dir.exists() else set()
        for clip_id in expected | set(paths):
            clips[clip_id] = paths.get(clip_id)
    return dict(sorted(clips.items()))


def check(phones: pd.DataFrame, target: dict) -> None:
    """The guarantees M3 and M4 build on. Fail here, not in their code."""
    assert (phones["n_frames"] >= 1).all(), "phone with < 1 frame"
    per_clip = phones.groupby("clip_id")["n_frames"].sum()
    bad = [c for c, n in per_clip.items() if n != target[c]]
    assert not bad, f"{len(bad)} clips do not sum to n_frames, e.g. {bad[:3]}"
    ends = phones["start_frame"] + phones["n_frames"]
    same_clip = phones["clip_id"].eq(phones["clip_id"].shift(-1))
    assert (ends[same_clip] == phones["start_frame"].shift(-1)[same_clip]).all(), "gap between phones"
    assert (phones.groupby("clip_id")["start_frame"].first() == 0).all(), "clip not starting at frame 0"


def report(phones: pd.DataFrame, excluded: pd.DataFrame) -> None:
    kept = set(phones["clip_id"])
    print(f"\nwrote {PHONES_CSV.relative_to(ROOT)}  ({len(phones)} phones, {len(kept)} clips)")
    print(f"wrote {EXCLUDED_CSV.relative_to(ROOT)}  ({len(excluded)} clips)")
    for reason, n in excluded["reason"].value_counts().items():
        print(f"  {reason:16s} {n}")

    spoken = phones[phones["phone"] != SIL]
    ms = phones["n_frames"] * SECONDS_PER_FRAME * 1000
    print(f"\nphone labels   {spoken['phone'].nunique()} with stress, "
          f"{spoken['phone'].str.rstrip('012').nunique()} without, plus {SIL!r}")
    print(f"phone length   {ms[phones['phone'] != SIL].mean():.1f} ms mean, "
          f"silence {ms[phones['phone'] == SIL].mean():.1f} ms mean")

    print("\nclips kept per split:")
    for split in ["train", "val", "test", "test_100"]:
        ids = set(read_filelist(split))
        print(f"  {split:9s} {len(ids & kept):6d} / {len(ids)}")

    lost = sorted(set(read_filelist("test_100")) & set(excluded["clip_id"]))
    if lost:
        print(f"\nWARNING: {len(lost)} test_100 clips excluded, so A1/A2 cannot be built "
              f"for them - tell M4 and M5: {lost}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="*", help=f"any of {CORPORA} (default: all)")
    args = ap.parse_args()
    corpora = args.corpora or CORPORA
    unknown = set(corpora) - set(CORPORA)
    if unknown:
        ap.error(f"unknown corpora {sorted(unknown)}, choose from {CORPORA}")

    if not METADATA_CLEAN.exists():
        print("run: make metadata", file=sys.stderr)
        return 1
    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str})
    target = dict(zip(meta["id"], meta["n_frames"]))

    clips = find_textgrids(corpora)
    if not clips:
        print(f"no TextGrids under {TEXTGRID_DIR.relative_to(ROOT)}/ - run: make align-pilot",
              file=sys.stderr)
        return 1

    rows, excluded = [], []
    for clip_id, path in tqdm(clips.items(), desc="TextGrid -> phones"):
        try:
            if path is None:
                raise Excluded("no_textgrid")
            if clip_id not in target:
                raise Excluded("bad_textgrid")
            rows.extend(clip_rows(clip_id, path, int(target[clip_id])))
        except Excluded as e:
            excluded.append((clip_id, str(e)))

    phones = pd.DataFrame(rows, columns=COLUMNS)
    excluded = pd.DataFrame(excluded, columns=["clip_id", "reason"])
    check(phones, target)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    phones.to_csv(PHONES_CSV, index=False)
    excluded.to_csv(EXCLUDED_CSV, index=False)
    report(phones, excluded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
