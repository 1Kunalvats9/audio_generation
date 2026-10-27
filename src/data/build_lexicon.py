"""Pronunciations for the words MFA's dictionary lacks  (M2)

9% of LJSpeech clips (1,179) contain a word missing from english_us_arpa:
names from the Warren Commission report ("hosty", "connally"), compounds
("motorcade", "palmprint") and acronyms ("FBI"). MFA aligns such a word as
"spn", and spn clips are excluded, so without this step we would lose 9% of
the data, mostly from a few source books, plus one test_100 clip.
Instead every such word gets a pronunciation BEFORE alignment:

    acronyms   spelled letter by letter ("fbi" -> EH1 F B IY1 AY1), see ACRONYMS
    the rest   MFA's english_us_arpa G2P model

Run from the repo root (make align-* runs the merge step itself):

    python -m src.data.build_lexicon            # merge configs/oov.dict into the MFA dictionary
    python -m src.data.build_lexicon --refresh  # first re-find the OOVs and re-run G2P (needs MFA)

configs/oov.dict is committed, so everyone aligns with the same pronunciations
without rerunning G2P. The merged dictionary goes to LEXICON.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import CFG, ROOT

MFA = ["conda", "run", "-n", "aligner", "mfa"]
BASE_NAME = CFG["mfa"]["dictionary"]                       # english_us_arpa
MFA_ROOT = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
BASE_DICT = MFA_ROOT / "pretrained_models" / "dictionary" / f"{BASE_NAME}.dict"
OOV_DICT = ROOT / "configs" / "oov.dict"
LEXICON = ROOT / "data" / "lexicon" / f"{BASE_NAME}_lj.dict"
CORPORA = ["pilot", "core", "rest"]

# Letter names, used to spell acronyms. Taken from english_us_arpa's own entries.
LETTERS = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F",
    "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1", "l": "EH1 L",
    "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1", "q": "K Y UW1", "r": "AA1 R",
    "s": "EH1 S", "t": "T IY1", "u": "Y UW1", "v": "V IY1", "w": "D AH1 B AH0 L Y UW0",
    "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}

# OOV words the reader spells out. Checked against the raw transcripts:
# "BBL" is "2 inch BBL" (barrel) on a gun advert, "UV" is "UV (ultra violet)".
ACRONYMS = ["bbl", "fbi", "fpcc", "mvd", "prs", "ussr", "uv", "wdsu", "wdsu-tv"]

# G2P output that was clearly wrong on review: a missing vowel ("jebb" -> JH B),
# a lost syllable, or no primary stress. Built from english_us_arpa's own entries
# where possible ("revved" = "rev" + D, "alek" = "alec"). Rare names whose real
# pronunciation is unknown are left as G2P made them.
OVERRIDES = {
    "alek": "AE1 L IH0 K",
    "busdriver": "B AH1 S D R AY2 V ER0",
    "cadigan": "K AE1 D IH0 G AH0 N",
    "connally": "K AO1 N AH0 L IY0",
    "dumaine": "D UW0 M EY1 N",
    "ervay": "ER1 V EY0",
    "euins": "Y UW1 IH0 N Z",
    "euins'": "Y UW1 IH0 N Z",
    "fortyish": "F AO1 R T IY0 IH0 SH",
    "henrichson": "HH EH1 N R IH0 K S AH0 N",
    "hibner": "HH IH1 B N ER0",
    "highchair": "HH AY1 CH EH2 R",
    "hurchel": "HH ER1 CH AH0 L",
    "interagency": "IH2 N T ER0 EY1 JH AH0 N S IY0",
    "jebb": "JH EH1 B",
    "lineup": "L AY1 N AH2 P",
    "mcwatters": "M AH0 K W AA1 T ER0 Z",
    "mcwatters'": "M AH0 K W AA1 T ER0 Z",
    "mohrenschildt": "M AO1 R AH0 N SH IH2 L T",
    "mohrenschildts": "M AO1 R AH0 N SH IH2 L T S",
    "mullay": "M AH1 L EY0",
    "nonreferral": "N AA0 N R IH0 F ER1 AH0 L",
    "preview": "P R IY1 V Y UW2",
    "probative": "P R OW1 B AH0 T IH0 V",
    "protist": "P R OW1 T IH0 S T",
    "protists": "P R OW1 T IH0 S T S",
    "punchmark": "P AH1 N CH M AA2 R K",
    "reappropriate": "R IY0 AH0 P R OW1 P R IY0 EY2 T",
    "redone": "R IY0 D AH1 N",
    "reexamining": "R IY0 IH0 G Z AE1 M IH0 N IH0 NG",
    "refaced": "R IY0 F EY1 S T",
    "referral": "R IH0 F ER1 AH0 L",
    "referrals": "R IH0 F ER1 AH0 L Z",
    "revill": "R EH1 V AH0 L",
    "revved": "R EH1 V D",
    "schizoid": "S K IH1 T S OY0 D",
    "shaneyfelt": "SH EY1 N IY0 F EH2 L T",
    "stomata": "S T OW1 M AH0 T AH0",
    "unremedied": "AH0 N R EH1 M AH0 D IY0 D",
    "viscose": "V IH1 S K OW0 S",
    "yardsman": "Y AA1 R D Z M AH0 N",
}


def spell(word: str) -> str:
    """"fbi" -> "EH1 F B IY1 AY1". Hyphens are skipped."""
    return " ".join(LETTERS[ch] for ch in word if ch != "-")


def read_dict(path: Path) -> dict[str, str]:
    """word -> first pronunciation. Handles both the plain and the probability format."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0] not in out:
            out[fields[0]] = fields[-1].strip()
    return out


def find_oovs(tmp: Path) -> list[str]:
    """Every word MFA's own tokenizer cannot find in the base dictionary, across all corpora."""
    words = set()
    for name in CORPORA:
        corpus = ROOT / "data" / f"mfa_{name}"
        if not corpus.exists():
            continue
        out = tmp / name
        subprocess.run(MFA + ["find_oovs", str(corpus), BASE_NAME, str(out), "--clean"],
                       check=True, capture_output=True)
        found = out / f"oovs_found_{BASE_NAME}.txt"
        words |= {w.strip() for w in found.read_text(encoding="utf-8").splitlines() if w.strip()}
    return sorted(words)


def refresh() -> None:
    """Find OOVs with MFA, pronounce them, write configs/oov.dict."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        oovs = find_oovs(tmp)
        print(f"{len(oovs)} OOV words across {CORPORA}")

        word_list = tmp / "oovs.txt"
        word_list.write_text("\n".join(w for w in oovs if w not in ACRONYMS) + "\n", encoding="utf-8")
        g2p_out = tmp / "g2p.dict"
        subprocess.run(MFA + ["g2p", str(word_list), BASE_NAME, str(g2p_out), "--clean"],
                       check=True, capture_output=True)
        prons = read_dict(g2p_out)

    prons.update({w: spell(w) for w in ACRONYMS if w in oovs})
    prons.update({w: p for w, p in OVERRIDES.items() if w in oovs})
    missing = sorted(set(oovs) - set(prons))
    if missing:
        print(f"WARNING: no pronunciation for {missing}", file=sys.stderr)

    lines = [f"{w}\t{prons[w]}" for w in sorted(prons)]
    OOV_DICT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OOV_DICT.relative_to(ROOT)}  ({len(lines)} words, "
          f"{sum(w in ACRONYMS for w in prons)} spelled as acronyms, "
          f"{sum(w in OVERRIDES for w in prons)} hand-corrected)")


def merge() -> None:
    """Base MFA dictionary + configs/oov.dict -> LEXICON."""
    if not BASE_DICT.exists():
        sys.exit(f"missing {BASE_DICT}\nrun: make mfa-setup")
    base = BASE_DICT.read_text(encoding="utf-8")
    base_prons = read_dict(BASE_DICT)
    known = set(base_prons)
    extra = [line for line in OOV_DICT.read_text(encoding="utf-8").splitlines()
             if line.split("\t")[0] not in known]

    # a typo in a hand-written pronunciation would make MFA reject the dictionary
    inventory = {p for pron in base_prons.values() for p in pron.split()}
    unknown = {p for line in extra for p in line.split("\t")[1].split()} - inventory
    if unknown:
        sys.exit(f"phones not in {BASE_NAME}: {sorted(unknown)} - fix {OOV_DICT.relative_to(ROOT)}")

    LEXICON.parent.mkdir(parents=True, exist_ok=True)
    LEXICON.write_text(base.rstrip("\n") + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(f"wrote {LEXICON.relative_to(ROOT)}  ({len(known)} base words + {len(extra)} from "
          f"{OOV_DICT.relative_to(ROOT)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-find OOVs and re-run G2P first")
    args = ap.parse_args()
    if args.refresh or not OOV_DICT.exists():
        refresh()
    merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
