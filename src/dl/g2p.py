"""Text to phonemes, for synthesising sentences nobody recorded  (M1 — phase 2).

Phase 1 could only re-synthesise clips we already had MFA alignments for. To
type a new sentence and hear it spoken, we need the same phoneme sequence MFA
would have produced — from text alone.

The pipeline deliberately mirrors how the training data was built, because the
model only ever saw phonemes produced that way:

    "Hello, this is 2026!"
      -> expand numbers        "hello, this is two thousand and twenty six!"
      -> src.text.normalize    same cleaner the corpus went through
      -> per-word lookup       configs/oov.dict first, then CMUdict via g2p_en
      -> silence insertion     at punctuation, and always at the end
      -> ['HH','AH0','L','OW1','sil', ...,'sil']

Why `configs/oov.dict` comes first: those 366 words are absent from CMUdict, and
M2 wrote the pronunciations MFA actually used when aligning the corpus. Using
anything else for them would contradict the durations the model trained on.

Silence follows the corpus rather than intuition. Measured over all 13,100
training clips: 100% end with silence, only 31% begin with one, and the median
clip has 4 silences. So a trailing `sil` is always appended, a leading one is
off by default, and punctuation becomes `sil`.

Every symbol returned is guaranteed to be in the frozen vocabulary — the class
checks, so a bad pronunciation fails here rather than as a confusing embedding
error later.

Usage:
    python -m src.dl.g2p "Hello, this is our project."
    from src.dl.g2p import G2P
    g2p = G2P()
    phones = g2p.phonemes("Hello there.")      # ['HH','AH0','L','OW1', ...]
    ids    = g2p.encode("Hello there.")        # ints for the embedding
"""

from __future__ import annotations

import re
import sys

from src.config import ROOT, SIL
from src.text.normalize import clean_text

OOV_DICT = ROOT / "configs" / "oov.dict"
CORPUS_LEXICON = ROOT / "data" / "cache" / "lexicon.json"

_PHONE_RE = re.compile(r"^[A-Z]+[0-9]?$")
_TOKEN_RE = re.compile(r"[a-z']+|[.,;:!?]")
_PAUSE_PUNCT = set(".,;:!?")
_DIGITS_RE = re.compile(r"\d+")


def expand_numbers(text: str) -> str:
    """"in 2026" -> "in two thousand and twenty six".

    LJSpeech's transcripts arrived with numbers already spelled out, so the
    phase 1 cleaner never needed to do this. Free text does.
    """
    try:
        import inflect
    except ImportError:                                    # pragma: no cover
        return text
    engine = inflect.engine()

    def spell(m: re.Match) -> str:
        words = engine.number_to_words(m.group(), andword="and")
        return words.replace("-", " ").replace(",", "")

    return _DIGITS_RE.sub(spell, text)


def build_corpus_lexicon(phones_csv=None, out=CORPUS_LEXICON) -> dict[str, list[str]]:
    """The pronunciation MFA actually used for each word, learned from the corpus.

    Forced alignment chooses between pronunciation variants by listening, so MFA
    gives function words their reduced forms — "to" as T AH0 rather than the
    citation T UW1, "and" as AH0 N D rather than AE1 N D. A dictionary cannot
    know which variant was spoken, so CMUdict returns the citation form and the
    sequence disagrees with everything the model trained on.

    Since phones.csv labels every phone with its word, we can simply read off
    the majority pronunciation per word. Built from the TRAINING SPLIT ONLY, so
    test transcripts never influence it.
    """
    import json
    from collections import Counter

    import pandas as pd

    from src.utils import read_filelist

    phones_csv = phones_csv or (ROOT / "data" / "cache" / "phones.csv")
    ph = pd.read_csv(phones_csv, dtype={"clip_id": str, "word": str})
    ph = ph[ph["clip_id"].isin(set(read_filelist("train")))]
    ph = ph[(ph["phone"] != SIL) & (ph["word_idx"] >= 0)]

    counts: dict[str, Counter] = {}
    for (_, _), g in ph.groupby(["clip_id", "word_idx"], sort=False):
        g = g.sort_values("idx")
        word = str(g["word"].iloc[0]).lower()
        counts.setdefault(word, Counter())[tuple(g["phone"])] += 1

    lexicon = {w: list(c.most_common(1)[0][0]) for w, c in counts.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(lexicon, f)
    return lexicon


def load_corpus_lexicon(path=CORPUS_LEXICON) -> dict[str, list[str]]:
    import json
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_lexicon(path=OOV_DICT) -> dict[str, list[str]]:
    """configs/oov.dict -> {word: [phones]}. Tab- or space-separated."""
    lexicon: dict[str, list[str]] = {}
    if not path.exists():
        return lexicon
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            word, _, rest = line.partition("\t")
            if not rest:
                word, _, rest = line.partition(" ")
            phones = rest.split()
            if word and phones:
                lexicon[word.strip().lower()] = phones
    return lexicon


class G2P:
    """Text -> phoneme symbols from the frozen vocabulary."""

    def __init__(self, vocab=None, lexicon_path=OOV_DICT):
        if vocab is None:
            from src.dl.vocab import PhoneVocab
            vocab = PhoneVocab.load()
        self.vocab = vocab
        self.known = set(vocab.symbols)
        self.lexicon = load_lexicon(lexicon_path)
        self.corpus = load_corpus_lexicon()      # what MFA actually used, highest priority
        self._engine = None                     # g2p_en is slow to construct; build on demand
        self.unknown_words: list[str] = []       # words that needed the neural fallback

    # -- pronunciation -------------------------------------------------------

    def _g2p_engine(self):
        if self._engine is None:
            try:
                from g2p_en import G2p
            except ImportError as e:             # pragma: no cover
                raise ImportError(
                    "g2p_en is needed for words outside configs/oov.dict.\n"
                    "Install it with: uv pip install g2p_en") from e
            self._engine = G2p()
        return self._engine

    def pronounce(self, word: str) -> list[str]:
        """One word -> its phones.

        Priority: the corpus lexicon (what MFA actually used, so it matches the
        model's training), then configs/oov.dict, then CMUdict/neural g2p_en.
        """
        word = word.lower().strip()
        if not word:
            return []
        if word in self.corpus:                  # the pronunciation the model trained on
            return list(self.corpus[word])
        if word in self.lexicon:                 # M2's hand-written OOV pronunciations
            return list(self.lexicon[word])
        self.unknown_words.append(word)
        return [t for t in self._g2p_engine()(word) if _PHONE_RE.match(t)]

    # -- sentences -----------------------------------------------------------

    def phonemes(self, text: str, leading_silence: bool = False,
                 trailing_silence: bool = True) -> list[str]:
        """A sentence -> the phone sequence, with silences where the corpus puts them."""
        cleaned = clean_text(expand_numbers(text))
        tokens = _TOKEN_RE.findall(cleaned)

        out: list[str] = [SIL] if leading_silence else []
        spoke = False
        for token in tokens:
            if token in _PAUSE_PUNCT:
                if out and out[-1] != SIL:       # never two silences in a row
                    out.append(SIL)
                continue
            phones = self.pronounce(token)
            if phones:
                spoke = True
                out.extend(phones)

        # Checked before the trailing silence is added, so punctuation alone
        # ("...") is refused rather than synthesised as a lone silence.
        if not spoke:
            raise ValueError(f"no pronounceable words in {text!r}")

        if trailing_silence and out[-1] != SIL:
            out.append(SIL)

        unknown = sorted({p for p in out if p not in self.known})
        if unknown:
            raise ValueError(
                f"produced symbols outside the frozen vocabulary: {unknown}\n"
                "The vocabulary comes from phones.csv; a pronunciation source disagrees with it.")
        return out

    def encode(self, text: str, **kw) -> list[int]:
        """Straight to embedding ids."""
        return self.vocab.encode(self.phonemes(text, **kw))


def _main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--build-lexicon":
        lex = build_corpus_lexicon()
        print(f"corpus lexicon: {len(lex):,} words from the training split")
        print(f"wrote {CORPUS_LEXICON.relative_to(ROOT)}")
        return 0

    text = " ".join(args) or "Hello, this is our project. It speaks 2026 words."
    g2p = G2P()

    print(f"text      {text}")
    print(f"cleaned   {clean_text(expand_numbers(text))}")
    phones = g2p.phonemes(text)
    print(f"phonemes  {' '.join(phones)}")
    print(f"ids       {g2p.encode(text)[:18]}{' ...' if len(phones) > 18 else ''}")
    print(f"\n{len(phones)} symbols, {phones.count(SIL)} silences")
    if g2p.unknown_words:
        print(f"not in configs/oov.dict, used CMUdict/neural: "
              f"{sorted(set(g2p.unknown_words))[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
