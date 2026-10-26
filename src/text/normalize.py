"""Text cleaning for LJSpeech transcripts, before forced alignment.

Rules run in this order inside clean_text():
  1. curly quotes and dashes -> plain ASCII
  2. brackets -> space (or removed, see REMOVE_SQUARE_BRACKET_CONTENT)
  3. accents stripped (é -> e)
  4. abbreviations expanded (Mr. -> mister)
  5. lowercase, tidy spaces
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Set to True if the listening check shows the reader does NOT speak the
# words inside [square brackets]. Then the brackets and their content go.
REMOVE_SQUARE_BRACKET_CONTENT = False

PUNCT_MAP = {
    "\u2019": "'",   # ’ right single quote / apostrophe
    "\u2018": "'",   # ‘ left single quote
    "\u201c": '"',   # “ left double quote
    "\u201d": '"',   # ” right double quote
    "\u2014": " - ", # — em dash
    "\u2013": " - ", # – en dash
}

# Only matched WITH the full stop ("Mr." not "mr"), case-insensitive.
# Spellings match Whisper's text normaliser, so WER compares like with like.
ABBREVIATIONS = {
    "mrs": "missus",
    "mr": "mister",
    "dr": "doctor",
    "st": "saint",       # check your St. rows: change to "street" if that is more common
    "co": "company",
    "jr": "junior",
    "sr": "senior",
    "gen": "general",
    "col": "colonel",
    "capt": "captain",
    "lt": "lieutenant",
    "sgt": "sergeant",
    "maj": "major",
    "rev": "reverend",
    "hon": "honorable",
    "esq": "esquire",
    "ltd": "limited",
    "ft": "fort",
}

# ---------------------------------------------------------------------------
# Compiled patterns (built once, reused for all 13,100 rows)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile("|".join(map(re.escape, PUNCT_MAP)))
_ABBR_RE = re.compile(
    r"\b(" + "|".join(sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    flags=re.IGNORECASE,
)
_PARENS_RE = re.compile(r"[()]")
_SQUARE_CONTENT_RE = re.compile(r"\[[^\]]*\]")
_SQUARE_RE = re.compile(r"[\[\]]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_WHITESPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# One function per rule
# ---------------------------------------------------------------------------

def normalize_punctuation(text: str) -> str:
    """Curly quotes and long dashes -> plain ASCII. Must run before strip_accents."""
    return _PUNCT_RE.sub(lambda m: PUNCT_MAP[m.group(0)], text)


def handle_brackets(text: str) -> str:
    """(round) brackets -> space, keeping the words. [square] -> space, or removed with content."""
    text = _PARENS_RE.sub(" ", text)
    if REMOVE_SQUARE_BRACKET_CONTENT:
        return _SQUARE_CONTENT_RE.sub(" ", text)
    return _SQUARE_RE.sub(" ", text)


def strip_accents(text: str) -> str:
    """é -> e, ü -> u. Splits each letter from its accent, then drops non-ASCII."""
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("ascii")


def expand_abbreviations(text: str) -> str:
    """Mr. -> mister, Mrs. -> missus, ... (only the forms with a full stop)."""
    return _ABBR_RE.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], text)


def collapse_whitespace(text: str) -> str:
    """Many spaces -> one, no space before punctuation, no spaces at the ends."""
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline used for every transcript."""
    text = normalize_punctuation(text)
    text = handle_brackets(text)
    text = strip_accents(text)
    text = expand_abbreviations(text)
    text = text.lower()
    return collapse_whitespace(text)


if __name__ == "__main__":
    examples = [
        "Mr. Smith\u2019s house",
        "a caf\u00e9 in W\u00fcrzburg",
        "the man (a clerk) said",
        "\u201cYes,\u201d said Mrs. Jones.",
        "too    many   spaces ",
    ]
    for ex in examples:
        print(f"{ex!r:40} -> {clean_text(ex)!r}")