#!/usr/bin/env python
"""LIVE DEMO — type a sentence, hear our model say it.

    python demo.py            speaks the SENTENCES below, then asks for more
    python demo.py -q         just speaks the SENTENCES and exits (no prompt)
    python demo.py "Say this" speaks one sentence and exits

Nothing here is a recording. Every word is generated: text becomes phonemes, the
model predicts how long each sound lasts, draws the spectrogram, and a vocoder
turns that into audio. No recording of these sentences exists.

   ↓↓↓  EDIT THIS LIST  ↓↓↓
"""

SENTENCES = [
    "Hello, this is our advanced machine learning project.",
    "We generate speech from text using phoneme durations and a neural network.",
]



import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("samples/live")
CHECKPOINT = Path("outputs/checkpoints/D1/full/latest.pt")

GREEN, BLUE, GREY, RED, BOLD, OFF = "\033[32m", "\033[36m", "\033[90m", "\033[31m", "\033[1m", "\033[0m"


def play(path: Path) -> None:
    """Play through whatever the machine has; silently skip if it has nothing."""
    for player in (["afplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(player[0]):
            subprocess.run(player + [str(path)], check=False)
            return
    print(f"{GREY}   (no audio player found — open {path} yourself){OFF}")


def say(text: str, model, vocab, stats, device: str, index: int, silent: bool = False) -> None:
    from src.audio import save_wav
    from src.config import SECONDS_PER_FRAME
    from src.dl.speak import speak

    text = text.strip()
    if not text:
        return
    start = time.time()
    try:
        wav, phonemes, durations = speak(text, model, vocab, stats=stats, device=device)
    except Exception as e:                       # unknown word, empty text, ...
        print(f"{RED}   could not say that: {e}{OFF}")
        return
    elapsed = time.time() - start
    if silent:                                   # warm-up pass: no file, no sound
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{index:02d}.wav"
    save_wav(path, wav)

    seconds = durations.sum() * SECONDS_PER_FRAME
    print(f'{BOLD}   "{text}"{OFF}')
    print(f"{GREY}   {len(phonemes)} phonemes · {seconds:.1f}s of speech · generated in {elapsed:.1f}s{OFF}")
    print(f"{GREY}   {' '.join(phonemes[:22])}{' …' if len(phonemes) > 22 else ''}{OFF}")
    print(f"{GREY}   saved {path}{OFF}")
    play(path)


def main() -> int:
    quiet = "-q" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if not CHECKPOINT.exists():
        print(f"{RED}No trained model at {CHECKPOINT}{OFF}")
        print("Train one first:\n  python -m src.dl.train --system D1 --joint-durations")
        return 1

    print(f"\n{BLUE}{BOLD}  Phonetic-Guided Speech Generation — live demo{OFF}")
    print(f"{GREY}  Nothing below is a recording. Every word is generated from text.{OFF}\n")
    print(f"{GREY}  loading the model and vocoder…{OFF}", end="", flush=True)

    from src.dl.config import device as best_device
    from src.dl.mel_stats import MelStats
    from src.dl.speak import load_model

    device = best_device()
    model, vocab = load_model(CHECKPOINT, "D1", device)
    stats = MelStats.load()
    # First call pays the lazy vocoder/CUDA setup cost; do it silently so the demo
    # does not open by saying a stray word out loud.
    say("ready", model, vocab, stats, device, 0, silent=True)
    print(f"\r{GREEN}  ready{OFF} {GREY}(device: {device}){OFF}                         \n")

    index = 1
    for text in (args or SENTENCES):
        say(text, model, vocab, stats, device, index)
        index += 1
        print()

    if args or quiet:
        return 0

    print(f"{BLUE}  Type a sentence and press Enter. Blank line or Ctrl-C to finish.{OFF}\n")
    while True:
        try:
            text = input(f"{BOLD}  say > {OFF}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        say(text, model, vocab, stats, device, index)
        index += 1
        print()

    print(f"{GREY}  audio saved in {OUT_DIR}/{OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
