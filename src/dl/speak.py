"""Say a sentence nobody recorded  (phase 2 demo).

The whole pipeline, end to end, from a string to a wav file:

    text  ->  G2P  ->  phoneme ids  ->  encoder  ->  duration predictor
                                            |              |
                                            +-> length regulator <-+
                                                     |
                                            decoder -> mel -> HiFi-GAN -> wav

This is the only entry point that is **fully autonomous**. Everywhere else in the
project, predicted durations are rescaled so the clip matches the true recording's
length -- oracle information a real system would not have. Here there is no
recording, so the predicted durations are used exactly as the model produced them
and they alone decide how long the audio is.

Usage:
    python -m src.dl.speak "Hello, this is our project." -o samples/demo.wav
    python -m src.dl.speak --checkpoint outputs/checkpoints/D1/full/latest.pt "Some text."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.audio import mel_to_wav, save_wav
from src.config import OUTPUTS, SECONDS_PER_FRAME
from src.dl.config import device as best_device
from src.dl.g2p import G2P
from src.dl.mel_stats import MelStats
from src.dl.vocab import PhoneVocab

DEFAULT_CHECKPOINT = OUTPUTS / "checkpoints" / "D1" / "full" / "latest.pt"
MIN_FRAMES = 1


def load_model(checkpoint=DEFAULT_CHECKPOINT, system: str = "D1", device: str = "cpu"):
    """Load a jointly trained acoustic model (it must carry a duration predictor)."""
    from src.dl.train import AcousticModel, load_checkpoint

    vocab = PhoneVocab.load()
    model = AcousticModel(vocab.size, system, joint_durations=True).to(device)
    load_checkpoint(Path(checkpoint), model, target_device=device)
    model.eval()
    if model.duration_predictor is None:
        raise ValueError(
            f"{checkpoint} has no duration predictor. Synthesising unseen text needs one — "
            "train with --joint-durations.")
    return model, vocab


@torch.no_grad()
def speak(text: str, model, vocab, stats=None, device: str = "cpu",
          speed: float = 1.0, sample_steps: int | None = None):
    """-> (waveform, phonemes, durations in frames). Durations are the model's own."""
    stats = stats or MelStats.load()
    g2p = G2P(vocab=vocab)
    phonemes = g2p.phonemes(text)
    ids = torch.tensor(vocab.encode(phonemes), device=device)[None]
    mask = torch.ones_like(ids, dtype=torch.bool)

    enc = model.encoder(ids, mask)
    log_dur = model.duration_predictor(enc, mask)
    frames = torch.exp(log_dur) / max(speed, 1e-3)
    # No recording to rescale against, so the prediction stands as-is; only the
    # physical floor of one frame per phone is enforced.
    durations = frames.round().clamp(min=MIN_FRAMES).long()

    expanded, frame_mask = model.regulator(enc, durations)
    if model.spec["system"] == "D1":
        mel_norm = model.generator(expanded, frame_mask)
    else:
        mel_norm = model.generator.sample(expanded, frame_mask, steps=sample_steps)

    mel = stats.denormalise(mel_norm[0].cpu())      # vocoder wants RAW log-mels
    wav = mel_to_wav(mel)
    return wav, phonemes, durations[0].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="+", help="the sentence to speak")
    ap.add_argument("-o", "--output", type=Path, default=Path("samples/demo.wav"))
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--system", default="D1", choices=["D1", "D2"])
    ap.add_argument("--device", default=best_device())
    ap.add_argument("--speed", type=float, default=1.0,
                    help=">1 speaks faster by shortening every predicted duration")
    args = ap.parse_args()

    text = " ".join(args.text)
    if not args.checkpoint.exists():
        print(f"missing {args.checkpoint}\nTrain first: "
              "python -m src.dl.train --system D1 --joint-durations", file=sys.stderr)
        return 1

    model, vocab = load_model(args.checkpoint, args.system, args.device)
    wav, phonemes, durations = speak(text, model, vocab, device=args.device, speed=args.speed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_wav(args.output, wav)

    seconds = durations.sum() * SECONDS_PER_FRAME
    print(f'text      "{text}"')
    print(f"phonemes  {' '.join(phonemes)}")
    print(f"durations {len(durations)} phones, {durations.sum()} frames = {seconds:.2f} s")
    print(f"           shortest {durations.min()}, longest {durations.max()} frames")
    print(f"\nwrote {args.output}  ({len(wav) / 22050:.2f} s of audio)")
    print("Durations are the model's own — nothing was rescaled to a real recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
