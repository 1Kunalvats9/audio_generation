"""Mel cepstral distortion  (M5 — phase 2).

MCD measures how far a synthesised spectrum sits from the real one, in dB. It
is the standard objective measure for acoustic models.

WHY THIS METRIC ONLY ARRIVES IN PHASE 2. In phase 1 the acoustic model was 70
averaged phoneme templates, so MCD would only ever have reported "very
different" — it could not distinguish a good template system from a bad one,
and it says nothing about whether speech is intelligible, which was phase 1's
question. Now that D1 and D2 predict mel spectrograms directly, MCD measures
precisely the quantity being optimised, and it is far more sensitive than WER
to the kind of improvement a better acoustic model produces.

    MCD = (10 / ln 10) * sqrt( 2 * sum_d (c_d^a - c_d^b)^2 )

averaged over frames, where c are mel cepstral coefficients: the DCT of the
log-mel spectrum. c_0 is dropped because it is overall loudness, not spectral
shape, and we do not want a quiet system penalised for being quiet.

Usage:
    python -m src.eval.mcd                     # score every system that has audio
    from src.eval.mcd import mel_cepstral_distortion
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.fftpack import dct

from src.config import N_MELS, OUTPUTS, SECONDS_PER_FRAME

N_COEFF = 13            # c1..c13, the usual choice for MCD
_SCALE = (10.0 / np.log(10.0)) * np.sqrt(2.0)
RESULTS = OUTPUTS / "results"
MCD_CSV = RESULTS / "mcd.csv"


def mel_cepstrum(log_mel: np.ndarray, n_coeff: int = N_COEFF) -> np.ndarray:
    """(n_mels, T) log-mel -> (n_coeff, T) cepstral coefficients, c0 dropped."""
    log_mel = np.asarray(log_mel, dtype=np.float64)
    if log_mel.ndim != 2:
        raise ValueError(f"expected a 2-D (n_mels, T) array, got shape {log_mel.shape}")
    ceps = dct(log_mel, type=2, axis=0, norm="ortho")
    return ceps[1:n_coeff + 1]


def mel_cepstral_distortion(pred: np.ndarray, true: np.ndarray,
                            n_coeff: int = N_COEFF) -> float:
    """MCD in dB between two log-mel spectrograms of the same shape."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}. "
                         "MCD needs frame-aligned spectrograms of equal length.")
    if pred.shape[1] == 0:
        raise ValueError("cannot score an empty spectrogram")

    a = mel_cepstrum(pred, n_coeff)
    b = mel_cepstrum(true, n_coeff)
    per_frame = _SCALE * np.sqrt(((a - b) ** 2).sum(axis=0))
    return float(per_frame.mean())


# ---------------------------------------------------------------------------
# Scoring whole systems
# ---------------------------------------------------------------------------


def _mel_of_wav(path) -> np.ndarray:
    """Re-analyse a synthesised waveform with the SAME mel recipe as training."""
    import soundfile as sf

    from src.audio import wav_to_mel
    wav, _ = sf.read(str(path), dtype="float32")
    return wav_to_mel(wav).numpy()          # wav_to_mel takes a numpy array


def predicted_mels(system: str, clip_ids, checkpoint=None, device: str = "cpu"):
    """Yield (clip_id, predicted_mel) — the ACOUSTIC MODEL's own output.

    Scoring re-analysed audio instead would mostly measure the vocoder: a real
    mel pushed through HiFi-GAN and re-analysed comes back with a log-mel MAE
    of about 1.4, which is larger than the gap between the acoustic models we
    are trying to compare. So MCD is computed before the vocoder, where it
    isolates the thing the model actually predicts.
    """
    import pandas as pd

    from src.audio import adjust_durations
    from src.config import CACHE_DIR, METADATA_CLEAN

    meta = pd.read_csv(METADATA_CLEAN, dtype={"id": str}).set_index("id")
    wanted = set(clip_ids)

    if system in ("A1", "A2"):
        from src.baseline.synthesize_baseline import synthesize_from_templates

        templates = dict(np.load(CACHE_DIR / "templates.npz"))
        phones = pd.read_csv(CACHE_DIR / "phones.csv", dtype={"clip_id": str, "word": str})
        phones = phones[phones.clip_id.isin(wanted)]
        pred_csv = OUTPUTS / "durations" / "test_pred.csv"
        pred_dur = (pd.read_csv(pred_csv, dtype={"clip_id": str})
                    if system == "A2" and pred_csv.exists() else None)

        for clip_id, g in phones.groupby("clip_id"):
            g = g.sort_values("idx").copy()
            if system == "A2":
                if pred_dur is None:
                    continue
                rows = pred_dur[pred_dur.clip_id == clip_id].sort_values("idx")
                if len(rows) != len(g):
                    continue
                target = int(meta.loc[clip_id, "n_frames"])
                g["n_frames"] = adjust_durations(
                    rows["pred_best"].to_numpy(np.float32), target)
            yield clip_id, synthesize_from_templates(g, templates).astype(np.float64)
        return

    if system in ("D1", "D2"):
        if checkpoint is None:
            raise ValueError(f"{system} needs --checkpoint to regenerate its predicted mels")
        import torch

        from src.dl.mel_stats import MelStats
        from src.dl.synthesize import prepare_clips, predict_mel
        from src.dl.train import AcousticModel, load_checkpoint
        from src.dl.vocab import PhoneVocab

        vocab = PhoneVocab.load()
        stats = MelStats.load()
        # A checkpoint trained with --joint-durations carries a duration predictor,
        # and load_checkpoint rejects an architecture mismatch, so match it.
        model = None
        for joint in (True, False):
            candidate = AcousticModel(vocab.size, system, joint_durations=joint).to(device)
            try:
                load_checkpoint(checkpoint, candidate, target_device=device)
            except ValueError:
                continue
            model = candidate
            break
        if model is None:
            raise ValueError(f"{checkpoint} does not match a {system} model in either "
                             "joint-duration configuration")
        model.eval()
        for clip in prepare_clips(sorted(wanted), vocab, "mfa"):
            ids = torch.tensor(clip["ids"], device=device)[None]
            dur = torch.as_tensor(clip["durations"], device=device)[None]
            with torch.no_grad():
                mel = predict_mel(model, ids, dur)[0].cpu().numpy()
            yield clip["clip_id"], stats.denormalise(mel).astype(np.float64)
        return

    raise ValueError(f"no predicted-mel path for system {system!r}")


def score_system(system: str, clip_ids, checkpoint=None, device: str = "cpu") -> dict:
    """Mean MCD for one system's predicted mels against the real cached ones."""
    from src.config import MEL_DIR

    scores, missing = [], 0
    for clip_id, pred in predicted_mels(system, clip_ids, checkpoint, device):
        ref = MEL_DIR / f"{clip_id}.npy"
        if not ref.exists():
            missing += 1
            continue
        true = np.load(ref).astype(np.float64)
        n = min(pred.shape[1], true.shape[1])
        scores.append(mel_cepstral_distortion(pred[:, :n], true[:, :n]))

    return {"system": system, "n_clips": len(scores), "missing": missing,
            "mcd_db": float(np.mean(scores)) if scores else float("nan")}


def chance_level(clip_ids, seed: int = 42) -> float:
    """MCD between unrelated real clips — the "no relationship" reference.

    Our coefficients are the DCT of an 80-band log-mel rather than the SPTK
    mel-generalised cepstrum most papers use, so the absolute numbers are not
    comparable to a published 4-8 dB. This control fixes the scale: it is what
    a system scores when its output has nothing to do with the target.
    """
    import numpy as np

    from src.config import MEL_DIR

    rng = np.random.default_rng(seed)
    ids = [c for c in clip_ids if (MEL_DIR / f"{c}.npy").exists()]
    scores = []
    for a, b in zip(ids, rng.permutation(ids)):
        if a == b:
            continue
        x = np.load(MEL_DIR / f"{a}.npy").astype(np.float64)
        y = np.load(MEL_DIR / f"{b}.npy").astype(np.float64)
        n = min(x.shape[1], y.shape[1])
        scores.append(mel_cepstral_distortion(x[:, :n], y[:, :n]))
    return float(np.mean(scores))


def main() -> int:
    import argparse
    from pathlib import Path

    import pandas as pd

    from src.utils import read_filelist

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d1-checkpoint", type=Path)
    ap.add_argument("--d2-checkpoint", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    clip_ids = read_filelist("test_100")
    ckpt = {"D1": args.d1_checkpoint, "D2": args.d2_checkpoint}
    rows = [{"system": "chance", "n_clips": len(clip_ids), "missing": 0,
             "mcd_db": chance_level(clip_ids)}]
    print(f"  chance (unrelated real clips): MCD {rows[0]['mcd_db']:.2f} dB")
    for system in ("A1", "A2", "D1", "D2"):
        if system in ckpt and ckpt[system] is None:
            print(f"  {system}: no checkpoint given, skipping")
            continue
        try:
            r = score_system(system, clip_ids, ckpt.get(system), args.device)
        except FileNotFoundError as e:
            print(f"  {system}: skipping ({e})")
            continue
        if r["n_clips"] == 0:
            print(f"  {system}: nothing to score")
            continue
        rows.append(r)
        print(f"  {r['system']}: MCD {r['mcd_db']:.2f} dB over {r['n_clips']} clips")

    if not rows:
        print("nothing scored.", file=sys.stderr)
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(MCD_CSV, index=False)
    print(f"\nwrote {MCD_CSV.relative_to(OUTPUTS.parent)}")
    print("Lower is better. Scored on the predicted mel, before the vocoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
