# Shaurya — Phase 2 acoustic generator

The assignment in `docs/PHASE2.md` is M4: D1, resumable training,
synthesis, and then D2. The existing encoder, frozen audio configuration,
vocabulary, mel statistics, splits and baseline models are reused unchanged.

## Implemented

- `src/dl/decoder.py`: residual temporal convolution decoder, frame positional
  encoding, per-frame LayerNorm, projection to `(B, 80, T)`, masked L1 loss.
  Frame positions distinguish repeated vectors within a single phoneme.
- `src/dl/train.py`: encoder/regulator/generator composition, AdamW, configured
  warmup and gradient clipping, frame-weighted validation, CSV logging,
  atomic checkpoints, and resume of model, optimizer, RNG and minibatch cursor.
  Config/vocabulary/statistics hashes prevent resuming with changed contracts.
- `src/dl/synthesize.py`: complete split preflight, GBT/MFA/neural duration
  options, ordered phone matching, mel denormalisation, frozen HiFi-GAN,
  original sample-count trimming, and `outputs/audio/D1/` or `D2/` output.
  Oracle MFA-duration outputs default to `D1_oracle/` or `D2_oracle/`.
- `src/dl/diffusion.py`: conditional temporal U-Net, cosine forward schedule,
  masked noise-prediction MSE and a DDIM sampler with 50 default steps.
  Downsampling explicitly pads odd lengths to preserve padding independence.

M3's duration module is absent in this checkout. The acoustic generators train
with true MFA durations; `--joint-durations` can compose M3's documented module
once it exists. GBT durations are the default at synthesis, keeping that part
of the comparison consistent with A2. Neural duration inference requires a
checkpoint trained with `--joint-durations`.

## Run D1 in order

Use the repository root and the environment created by `make install`.
The checkpoint directory is a runtime artifact; do not commit weights or data.

```bash
source .venv/bin/activate
python -m src.dl.config
make test

# Restore the team's LJSpeech data and caches, or regenerate the missing pieces:
make data metadata mels
# Keep the committed phone_vocab.json and mel_stats.json unchanged.
# debug_200.txt is already committed; no dl-prep is needed here.

# Deliberately memorise ten TRAINING examples, before launching a full run.
python -m src.dl.train --overfit --steps 2000 --batch-size 10
# The gate passes only if final validation L1 <= 10% of initial L1.
# If it has not passed, inspect the loss curve and continue the same run:
python -m src.dl.train --overfit --steps 5000 --batch-size 10 \
  --resume outputs/checkpoints/D1/overfit/latest.pt

# This starts a fresh full model, after checking the passed sanity checkpoint.
python -m src.dl.train --steps 60000
# Resume to an ABSOLUTE total step count, with the same split and batch size:
python -m src.dl.train --steps 60000 \
  --resume outputs/checkpoints/D1/full/latest.pt

# Generate phase-1 GBT durations if outputs/durations/test_pred.csv is absent.
make durations
python -m src.dl.synthesize \
  --checkpoint outputs/checkpoints/D1/full/latest.pt
```

The regular D1 log is `outputs/results/train_log.csv`, with
`step,train_loss,val_loss,lr`. A blank validation field means validation was
not scheduled at that step. Overfit logs live beside their checkpoints;
D2 logs are separate, so they cannot overwrite D1's experiment.
The latest checkpoint and periodic `step_XXXXXX.pt` snapshots include the
optimizer and random states. Loading uses Python checkpoint serialization:
use checkpoints created locally or received from the trusted team.

For an explicit short development run after the sanity gate, use
`--split debug_200 --steps 1000`. For a new experiment, select distinct
`--checkpoint-dir` and `--log` paths. Resume with the same data, architecture,
learning rate, warmup and batching. `--device cpu|mps|cuda` overrides detection.
The ten-clip gate deliberately uses the same examples for optimisation and
validation; the full run instead validates on the existing held-out `val` split.
Synthesis refuses an overfit checkpoint for held-out evaluation.

## D2 comes after D1

First establish D1's real-data overfit, full-training and audio results.
Then reuse the same data and contracts for diffusion:

```bash
python -m src.dl.train --system D2 --overfit --steps 5000 --batch-size 10
python -m src.dl.train --system D2 --steps 60000
python -m src.dl.train --system D2 --steps 60000 \
  --resume outputs/checkpoints/D2/full/latest.pt
python -m src.dl.synthesize \
  --checkpoint outputs/checkpoints/D2/full/latest.pt --sample-steps 50
```

D2 uses the same ten-clip relative loss gate, measured using fixed validation
noise/timesteps. Loss is noise MSE, so compare its curve with D2's initial loss,
not D1's L1. Inspect generated mels/audio as well before a full run. No D2
quality or convergence claim follows from having an implemented sampler.

## Verification and remaining experiment work

`make test`: **259 passed, 7 skipped**. The skips belong to the missing M3
duration predictor and M5 MCD modules. Both CLI help commands also run.

CPU tests cover all existing decoder contracts, loss masking (including NaNs
in padding), optimisation, encoder gradients, exact interrupted-versus-
uninterrupted training, checkpoint compatibility, waveform trimming and mel
denormalisation. Diffusion tests check the cosine schedule, noise-prediction
loss, gradients, odd-length padding independence and finite sampled outputs. D1 and D2 also
passed forward/backward smoke checks on this Mac's MPS device. Synthesis
seeds Python, NumPy and torch with the shared seed before loading/sampling.

A deterministic ten-example **synthetic** integration experiment lowered
normalised mel L1 from **0.900686 to 0.039483** in 400 steps (95.6% reduction).
It uses small artificial targets and a smaller model to check wiring. It is
also included as a regression test. It does **not** certify the ten-real-clip
gate or speech quality, and its temporary checkpoints are discarded.

This checkout has committed alignments/statistics but **no LJSpeech raw audio,
metadata_clean.csv, cached mels or trained neural weights**. Consequently the
real ten-clip experiment, full training and generation of the 100 D1/D2 wavs
remain pending data restoration and training time. No WER, MCD or listening
scores are invented. M5 can run evaluation once those output folders exist.
