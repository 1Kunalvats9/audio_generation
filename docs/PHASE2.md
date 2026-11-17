# Phase 2 — Deep Learning · work plan

> **Phase 1 answered a question. Phase 2 acts on the answer.**
>
> We measured that the acoustic model costs **21.17 points of WER and 2.32 MOS**, while duration
> prediction costs nothing measurable. So phase 2 replaces the acoustic model — not the duration
> model — with a neural network.

---

## 0. "We need a model that generates audio now, right?"

Half right, and the distinction matters for the report.

**We already generate audio.** A1 and A2 are synthesised waveforms. What was never *learned* is the
acoustic model: phase 1 used 70 averaged phoneme templates — a lookup table, not a network.

Phase 2 replaces that averaging with a **trained neural generator**. The input (phonemes +
durations) and the output (an 80-band mel, then the same frozen vocoder) stay exactly the same.
Only the box in the middle changes.

```
phase 1:  phonemes + durations ->  [ 70 averaged templates ]  -> mel -> HiFi-GAN -> audio
phase 2:  phonemes + durations ->  [  NEURAL GENERATOR     ]  -> mel -> HiFi-GAN -> audio
```

Keeping everything else identical is deliberate: it means phase 2's numbers are **directly
comparable to phase 1's**, and A1/A2 become the baselines we have to beat.

---

## 1. The one real decision: which model

Our title says *latent diffusion*. There are three routes, and we should take two of them in order.

| | Model | Trains in | Risk | Matches title |
| --- | --- | --- | --- | --- |
| **D1** | Deterministic decoder (FastSpeech 2 style): conv/transformer stack, predicts the mel directly, L1 loss | hours | **low** | no |
| **D2** | Mel-space diffusion (Grad-TTS style): U-Net denoiser over the 80×T mel | ~a day | medium | mostly |
| — | True latent diffusion (AudioLDM style): train a VAE, then diffuse in its latent space | days, two models | **high** | exactly |

### Recommendation: build D1 first, then D2. Do not start with diffusion.

**Why this order is the whole risk strategy.** D1 is roughly 200 lines and will almost certainly
work. Once it exists we are guaranteed a submittable phase 2 with a real neural model and a real
comparison against A1/A2. D2 is then the ambitious headline result, attempted from a position of
safety. If D2 underperforms or will not converge, we still have a complete project and we report
the negative result honestly — which phase 1 already showed us how to do.

Starting with full latent diffusion (VAE **and** diffusion, two models, two sets of failure modes,
on student laptops) is how teams end up with nothing to submit.

**On the title.** A mel spectrogram *is* a compressed representation of audio, so mel-space
diffusion is defensible as "latent". If we get D2 working we say so precisely: diffusion in mel
space, with true latent diffusion (AudioLDM) named as the next step. Phase 1's report already
takes exactly this line.

### The resulting ladder for the report

```
A1   templates + real durations        23.65% WER   1.84 MOS   <- phase 1
A2   templates + predicted durations   21.15% WER   1.76 MOS   <- phase 1
D1   neural decoder                         ?            ?     <- phase 2
D2   mel diffusion                          ?            ?     <- phase 2
```

Four systems on one axis, each changing one thing. This continues phase 1's ablation rather than
starting a separate experiment, and it is the strongest possible structure for the report.

---

## 2. Hardware — settle this in week 1

Training a neural net on 7.4 million mel frames is not like phase 1, where nothing was trained on
audio at all (sklearn on a table took one minute; the vocoder and Whisper were pre-trained).

| Machine | Status |
| --- | --- |
| Kunal's Mac (arm64, 8 cores) | **MPS works**, CUDA does not. Fine for writing and debugging; slow for full training |
| The NVIDIA laptop in the team | **This is our training machine.** Install the CUDA build from pytorch.org and confirm `torch.cuda.is_available()` |
| Google Colab / Kaggle | Free T4/P100. Kaggle gives ~30 GPU-hours a week. **Set this up as the fallback in week 1, not week 3** |

**Rule:** everyone develops on their own machine against a tiny subset (say 200 clips) so the code
runs anywhere in seconds. Only full training runs go to the GPU machine. Never let the team block
on one laptop.

---

## 3. What carries over (most of the hard work is done)

This is why phase 2 is tractable at all:

- **`data/cache/phones.csv`** — 949,856 aligned phone durations. This is the conditioning signal a
  neural TTS model needs, and it is committed to git.
- **`data/cache/mels/*.npy`** — 13,100 cached mels, 1.1 GB, the training targets. One command to
  regenerate.
- **The frozen HiFi-GAN vocoder** — unchanged, so D1/D2 stay comparable to A1/A2.
- **The splits** — train 12,501 / val 300 / test 299 / test_100, seed 42, source-disjoint.
- **`configs/base.yaml` + `src/config.py`** — the frozen numeric contract. Phase 2 adds to it, never
  edits it.
- **The whole evaluation harness** — WER, CER, MOS, significance tests, failure log. M5 adds two
  rows and everything else just works.

**We are not starting from scratch. We are swapping one component.**

---

## 4. New file contracts (freeze these before writing code)

Phase 1 worked because five people coded against fixed formats. Same discipline:

| File | Owner | What it is |
| --- | --- | --- |
| `configs/dl.yaml` | M1 | All phase-2 hyper-parameters. Nothing hard-coded in a model file |
| `data/cache/phone_vocab.json` | M1 | The 70 phone symbols → integer IDs. **Frozen.** Everyone imports it |
| `data/cache/mel_stats.json` | M1 | Mel mean/std over the training split, for normalisation |
| `src/dl/dataset.py` | M1 | Yields `(phone_ids, durations, mel, lengths)` batches, padded |
| `src/dl/encoder.py` | M2 | `(B, P) phone ids -> (B, P, H)` encoder outputs |
| `src/dl/duration_predictor.py` | M3 | `(B, P, H) -> (B, P)` predicted log-durations + the length regulator |
| `src/dl/decoder.py` | M4 | `(B, T, H) -> (B, 80, T)` the D1 generator |
| `src/dl/diffusion.py` | M4 | The D2 denoiser, noise schedule and sampler |
| `src/dl/train.py` | M4 | Training loop, checkpointing, resume |
| `outputs/audio/D1/`, `D2/` | M4 | Synthesised wavs, same layout as A1/A2 |
| `outputs/results/train_log.csv` | M5 | step, train loss, val loss, learning rate |

**Tensor shape conventions, agreed once:** batch first, mels as `(B, 80, T)`, phoneme sequences as
`(B, P)`, padding mask `True` = real frame. Writing these down now prevents a week of transpose bugs.

---

## 5. Who does what

### M1 — Kunal — Data and the training pipeline
Owns: `src/dl/dataset.py`, `src/dl/vocab.py`, `configs/dl.yaml`, mel statistics.

1. Build `phone_vocab.json` from the 70 symbols in `phones.csv`. Freeze it and tell the team.
2. Compute mel mean/std over the **training split only** (not all 13,100 — that leaks).
3. Write the `Dataset` and `DataLoader`: loads a clip's mel, its phone IDs and its durations,
   pads a batch to the longest member, returns a mask. Bucket by length so batches are not 80%
   padding.
4. **Grapheme-to-phoneme for unseen text.** Phase 1 could only synthesise clips we already had
   alignments for. A real demo needs to say a sentence nobody recorded. Use the MFA dictionary we
   already have, plus `g2p_en` for out-of-vocabulary words.
5. A 200-clip debug subset so everyone's code runs in seconds on a laptop.

### M2 — Tanishk — Text encoder and conditioning
Owns: `src/dl/encoder.py`.

1. Phoneme embedding (70 symbols → H dims) plus positional encoding.
2. An encoder stack: start with 4 conv layers + layer norm; move to a small transformer only if it
   helps. Simple first.
3. The **length regulator**: given encoder outputs `(B, P, H)` and durations `(B, P)`, repeat each
   phoneme's vector by its duration to produce `(B, T, H)`. This is the piece that turns a
   per-phoneme sequence into a per-frame one, and it is the heart of FastSpeech 2's design.
4. Unit-test it the way phase 1 tested everything: a hand-made 5-phoneme example where you can
   verify the output frame count equals the sum of the durations.

### M3 — Pushkar + Kunal — Neural duration predictor
Owns: `src/dl/duration_predictor.py`, and the phase-1 comparison.

1. A small network (2 conv layers + projection) that predicts log-duration per phoneme from the
   encoder output — the FastSpeech 2 variance adaptor.
2. Train it jointly with the rest, using MFA durations as the target.
3. **Compare against phase 1's 19.65 ms MAE on the same test split.** This is a clean, directly
   reportable result: does a jointly-trained neural predictor beat engineered features and boosted
   trees? Either answer is interesting.
4. Keep `src/baseline/duration_models.py` untouched so the comparison stays honest.

### M4 — Shaurya — The acoustic generator  ← the critical path
Owns: `src/dl/decoder.py`, `src/dl/diffusion.py`, `src/dl/train.py`.

1. **D1 first.** A decoder that maps `(B, T, H)` to `(B, 80, T)`, trained with L1 loss against the
   real mel. Conv stack with residual connections is enough. Get this converging before anything
   else.
2. Training loop: AdamW, gradient clipping, checkpoint every N steps, **resume from checkpoint**
   (laptops sleep, Colab disconnects — this is not optional).
3. Synthesise D1 into `outputs/audio/D1/` for the same 100 `test_100` clips.
4. **Then D2.** A U-Net denoiser over the mel, conditioned on the same `(B, T, H)`, with a cosine
   noise schedule. Sample with 50 steps to start; tune later.
5. Sanity gate before any full run: **overfit 10 clips deliberately.** If the model cannot
   reproduce 10 training examples nearly perfectly, something is wired wrong and a 12-hour run
   will only waste a day.

### M5 — Chaitanya — Evaluation and experiment tracking
Owns: `src/eval/mcd.py`, extensions to the phase-1 harness, `notebooks/04_phase2_demo.ipynb`.

1. Add `D1` and `D2` to `SYSTEM_DIRS` in `src/eval/whisper_wer.py` and to `SYSTEMS` in
   `make_results.py`. That is genuinely most of the work — the harness was built to extend.
2. **Add MCD (mel cepstral distortion).** It was the wrong metric in phase 1 (averaged templates
   would just score "very different"), but now that we predict mels directly it measures exactly
   what the model is optimising. Explain that change of reasoning in the report.
3. Log training curves to `outputs/results/train_log.csv` and plot train vs val loss — the
   standard evidence that we are not overfitting.
4. Re-run the listening study with D1/D2 added. **Make it blind this time** (shuffle filenames,
   hide which system is which) — phase 1's report flags non-blindness as a limitation, and fixing
   it is cheap.
5. Rerun WER and the full results table so every system sits in one comparison.

---

## 6. Order of work

```
M1 vocab + dataset ──┬──> M2 encoder ──> M3 duration predictor ──┐
                     │                                           ├──> M4 D1 ──> M4 D2 ──> M5 eval
                     └──> M4 decoder skeleton ───────────────────┘
```

- **M1 is the critical path for week 1.** Nobody can train anything without the dataset and vocab.
- M2 and M4 can develop in parallel against fake tensors of the agreed shapes — do not wait.
- M3 plugs in after M2's encoder exists.
- M5 can extend the harness on day one using A1/A2 as stand-ins for D1/D2.

---

## 7. Milestones

| | Goal | Done when |
| --- | --- | --- |
| **W1** | Infrastructure | vocab frozen, dataset yields a correct batch, GPU machine confirmed, 200-clip subset runs end to end |
| **W2** | D1 overfits | model reproduces 10 training clips almost exactly; loss curve falls |
| **W3** | D1 trained | full training run finished, `outputs/audio/D1/` populated, WER + MCD computed |
| **W4** | D2 attempted | diffusion converges on the subset, then full run if time allows |
| **W5** | Write-up | report, deck, blind listening study, final results table |

**If W4 slips, we submit with D1 only and report D2 as attempted-and-why-it-did-not-finish.** That
is an acceptable outcome and far better than a half-trained model nobody can explain.

---

## 8. Risks, and what we do about them

| Risk | Mitigation |
| --- | --- |
| No usable GPU | Confirm the NVIDIA laptop **and** set up Kaggle in week 1. Do not discover this in week 3 |
| Diffusion will not converge | D1 exists first, so we always have something to submit |
| Shape/padding bugs eat a week | Freeze tensor conventions now (§4); unit-test the length regulator |
| Training run lost to a sleeping laptop | Checkpoint + resume from the first commit, not retrofitted |
| Model memorises the 12,501 training clips | We already have a source-disjoint test split; report train vs val loss |
| We run out of time writing up | The report and deck are reused from phase 1 — same LaTeX, same generator, new numbers |

---

## 9. Day-one checklist

- [ ] **Everyone:** `git pull`, confirm `make test` passes (170 tests)
- [ ] **Everyone:** run `torch.cuda.is_available()` and post the result in the group — we need to
      know today who has a GPU
- [ ] **Kunal:** freeze `phone_vocab.json` and `configs/dl.yaml`, announce the tensor conventions
- [ ] **Tanishk:** write the length regulator and its unit test against fake tensors
- [ ] **Pushkar:** read FastSpeech 2 §2.2 (the variance adaptor) — that is the component we are building
- [ ] **Shaurya:** decoder skeleton that takes `(B, T, H)` and returns `(B, 80, T)` of the right shape,
      even if the weights are random
- [ ] **Chaitanya:** add `D1`/`D2` to the harness and confirm it skips them cleanly when the folders
      are empty

---

## 10. What phase 2's report will argue

Phase 1 found that the acoustic model dominates. Phase 2 tests the obvious consequence:

> **If we replace the averaged templates with a learned generator, does the 21-point gap close?**

That is a real hypothesis derived from our own measurement, not a feature request. Whatever the
answer, we can report it honestly — and the fact that phase 1 *predicted where to look* is the
strongest thing we can say about the project as a whole.

**Reference:** Ren et al., *FastSpeech 2*, ICLR 2021 — <https://arxiv.org/abs/2006.04558>.
Sections 2.2 (variance adaptor) and 2.3 (the decoder) describe almost exactly the architecture D1
implements.
