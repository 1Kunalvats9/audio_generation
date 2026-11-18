# The deep learning part, explained

*For the five of us. Plain English, and for every choice: what we did, why, and why not the
obvious alternative. If you can explain this document you can answer the viva.*

---

## Contents

1. [What phase 2 actually changed](#1-what-phase-2-actually-changed)
2. [The pipeline, piece by piece](#2-the-pipeline-piece-by-piece)
3. [How training works](#3-how-training-works)
4. [The metrics — and why not the other ones](#4-the-metrics--and-why-not-the-other-ones)
5. [The results, and what they prove](#5-the-results-and-what-they-prove)
6. [What we are being honest about](#6-what-we-are-being-honest-about)
7. [Questions you will be asked](#7-questions-you-will-be-asked)

---

## 1. What phase 2 actually changed

**The one-sentence version:** phase 1 built speech out of 70 average sounds glued together;
phase 2 replaced that gluing with a neural network that *draws* the sound.

```
phase 1:  phonemes + durations → [ 70 averaged templates ] → spectrogram → vocoder → audio
phase 2:  phonemes + durations → [   NEURAL NETWORK      ] → spectrogram → vocoder → audio
                                  ^^^^^^^^^^^^^^^^^^^^^^^^
                                  only this box changed
```

**Why change only that box?** Because phase 1 *measured* that this box was the problem. It found
that swapping real durations for predicted ones cost nothing (p = 0.048 / 0.100, no clear effect),
while the templates cost 21 percentage points of word error rate. So phase 2 is not "let's try a
neural network" — it is a test of a specific prediction phase 1 made.

**Why keep everything else identical?** If we changed the vocoder or the durations at the same
time, and the result improved, we could not say *which* change did it. Holding everything fixed
is the entire reason the comparison means anything.

---

## 2. The pipeline, piece by piece

### 2.1 The phoneme vocabulary — `src/dl/vocab.py`

**What:** every phone symbol gets a number. 70 phones + padding = 71 ids.

**Why numbers at all?** A network cannot consume the string `"AH1"`. It needs an index into an
embedding table — a lookup of 71 rows, each a learned 192-number vector describing that sound.

**Why is padding id 0, specifically?** Two things come free. `nn.Embedding(padding_idx=0)` tells
PyTorch never to train that row, and `ids != 0` becomes a correct "this is a real phoneme" mask
without storing anything extra.

**Why freeze the file?** Because an embedding row only means something relative to the vocabulary
it was trained with. If someone rebuilt the vocabulary in a different order, row 34 would stop
meaning `HH`, and every saved checkpoint would silently produce nonsense. It is committed to git
so all five of us train against byte-identical ids.

### 2.2 Spectrograms — `src/audio.py`, `src/dl/mel_stats.py`

**What:** audio becomes a picture. Time across, frequency up (80 bands), brightness = energy.
One column = **11.61 ms**.

**Why 80 bands and not 40 or 256?** We did not choose freely — HiFi-GAN, the pre-trained vocoder,
was trained expecting exactly 80. Everything downstream has to match it.

**Why "mel" bands rather than plain frequency?** Human hearing resolves low pitches much more
finely than high ones. Mel spacing gives the low frequencies more rows, so the picture spends its
detail where ears actually notice.

**Why the logarithm?** Loudness is perceived multiplicatively — the step from a whisper to speech
feels like the step from speech to a shout, though the energies differ enormously. A log turns
that into even steps, which is also much easier for a network to fit.

**Why normalise, and why with the training split only?** Each of the 80 bands has a different
typical level, and networks initialise assuming inputs near mean 0, standard deviation 1. We
compute the mean and standard deviation of each band **over the 12,501 training clips only**.
Using all 13,100 would let information about the validation and test clips leak into training —
a small leak, but the kind an examiner will ask about.

**The trap:** `batch["mel"]` is normalised, but **HiFi-GAN expects raw log-mels**. Anything going
to the vocoder must pass through `MelStats.denormalise()` first. Getting this wrong produces
audio that sounds roughly right but is subtly degraded — you will not catch it by ear.

### 2.3 The dataset — `src/dl/dataset.py`

**What:** hands the model batches of `(phoneme ids, durations, spectrogram)`.

**Why padding and masks?** Clips have different lengths, but a tensor must be rectangular. So we
pad short clips with zeros and carry a mask saying which entries are real. Every later stage must
respect that mask or it learns from padding.

**Why "length bucketing"?** If a 100-frame clip shares a batch with a 900-frame clip, 89% of the
batch is zeros — wasted compute. Bucketing groups similar lengths together. We measured 15.6%
padding with it, instead of far more.

**Why it later caused a crash, honestly:** bucketing puts the *longest* clips in the same batch,
and that batch is the memory peak. A batch of 16 near-1000-frame clips exhausted the laptop GPU's
9 GB. We halved the batch to 8. Two things bucketing optimises — less padding, lower peak memory —
pull in opposite directions.

### 2.4 The encoder — `src/dl/encoder.py`

**What:** reads the phoneme sequence, outputs one 192-number vector per phoneme that knows about
its neighbours.

**Why does context matter?** A `T` before a pause is not the same sound as a `T` between two
vowels. The encoder's job is to let each phoneme's vector reflect what is around it.

**Why convolutions and not a transformer?** Convolutions were the simpler thing that could work:
fewer parameters, faster on a laptop, and well suited to local context, which is most of what
matters in phonetics. FastSpeech 2 uses self-attention, and that is the obvious upgrade — but
starting with the simpler model meant we got a working system before spending the compute budget.

**Why sinusoidal position encodings rather than learned ones?** They are fixed, free, and they
extrapolate to sequence lengths never seen in training. Learned positions would need every length
to appear during training.

**Why we mask padding before every convolution — the important bit.** A convolution reaches
across its kernel. Without masking, a padded slot bleeds into the real phonemes next to it, which
means **a clip's encoding depends on how long the other clips in its batch happened to be.** The
model would then behave differently at batch size 1 than at 16, and training would quietly
degrade. There is a test for exactly this: the same sequence padded to two different lengths must
encode identically.

### 2.5 The length regulator — the join that makes it all work

**What:** turns one vector *per phoneme* into one vector *per frame*, by repeating each phoneme's
vector as many times as its duration says.

```
phonemes:   [HH] [AH0] [L]          durations: 4, 7, 5
frames:     HH HH HH HH AH0 AH0 ... L L L L L     (16 frames)
```

**Why is this the heart of the design?** It is what makes durations a first-class *input* rather
than something the decoder must guess. This is exactly FastSpeech 2's mechanism, and it is why
the whole A1-vs-A2-vs-D1 comparison is even possible: we can hand the model any durations we like
and see what changes.

**Why not let attention figure out the alignment?** That is what Tacotron-style models do, and
it is precisely what makes them hard to control and prone to skipping or repeating words.
Explicit durations trade some flexibility for total control — and control is the point of our
experiment.

**Nice property:** padded phonemes carry duration 0, so they produce no frames and need no
special handling. Gradients also scale naturally — a phoneme that produced 7 frames receives 7
frames' worth of gradient.

### 2.6 The duration predictor — `src/dl/duration_predictor.py`

**What:** predicts how long each phoneme should last, so synthesis needs no forced aligner.

**Why does it exist if we already have MFA durations?** Because MFA needs the *audio* to produce
durations, and at synthesis time we only have text. This is FastSpeech 2's variance adaptor.

**Why predict the logarithm of duration?** Durations are extremely skewed: the median phone is 7
frames, but a silence can reach 481. With plain squared error on raw frames, the model would
spend nearly all its effort getting a handful of long pauses right and get ordinary sounds wrong.
A log makes the error *relative* instead of absolute. Phase 1's boosted trees did the same thing
for the same reason.

**Why train it jointly with the decoder instead of separately?** One run, one checkpoint, and the
encoder learns representations useful for both jobs. It is also what FastSpeech 2 does.

**And it lost to phase 1 — why report that?** 23.81 ms versus the boosted trees' 19.65 ms, 21%
worse. We report it because it is true and because it is informative: it says the classical
baseline was a real competitor, not a straw man. Three likely reasons: it trained as a *secondary*
loss for only 9,000 steps; it sees only the encoder's output, while the trees were handed 21
engineered features including distance-to-the-nearest-pause; and gradient boosting is simply very
strong on tabular problems.

### 2.7 The decoder, D1 — `src/dl/decoder.py`

**What:** frames in, spectrogram out. A stack of residual convolutions, then a projection to 80
bands.

**Why L1 loss (absolute error) rather than L2 (squared)?** Squared error punishes large errors
disproportionately, and the model responds by predicting something blurry and average — exactly
the failure phase 1's templates already demonstrated. L1 tolerates a few big misses in exchange
for sharper detail overall.

**Why "residual"?** Each layer learns a *correction* to its input rather than a whole new
representation. Deep stacks train far more reliably that way; it is the standard trick since
ResNet.

### 2.8 Diffusion, D2 — `src/dl/diffusion.py`

**What:** instead of drawing the spectrogram in one shot, start from pure noise and clean it up
over many small steps.

**Why would that be better?** One-shot models tend to produce the *average* plausible output,
which sounds muffled. Diffusion samples from the distribution instead, so it can commit to sharp
detail.

**Why cosine noise schedule?** It adds noise more gently early on, which empirically trains better
than the original linear schedule.

**Why build D1 first?** Risk. D1 is simple and almost certain to work; diffusion is finicky. With
D1 done we were guaranteed a submittable phase 2, and D2 became upside rather than a single point
of failure. This is the sequencing the plan committed to before any code was written.

**Why not full latent diffusion, which the project title mentions?** AudioLDM-style latent
diffusion needs a *second* trained model — a VAE to compress spectrograms into a latent space —
before the diffusion model can even start. Two models, two sets of failure modes, on laptops. We
diffuse in mel space instead (as Grad-TTS does) and name true latent diffusion as the next step.

### 2.9 The vocoder — pre-trained and frozen

**What:** HiFi-GAN turns the spectrogram back into an audible waveform.

**Why can't we just invert the spectrogram?** It throws away *phase* — the timing alignment of
the waveform. The picture tells you which frequencies are present and how strongly, but not how
they line up. A vocoder reconstructs a plausible phase.

**Why not train our own?** Two reasons. It needs far more compute than we have. More importantly,
**if the vocoder changed between systems, the comparison would be meaningless** — we could not
tell whether D1 beat A1 because of the acoustic model or because of a different vocoder. Freezing
it is experimental design, not laziness.

---

## 3. How training works

**Teacher forcing.** During training the model is given the *true* MFA durations, not its own
predictions. It learns "given these durations, draw this spectrogram". Only at synthesis does the
duration predictor supply them. Mixing this up — training on predicted durations — is a classic
TTS bug.

**Learning-rate warmup.** The learning rate starts near zero and ramps up over 2,000 steps. Large
steps on a randomly initialised network can destabilise it permanently.

> **This cost us an hour and is worth knowing.** Our first overfit check failed at a ratio of
> 0.36 and looked like a broken model. It was not: at step 400 the learning rate was still
> `6e-05` against a target of `3e-4` — the model had barely started. At 4,000 steps it passed
> comfortably at 0.080. **Always check the learning rate before concluding a model cannot learn.**

**The ten-clip gate.** Before any long run, the model must prove it can memorise ten clips —
final loss ≤ 10% of initial. If a model cannot overfit ten examples, something is miswired and a
multi-hour run will only waste the day. Ours passed: 0.9299 → 0.0743.

**Why the gate had to be relaxed for D2.** A diffusion model predicts *noise* at a *random*
timestep each step. Even a perfectly fitted diffusion model keeps a substantial loss, because the
target is partly random by construction. The 0.1 threshold was calibrated for D1's reconstruction
loss and simply does not transfer. We raised the threshold for D2 rather than deleting the check.

**Checkpoint and resume.** Written from the first commit, not retrofitted — laptops sleep and
cloud sessions disconnect. The checkpoint also stores a hash of the frozen contracts, so loading
a checkpoint whose vocabulary or mel statistics have changed fails loudly instead of silently
producing nonsense.

---

## 4. The metrics — and why not the other ones

An evaluation metric is just an agreed number for "how good is this", so comparisons are fair
instead of arguments. We use four, because each is blind to something the others catch.

### MAE — for durations

**What:** average of |predicted − actual|, in frames and milliseconds.

| Why not… | Reason |
| --- | --- |
| **RMSE** | Squares the errors, so a few 481-frame silences dominate the score. We *report* it, but never select on it |
| **MAPE** (% error) | Divides by the true value. A 1-frame phone makes it explode toward infinity |
| **R²** | Unitless. It can say "0.7" but not "we are 19.65 ms off", and milliseconds is what a human can judge |
| **Accuracy** | Wrong family — this is regression, not classification. There is no "correct class" |

**Why with and without silence?** Silence is 5.3% of phones but averages 14.6 frames against 7.4,
reaching 481. Left in, it dominates the headline. The no-silence number measures what the model
learned about *speech*; the inclusive one is what synthesis actually consumes.

### WER and CER — for intelligibility

**What:** (substitutions + deletions + insertions) ÷ words, after Whisper transcribes our audio.

| Why not… | Reason |
| --- | --- |
| **MOS alone** | Needs humans every time anything changes. Does not scale to 600 clips, not reproducible |
| **MCD alone** | Measures spectral *distance*, which says nothing about whether a listener could understand the words |
| **PESQ / STOI** | Built for telephony and noise degradation, not synthesis artifacts |

**Why CER as well?** WER is brutal — one wrong letter destroys a whole word. D1 scores 3.74% WER
but only 1.05% CER, which tells you the remaining errors are small mangles, not gibberish.

**The validation that matters:** we ran WER on the **real recordings** first. 1.85%. If that had
come out at 40%, the harness would be broken and every other number worthless. Always check the
ruler before measuring.

### MCD — for spectral accuracy (new in phase 2)

**What:** distance between the predicted and real spectrum, in dB.

**Why it only appears now.** In phase 1 the acoustic model was 70 averaged templates, so MCD
would only ever have reported "very different" — it could not distinguish a good template system
from a bad one. Now that the model predicts spectra directly, MCD measures exactly what is being
optimised, and it is far more sensitive than WER to the improvements a better acoustic model gives.
**Explaining why the right metric changed between phases is itself a good viva answer.**

**Why measured before the vocoder — we got this wrong first.** The obvious version scores the
synthesised *audio* by re-analysing it. We tried that and got A1 = 55.93 dB versus R1 = 56.40 —
A1 apparently beating a near-perfect copy, which is impossible. The cause: pushing a *real*
spectrogram through HiFi-GAN and re-analysing it returns a log-mel MAE of about **1.4**, which is
larger than the gap between the acoustic models we are comparing. It was measuring vocoder noise.
Scoring the predicted spectrogram *before* the vocoder fixed the ordering.

**Why a "chance" row?** Our coefficients come from the 80-band log-mel rather than the SPTK
cepstrum most papers use, so our numbers are not comparable to a published 4–8 dB. The chance row
(86.14 dB — two unrelated real clips) fixes the scale: it is what a system scores when its output
has nothing to do with the target. D1 at 30.14 is well under half of chance.

### MOS — for naturalness

**What:** 5 listeners × 5 clips, rated 1–5.

**Why bother when we have WER?** WER is completely blind to sounding robotic. Audio can score 0%
WER and still sound like a 1980s answering machine. These measure different things, and in our
phase 1 results they actually pointed in *opposite* directions on A1 vs A2 — which is precisely
why having both let us say honestly that neither difference was real.

### Wilcoxon and paired t-test — is the difference real?

**Why paired?** The same clips go through both systems, so pairing removes "this clip is hard" as
a confounder.

**Why Wilcoxon rather than just a t-test?** Per-clip WER is bounded at 0, heavily skewed and full
of ties — it breaks the t-test's normality assumption. Wilcoxon ranks instead of assuming a shape.

**Why report both when they disagree?** Phase 1's A1-vs-A2 gave p = 0.048 and p = 0.100. Quoting
only the one under 0.05 would be cherry-picking. Reporting both is what let us make the correct,
weaker claim: *no measurable effect*, rather than *predicted beats real*.

### Why hold out a whole book rather than random clips?

A random split would put sentences from the same book, recorded in the same session, on both
sides. Holding out all 299 clips of `LJ005` guarantees no test sentence shares a source document
with training. With a single speaker we cannot do the usual speaker-disjoint split, so this is
the strongest honest alternative.

---

## 5. The results, and what they prove

All on the same 100 clips from the held-out book `LJ005`.

| System | Acoustic model | Durations | WER | CER | MCD dB | MOS |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | — real recording | — | 1.85% | 0.71% | — | 4.88 |
| D1\* | **neural decoder** | measured | **2.37%** | 0.63% | — | — |
| R1 | real spectrogram | real | 2.48% | 0.95% | — | 4.16 |
| **D1** | **neural decoder** | predicted | **3.74%** | 1.05% | **30.14** | — |
| A2 | 70 averaged templates | predicted | 21.15% | 10.02% | 63.04 | 1.76 |
| A1 | 70 averaged templates | measured | 23.65% | 11.83% | 43.86 | 1.84 |
| D2 | mel diffusion | predicted | 123.17% | 93.18% | 67.57 | — |
| *chance* | *unrelated real clips* | | | | *86.14* | |

\* oracle-duration ablation. WER above 100% means the recogniser invents extra words.

### Finding 1 — the acoustic model really was the bottleneck

Hold the durations fixed and change only the acoustic model: **21.15% → 3.74%**, Wilcoxon
**p = 2.2×10⁻¹⁵**. That closes **83%** of the distance between phase 1's best system and the real
recording, and lands 1.3 points from the vocoder's own ceiling. MCD agrees independently:
30.14 dB against 43.86.

**Why this matters more than the number itself:** phase 1 did not guess this. It *measured* that
duration error cost nothing and templates cost 21 points, and its report concluded that effort
belonged in the acoustic model. Phase 2 changed that one component and the predicted effect
appeared. A prediction made in one phase and confirmed in the next is the strongest thing we can
say about the project.

### Finding 2 — only now does timing matter

This is the subtler half, and the best thing to talk about in the viva.

| | measured vs predicted durations | verdict |
| --- | --- | --- |
| **Phase 1** (templates) | 23.65% vs 21.15% | p = 0.048 / 0.100 — **no clear effect** |
| **Phase 2** (neural decoder) | 2.37% vs 3.74% | **p = 0.0014** — clear, correct direction |

Nothing about the durations changed between those two rows. The *acoustic model* changed. With
templates, everything was so degraded that duration error was invisible; with a competent decoder
the same duration error costs 1.37 points and is unambiguous (better on 21 of 100 clips, worse on
3).

Phase 1's own limitations section had written that "an acoustic model this coarse masks duration
effects entirely". It did. **A measurement is only as sharp as the instrument behind it.**

### Finding 3 — the diffusion model failed, and we know why

D2 trains (validation loss 0.173) and its output is statistically speech-like — standard
deviation 0.97 against real speech's 0.93, and 0% of values pinned at the sampler's clamp. But it
scores **123% WER**: unintelligible.

The diagnosis is that it learned *what spectrograms look like in general* without learning *how
to condition on the phoneme sequence*. It produces plausible texture with the wrong content.
That is exactly what a badly undertrained diffusion model does, and 9,000 steps is a very small
budget for diffusion — these models usually need hundreds of thousands.

**Along the way we did find a genuine bug, which is worth describing.** The first D2 audio was
pure clipping: RMS 0.86 where real speech is 0.067, 80% of samples pinned at full scale. The
DDIM sampler computes

```python
clean = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
```

and `alpha_bar[999] = 2.4e-09`, so that division **amplifies any error in the predicted noise by
20,291×** on the very first step. We measured the resulting mels at ±2787 where the data spans
±3.4. Clamping the predicted clean signal to the data range — the standard DDIM guard — fixed it
in one line, and RMS went from 0.86 to 0.0629.

**How we knew it was a sampler bug rather than an undertrained model:** the magnitude. A model
that needs more training degrades *gracefully*; it does not overshoot by a factor of a thousand.
Divergence that large points at arithmetic, not learning.

### Why MCD is unfair to D2, and WER is the fairer judge here

D1 is trained with L1 loss, which makes it predict the conditional **mean** spectrum — precisely
what minimises a distance-to-target metric like MCD. Diffusion deliberately **samples** variation
instead, so it scores worse on MCD even when it sounds good. D2's 67.57 dB therefore overstates
the gap slightly. Its 123% WER does not: that is a real failure.

---

## 6. What we are being honest about

**D1 is undertrained.** Train and validation loss were still falling together (0.510 / 0.504) at
9,000 steps, roughly six epochs, on a laptop GPU. The ten-clip overfit run reached 0.074, so the
architecture has far more capacity than we used. **3.74% is a lower bound**, not a converged
result.

**No CUDA was available.** Everything ran on Apple MPS. Length-bucketed batches of 16 exhausted
its 9 GB, so we halved the batch — note that bucketing reduces padding waste *and* concentrates
the longest clips in one batch, and those two effects fight each other.

**The neural duration predictor lost to phase 1.** 23.81 ms against 19.65 ms for the boosted
trees. We report it rather than quietly preferring the neural number.

**D2 did not work.** Reported as a failure with a diagnosis, not hidden.

**The system is not fully autonomous.** Predicted durations are still rescaled so the clip matches
its true total length — oracle information a deployed system would not have. The comparison
isolates the *relative* distribution of durations, not total-length prediction.

**The listening study is 5 team members, not blind naive listeners**, and it predates D1/D2 so it
covers only the phase 1 systems.

---

## 7. Questions you will be asked

**"What did you actually train?"**
Three duration models in phase 1 (per-phone mean, ridge, boosted trees), the phoneme templates,
and in phase 2 the encoder, duration predictor, decoder and diffusion model. We did **not** train
MFA, HiFi-GAN or Whisper — those are pre-trained, and the vocoder is deliberately frozen so it
cannot differ between systems.

**"Why is your diffusion model worse than your simple decoder?"**
Because it is drastically undertrained — 9,000 steps where diffusion typically needs hundreds of
thousands — and because sampling introduces variation that an L1-trained decoder does not have.
We can show it learned the right *statistics* (std 0.97 vs real 0.93) without learning the
*conditioning*, which is the expected failure mode.

**"Your title says latent diffusion. Where is it?"**
Phase 1 was explicitly scoped as the classical baseline; the README says the neural model comes
next. We diffuse in mel space, as Grad-TTS does. True latent diffusion (AudioLDM) needs a VAE
trained first, which is two models and two failure modes on laptops — we name it as the next step.

**"Isn't it cheating that you use the true clip length?"**
Partly, and we say so. Durations are rescaled to the real total length, which a deployed system
would not know. What the comparison isolates is the relative distribution of durations within a
clip. Predicting total length is listed as the first next step.

**"Why did duration matter in phase 2 but not phase 1?"**
Nothing about the durations changed — the acoustic model did. The templates degraded the audio so
much that duration error was invisible underneath. This is the project's most interesting result,
and phase 1's report predicted it.

**"How do you know the alignment you trained on is correct?"**
A test with a built-in control. Energy should change at speech/silence boundaries: it does for
94.6% of them. Shift every boundary five frames and rerun: 53.7%, which is chance. A broken
aligner scores the same both times.

**"Why 11.61 ms per frame?"**
256 samples of hop at 22,050 Hz. Every duration in the project counts these frames, and the
conversion lives in one file so nobody can round it differently.

**"What would you do with more compute?"**
Train D1 to convergence first — it is the cheapest available improvement and the current number
is a floor. Then give D2 a budget appropriate to diffusion. Then predict total utterance length
so the system is autonomous, and repeat the listening study blind now that there is a system
whose prosody is good enough for naturalness differences to be audible.
