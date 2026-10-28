# Phonetic-Guided Speech Generation — Project Report

> **This is the plain-language companion document, not the submission.** The graded report is
> the IEEE-format paper in [`docs/report/`](report/) (`main.tex` → `main.pdf`). This file covers
> the same work in ordinary English and adds the viva preparation section, which does not belong
> in a conference paper. Every number in both comes from the same CSVs in `outputs/results/`.

**Course:** Advanced Machine Learning
**Dataset:** LJSpeech 1.1 (13,100 clips, 23.92 hours, one speaker)
**Code:** this repository · **Anchor paper:** Ren et al., *FastSpeech 2* (ICLR 2021)

---

## Contents

1. [What we built, in one paragraph](#1-what-we-built-in-one-paragraph)
2. [The problem statement](#2-the-problem-statement)
3. [The paper we took inspiration from](#3-the-paper-we-took-inspiration-from)
4. [The data](#4-the-data)
5. [What we did to the data, step by step](#5-what-we-did-to-the-data-step-by-step)
6. [The four systems — the experiment design](#6-the-four-systems--the-experiment-design)
7. [Results](#7-results)
8. [Conclusions](#8-conclusions)
9. [Scope, limitations, and things we are being honest about](#9-scope-limitations-and-things-we-are-being-honest-about)
10. [Who did what](#10-who-did-what)
11. [How to reproduce everything](#11-how-to-reproduce-everything)
12. [Likely questions, and how to answer them](#12-likely-questions-and-how-to-answer-them)
13. [References](#13-references)

### The result in one line

Predicting phone durations instead of measuring them cost us **nothing measurable** (21.15% vs
23.65% WER, p = 0.048 to 0.100). Replacing real spectrograms with averaged phoneme templates cost
**21 percentage points**. The acoustic model matters roughly thirty times more than duration
accuracy.

---

## 1. What we built, in one paragraph

We built a small text-to-speech system and used it to answer one specific question:
**how much does it matter that we know exactly how long each speech sound should last?**

To do this we take real recordings of a single speaker, use a tool called a *forced aligner* to
find out exactly where every individual sound (every "phone") starts and ends, and then rebuild
the audio from scratch in two ways — once using those true, measured durations, and once using
durations **predicted by a machine-learning model**. Comparing the two tells us how much of the
final speech quality depends on getting duration right, separately from everything else.

---

## 2. The problem statement

A text-to-speech system has to solve two different problems at once:

1. **How long does each sound last?** The word "cat" is three sounds — `K`, `AE`, `T` — but they
   are not equal in length. The vowel is usually much longer than the consonants, and the same
   sound stretches at the end of a sentence and shortens in the middle of a fast phrase.
2. **What does each sound actually sound like?** The frequency content that makes `S` hiss and
   `AE` ring.

Most modern systems learn both at the same time inside one big neural network, which makes it
impossible to tell which of the two is responsible when the output sounds wrong.

**Our problem statement:** *isolate the duration half.* Hold the "what it sounds like" part fixed,
and change only where the durations come from. Then measure the difference.

This gives us a clean, controlled comparison:

- **A1** — rebuild speech using the **true** durations measured from the real recording.
- **A2** — rebuild speech using **predicted** durations from our ML model.

Everything else about A1 and A2 is identical. So any difference between them is caused by
duration prediction error and nothing else.

### Why this is a machine-learning project

The machine-learning content is the duration model: a supervised regression problem with
**905,833 training examples** (one per phone occurrence in the training split), 21 engineered
features, three models compared, proper train/validation/test discipline, and a baseline that the
learned models have to beat. We report mean absolute error in milliseconds, plus the downstream
effect on speech intelligibility.

---

## 3. The paper we took inspiration from

**Ren, Hu, Tan, Qin, Zhao, Zhao, Liu — "FastSpeech 2: Fast and High-Quality End-to-End Text to
Speech", ICLR 2021.** <https://arxiv.org/abs/2006.04558>

FastSpeech 2 fits this project unusually well, and not by coincidence. The original FastSpeech
got its phone durations by distilling them out of a slower "teacher" model. FastSpeech 2's
central contribution was to **throw that away and use durations measured by the Montreal Forced
Aligner instead**, because the aligner's durations are more accurate. It then feeds those
durations into a component the authors call the **variance adaptor**, which predicts duration
(along with pitch and energy) at synthesis time.

So the exact thing FastSpeech 2 changed is the exact thing our experiment measures:

| | FastSpeech 2 | This project |
| --- | --- | --- |
| Where durations come from in training | Montreal Forced Aligner | Montreal Forced Aligner |
| Where durations come from at synthesis | learned duration predictor | our duration model (A2) |
| Upper bound if prediction were perfect | not isolated | **measured directly (A1)** |

FastSpeech 2 reports that MFA durations improve quality but never isolates *how much* of its
quality comes from the duration predictor specifically. **Our A1 vs A2 comparison is that
ablation**, run on a much smaller scale. That is the honest framing of our contribution: not a
new model, but a controlled measurement of an assumption the paper makes.

### Other work we rely on

- **McAuliffe, Socolof, Mihuc, Wagner, Sonderegger — Montreal Forced Aligner, Interspeech 2017.**
  The HMM/triphone forced aligner we use to get ground-truth phone boundaries.
- **Kong, Kim, Bae — HiFi-GAN, NeurIPS 2020.** The neural vocoder that turns a mel spectrogram
  back into a waveform. We use the pre-trained `speechbrain/tts-hifigan-ljspeech` checkpoint and
  do not train it.
- **Radford et al. — Whisper.** Used as an automatic listener to measure intelligibility.
- **Ito & Johnson — The LJ Speech Dataset, 2017.** Our data.
- **Popov et al. — Grad-TTS, ICML 2021** and **Liu et al. — AudioLDM, ICML 2023.** Diffusion-based
  speech generation; see the scope note in §9.

---

## 4. The data

**LJSpeech 1.1** — 13,100 short audio clips read by a single female speaker from seven
public-domain non-fiction books. 22,050 Hz, mono, 16-bit WAV. Total 23.92 hours.

We verified every claim about the data rather than trusting the documentation:

| Check | Result |
| --- | --- |
| Rows in `metadata.csv` / unique IDs | 13,100 / 13,100 |
| WAV files vs IDs | 13,100 — 0 missing, 0 extra |
| Total audio | 23.92 h |
| Clip length | 1.11 s min · 6.76 s median · 10.10 s max |
| Sample rate / channels | 22,050 Hz mono on every file |
| Rows where raw text ≠ normalised text | 1,505 (11.5%) |
| Digits remaining after our cleaning | 0 |
| Non-ASCII remaining after our cleaning | 0 |

**Why one speaker makes this easier and harder.** Easier: there is no voice variation to model, so
a small system can sound reasonable. Harder: we cannot do the standard "hold out unseen speakers"
evaluation, because there is only one speaker. Our honest alternative is described in §5.2.

---

## 5. What we did to the data, step by step

### 5.1 Text cleaning

The dataset gives each clip two transcripts: the raw text and a partly-normalised one. We needed
text containing only letters and spaces, because the forced aligner looks up each word in a
pronunciation dictionary and cannot look up "1864" or "£5".

What we did: lower-cased everything, spelled out numbers and money, expanded abbreviations,
stripped punctuation the aligner handles itself, and checked that **zero digits and zero
non-ASCII characters** survived.

Three decisions we settled by actually looking at the data rather than guessing:

- **`St.` → "saint".** All 16 occurrences are place names (St. Albans, Bury St. Edmunds). Exactly
  one — "Giltspur St. Compter" — really means *Street*. One clip in 13,100 is not worth a special
  case.
- **`ft.` → "fort"** is right: the only occurrence is "Dallas-Ft. Worth".
- **`messrs.` is deliberately left alone.** The aligner's dictionary already has `messrs` as
  `M EH1 S ER0 Z`, which is how the speaker actually says it.

### 5.2 Splitting the data

With one speaker we cannot hold out unseen voices, so we hold out **whole source texts** instead.
All 299 clips from book `LJ005` become the test set, so no sentence from a test book is ever seen
in training.

| Split | Clips | Hours | Used for |
| --- | --- | --- | --- |
| train | 12,501 | 22.83 | fitting the duration models and building templates |
| val | 300 | 0.54 | choosing hyper-parameters and picking the best model |
| test (`LJ005`) | 299 | 0.55 | the duration numbers we report |
| test_100 | 100 | 0.19 | the 100 clips we actually synthesise and listen to |

Everything uses random seed 42 so the split is identical on every machine.

### 5.3 Forced alignment — getting the ground truth

This is the step that makes the whole project possible. A **forced aligner** is given the audio
*and* the correct transcript, and works out where in time each individual sound occurs. We used
the Montreal Forced Aligner with its `english_us_arpa` acoustic model.

One practical obstacle: **366 words in LJSpeech are missing from MFA's dictionary** (names,
compounds, acronyms like "FBI"), affecting 1,179 clips. Without pronunciations, those clips would
be dropped. We generated a supplementary dictionary (`configs/oov.dict`) so that **all 13,100
clips aligned and none was excluded.**

Result: **949,856 phone segments across 70 distinct phone labels**, each with a start frame and a
length.

**How we know the alignment is actually correct.** This matters — if the alignment were wrong,
every number downstream would be meaningless. We ran a test with a built-in control: at each
boundary between speech and silence, the loudness should change in the expected direction.

| Test | Result |
| --- | --- |
| Boundaries where energy moves the right way | **94.6%** of 1,878 |
| Same test, boundaries deliberately shifted by 5 frames | **53.7%** |

The shifted version is the control: a broken aligner would score about the same either way.
Scoring 94.6% against 53.7% is strong evidence the boundaries sit on the real acoustic events.

As a second check, the measured durations are phonetically sensible: `DH` (the "th" in "the") is
shortest at 35 ms, diphthongs are longest, and **stressed vowels average 108 ms against 67 ms for
unstressed ones** — a well-known effect that we did not build in anywhere.

### 5.4 Turning audio into pictures (mel spectrograms)

Neural speech systems do not work on raw waveforms. They work on a **mel spectrogram**: a
picture where the horizontal axis is time, the vertical axis is frequency (80 bands spaced the way
human hearing works), and brightness is energy.

Our settings: 1,024-sample window, **256-sample hop**, 80 mel bands. The hop is what sets our time
resolution: **one frame = 11.61 milliseconds**. Every duration in this project is counted in these
frames.

One subtlety that caused real bugs: the vocoder we use was trained with a very specific mel
recipe (amplitude rather than power, "slaney" normalisation, natural log after clamping at
1e-5). We verified our mel function matches SpeechBrain's own to **MAE 0.000000** before trusting
it. A mismatch here produces audio that sounds plausible but is subtly wrong.

We cached all 13,100 mel spectrograms (1.1 GB) so nothing needs recomputing.

### 5.5 The duration model — the machine-learning core

**The task.** Given a phone and its context, predict how many frames it will last. This is
supervised regression: 905,833 training examples, one per phone occurrence.

**Why we predict the logarithm of the duration.** Durations are heavily skewed — most phones are
about 7 frames, but a silence can run to 481. If we minimised squared error on raw frame counts,
almost all of the model's effort would go into the handful of long pauses. So we regress
`log(n_frames)` and exponentiate at the end. **FastSpeech 2's duration predictor does exactly the
same thing**, which keeps us comparable to the paper.

**The 21 features.** All of them come from the *text* — none from the audio. That is deliberate
and it is the central methodological point of this section (see "the leak we avoided" below).

*Categorical (5):*

| Feature | Why |
| --- | --- |
| `phone` | the identity of the sound, keeping its stress digit (`AH0` vs `AH1`) |
| `prev_phone`, `next_phone` | sounds are shortened and lengthened by their neighbours |
| `cls` | silence / vowel / fricative / stop / nasal / other |
| `stress` | stressed vowels are measurably longer (108 ms vs 67 ms) |

*Numeric (16):* position within the word (`idx_in_word`, `word_len`, and the ratio), whether the
phone starts or ends a word, position of the word in the sentence, position of the phone in the
clip, how many phones and words the clip has, whether this phone or its neighbours are silence,
and — the interesting pair — **how many phones away the nearest silence is on each side**.

That last pair captures **pre-pausal lengthening**: the sound just before a pause gets stretched.
It is a real, documented property of human speech, and it shows up clearly in our data. In the
very first clip of the dataset, the `NG` immediately before the pause lasts 18 frames against a
corpus average of 7.4.

**The leak we avoided, and why it cost us nothing.** The tempting feature is the clip's total
length divided by its number of phones — an excellent tempo estimate. It is also cheating: at
synthesis time you do not know how long the finished audio will be. We excluded every feature
derived from the true total length, and we enforce it with a test that multiplies every duration
by three and asserts the feature table comes back byte-identical.

There is a second, subtler reason this costs nothing. The synthesis step rescales all predicted
durations so the clip lands on its true total length. A clip-level tempo multiplier would
therefore be divided straight back out. **The model only ever needs to get durations right
relative to each other within a clip** — and recognising that is what makes excluding the feature
free rather than a sacrifice.

**Three models, compared properly.** Trained on `train`, selected on `val`, reported once on
`test`:

1. **Per-phone mean** — just the average length of each phone type. The baseline the learned
   models must beat. (We average in log space, so this is a geometric mean.)
2. **Ridge regression** — one-hot encoded categoricals plus standardised numerics. The
   regularisation strength `alpha` was chosen on `val` from {0.1, 1, 10, 100}.
3. **Gradient-boosted trees** — `HistGradientBoostingRegressor`, which splits on categories
   natively instead of needing one-hot expansion. Training stops when validation error stops
   improving; it converges at **1,052 iterations**.

Results are in §7.

### 5.6 Rebuilding the audio

To isolate duration, the "what it sounds like" half has to be as simple and as *fixed* as
possible. So we use **phoneme templates** rather than a learned acoustic model:

For each of the 70 phones, take every occurrence in the training set, cut it into three equal
parts (beginning / middle / end), and average each part across all occurrences. That gives one
fixed `(80 frequency bands × 3 states)` template per phone — a kind of "average `S`", "average
`AH1`", and so on.

To synthesise a clip: look up each phone's template, stretch its three states to fill that
phone's duration, lay them end to end to make a full mel spectrogram, and pass that to the
**pre-trained HiFi-GAN vocoder** to get a waveform.

Because the templates are identical in A1 and A2 and the vocoder is frozen, **the only thing that
differs between them is the durations.** That is what makes the comparison clean.

---

## 6. The four systems — the experiment design

| System | What it is | What it tells us |
| --- | --- | --- |
| **R0** | the real recording, untouched | the floor — how well our measuring instrument works at all |
| **R1** | real mel spectrogram → vocoder | the ceiling for anything using this vocoder |
| **A1** | templates + **true** MFA durations | how good templates can get with perfect durations |
| **A2** | templates + **predicted** durations | the complete, honest pipeline |

Reading the chain is the point of the whole project. Each step down tells you the cost of one
specific thing:

- **R0 → R1** is the cost of the vocoder.
- **R1 → A1** is the cost of replacing real spectrograms with crude averaged templates.
- **A1 → A2** is the cost of *predicting* durations instead of measuring them. **This is the
  number the project exists to produce.**

**How we measure quality.** We do not run human listening tests — we have no panel and no time.
Instead we use **Whisper** (OpenAI's speech recogniser) as a stand-in listener: feed it our
synthesised audio, and compare its transcript against the true text. If the audio is intelligible,
Whisper transcribes it correctly.

We report **Word Error Rate (WER)** and **Character Error Rate (CER)**. Both reference and
hypothesis go through Whisper's `EnglishTextNormalizer` first, so we measure whether the *speech*
was understood rather than punctuation or spelling differences.

**What WER does and does not measure:** it measures *intelligibility*, not *naturalness*. Audio
can be perfectly intelligible and still sound robotic. Given that our acoustic model is literally
an average spectrum per phone, we expect exactly that — intelligible but obviously synthetic.
This is a stated limitation, not a surprise.

---

## 7. Results

### 7.1 The duration model

Trained on 12,501 clips, hyper-parameters chosen on 300 validation clips, reported once on the
299 test clips from the held-out book. One frame is 11.61 ms.

| Model | MAE (frames) | MAE (ms) | MAE excluding silence (ms) | RMSE (frames) |
| --- | --- | --- | --- | --- |
| Per-phone mean (baseline) | 2.637 | 30.61 | 26.88 | 4.34 |
| Ridge regression | 2.338 | 27.14 | 23.31 | 3.93 |
| **Gradient-boosted trees** | **1.693** | **19.65** | **16.84** | **3.08** |

**Boosted trees reduce duration error by 35.8% relative to the baseline.** The ordering —
baseline worse than ridge worse than trees — is the reassuring one: if the linear model had beaten
the trees we would suspect a leak or a bug. Validation and test numbers agree closely (1.702 vs
1.693 frames for the chosen model), so we are not over-fitting the selection.

Boosting stopped on its own at 1,052 iterations, so the result is a converged model rather than
one truncated by a budget.

### 7.2 Intelligibility of the synthesised speech

100 clips per system, transcribed by Whisper `small.en`, scored with `jiwer` after normalising
both reference and hypothesis.

| System | What it is | WER | CER |
| --- | --- | --- | --- |
| **R0** | the real recording | **1.85%** | 0.71% |
| **R1** | real spectrogram → vocoder | **2.48%** | 0.95% |
| **A1** | templates + **true** durations | **23.65%** | 11.83% |
| **A2** | templates + **predicted** durations | **21.15%** | 10.02% |

### 7.3 Reading the chain — where the quality actually goes

| Step | Cost in WER | What it is the price of |
| --- | --- | --- |
| R0 → R1 | **+0.63 pp** | the vocoder |
| R1 → A1 | **+21.17 pp** | replacing real spectrograms with averaged templates |
| A1 → A2 | **−2.50 pp** | predicting durations instead of measuring them |

Two things stand out, and the second one is the main result of the project.

**The vocoder is almost free.** Going from a real recording to a real spectrogram and back through
HiFi-GAN costs 0.63 percentage points. The spectrogram representation and the vocoder are not
where quality is lost. (R0's own 1.85% is not zero partly because of text-normalisation artifacts:
three of its ten worst clips are cases where the reference reads "george the fourth" and Whisper
writes "george iv" — same meaning, counted as errors.)

**The acoustic model dominates everything.** Replacing real spectrograms with one averaged
spectrum per phone costs **21.17 percentage points** — thirty-four times the vocoder's cost. This
is where essentially all the degradation lives.

### 7.4 The surprise: predicted durations did not hurt

We expected A2 to be worse than A1, because A1 uses durations measured from the real recording
while A2 has to predict them. **It was not.** A2 came out 2.50 points *better*.

We tested whether this is real rather than noise, on the same 100 clips:

| Test | Result |
| --- | --- |
| Clips where A2 beat A1 | 40 |
| Clips where A1 beat A2 | 30 |
| Clips tied | 30 |
| Wilcoxon signed-rank | p = 0.048 |
| Paired t-test | p = 0.100 |

**The honest reading is that this difference is marginal.** One test clears the conventional 0.05
threshold and the other does not, and A2 wins on only 40 of 100 clips. We should not claim that
predicted durations are *better* than real ones.

What we can claim, and what the evidence supports, is the more interesting statement:

> **At this level of acoustic quality, duration prediction error has no measurable harmful effect
> on intelligibility.** Our model's 19.65 ms average error is simply not the bottleneck — the
> averaged templates are, by a factor of roughly thirty.

**Why might A2 even be slightly ahead?** We investigated and can only offer a partial answer.
Predicted durations contain far fewer pathologically short phones (**0.8% under 3 frames, versus
3.5% for the true durations**), and a phone of one frame cannot be rendered by a three-state
template at all. Across clips, the more short phones A2 removed, the more it improved
(r = +0.227, p = 0.023). But the direct correlations are null — A1's WER does not track its own
short-phone fraction (r = +0.014, p = 0.89) — so this is a **hypothesis we could not confirm**,
not a demonstrated mechanism.

We also found an artifact working *against* A2 that we would fix next: the routine that rescales
predicted durations to the clip's true length dumps any leftover frames onto the single longest
phone, which produced phones of up to 154 frames (1.8 seconds) in 34 of the 100 clips, against a
true maximum of 76. Spreading that residual proportionally would be the obvious improvement.

---

## 8. Conclusions

1. **Forced alignment gives timing we can trust.** 94.6% of phone boundaries move energy in the
   expected direction, against 53.7% for a deliberately misaligned control, and all 13,100 clips
   aligned with none excluded. The measured durations also reproduce known phonetics we did not
   build in — stressed vowels at 108 ms against 67 ms unstressed.

2. **Context features clearly beat a per-phone average.** Gradient-boosted trees cut duration
   error by 35.8% relative to the baseline, from 30.61 ms to 19.65 ms. Phone identity with stress,
   the neighbouring phones, and distance to the nearest pause are what carry that improvement.

3. **Duration accuracy was not the bottleneck — and this is the finding that surprised us.** The
   experiment was designed to measure the cost of predicting durations rather than measuring them.
   That cost turned out to be statistically indistinguishable from zero (and nominally negative),
   while the crude acoustic model cost 21 percentage points. **We set out to measure one thing and
   learned that the other thing matters roughly thirty times more.**

4. **What this says about FastSpeech 2.** The paper's decision to take durations from a forced
   aligner is well founded — the alignment really is accurate. But our ablation suggests that at
   low acoustic quality, improving the duration predictor has little effect on intelligibility.
   The variance adaptor's value is more likely in *naturalness*, which our evaluation cannot see.
   Testing that would need a human listening study.

5. **Where the next effort should go.** Not into better duration modelling. Into the acoustic
   model — which is precisely where a diffusion-based generator would sit, and where the alignment
   and duration machinery we built is the input it would need.
## 9. Scope, limitations, and things we are being honest about

**The project title mentions latent diffusion; there is no diffusion model here.** We should say
this plainly rather than let it be discovered. What we built is phoneme templates plus a
pre-trained HiFi-GAN vocoder. A diffusion-based acoustic model (Grad-TTS operates in mel space;
AudioLDM is the genuinely *latent* one) would be the natural next step, and the alignment and
duration machinery we built is exactly what such a model would need as input. But it is future
work, not something we did.

**A2 is given the clip's true total length.** Our synthesis rescales predicted durations so the
clip ends up exactly as long as the real recording. That is oracle information a deployed
text-to-speech system would not have — it would have to predict total length as well. So what A1
vs A2 isolates is the *relative* distribution of durations inside a clip, not total-length
prediction. We made this choice so both systems hand the vocoder spectrograms of identical length,
which keeps the comparison clean, but it does mean **A2 is not a fully autonomous system** and we
should not claim it is.

**Our acoustic model is deliberately crude.** One averaged spectrum per phone throws away all
coarticulation — the way neighbouring sounds blend into each other — and all prosody beyond
duration. This is a *feature* of the experimental design, because it holds the acoustic half
constant so duration can be isolated, but it does cap how good A1 and A2 can ever sound.

**WER measures intelligibility, not naturalness.** Audio can be perfectly transcribable and still
sound robotic. We have no human listening test, so we cannot make any claim about naturalness.

**Whisper has its own error floor.** Three of R0's ten worst clips are not synthesis failures at
all — they are text-normalisation artifacts, where the reference says "george the fourth" and
Whisper says "george iv". Identical meaning, counted as errors. So a small non-zero WER is
unavoidable even on perfect audio.

**One speaker only.** We cannot test whether any of this generalises to other voices. The
whole-book holdout is the best available substitute, but it is not speaker-disjoint evaluation.

**The duration model never sees audio.** It predicts from text features alone. That is correct
for a synthesis system, but it means it cannot know that *this particular* reading was unusually
fast.

---

## 10. Who did what

Five members, five parts, developed in parallel against two shared contracts: `configs/base.yaml`
for every number, and fixed file formats for every hand-off. Nobody could change either without
telling the team, which is what let five people work on one pipeline at once.

| Member | Part | What they produced |
| --- | --- | --- |
| **Kunal Vats** | Data, and the duration model with Pushkar | Text cleaning and normalisation, the whole-book splits, `configs/base.yaml` and `src/config.py` (the shared settings every other part imports), the MFA corpus folders |
| **Pushkar Jain** | Duration model, with Kunal | The 21-feature builder, the three models, model selection on validation, the MAE harness and the two result tables |
| **Tanishk Agrawal** | Alignment | Montreal Forced Aligner setup, the 366-word supplementary pronunciation dictionary, TextGrid → `phones.csv` conversion, and the alignment validation notebook |
| **Shaurya Sharma** | Sound | `src/audio.py` (wav ↔ mel, HiFi-GAN), mel caching for all 13,100 clips, the 70 phoneme templates, and synthesis of R1 / A1 / A2 |
| **Chaitanya Shekhawat** | Evaluation | The Whisper WER/CER harness with proper text normalisation on both sides, the results table, and the failure log |

### How the parts depended on each other

```
Kunal (data) ──> Tanishk (alignment) ──> Kunal + Pushkar (durations) ──┐
      │                   │                                           ├──> Chaitanya (evaluation)
      └──> Shaurya (sound)┴───────────────────────────────────────────-┘
```

Alignment was the critical path: nothing could be modelled until `phones.csv` existed. The
duration model was written and unit-tested against a hand-made fake `phones.csv` so that work did
not have to wait for it.

### Engineering practices worth mentioning

- **170 automated tests**, including tests that guard the shared contracts — the frame-count
  formula, the mel recipe matching the vocoder, and the no-leakage property of the features.
- **Three real bugs were caught by running the pipeline end to end**, not by reading code: a
  residual-distribution bug that broke 27 of 299 clips, a numpy truth-value crash that stopped
  synthesis entirely, and a NaN that had silently corrupted 61 of 70 phoneme templates. Each is
  now covered by a regression test. These are worth describing in the viva — finding them is the
  difference between code that compiles and a pipeline that works.

---

## 11. How to reproduce everything

```bash
make setup                       # Python 3.11 venv, packages, dataset download + checks
make metadata                    # clean text, build the splits
make mels                        # cache 13,100 mel spectrograms (about 1 minute)
make durations                   # train the duration models, write predictions + MAE
make templates                   # build the 70 phoneme templates
make synth                       # synthesise R1, A1, A2 for 100 clips
make wer results                 # Whisper transcription + the results table
make test                        # the full test suite
```

`data/cache/phones.csv` (the alignment output) is committed, so **nobody needs to install the
Montreal Forced Aligner to reproduce our results.** To regenerate it from scratch instead:
`make mfa-setup lexicon align-core align-rest phones` (about 15 minutes on 8 cores, needs conda).

Random seed is 42 throughout, and all shared settings live in `configs/base.yaml` so no number is
duplicated between members' code.

---


---

## 12. Likely questions, and how to answer them

### About the design

**"What exactly is the difference between A1 and A2?"**
Only one thing: where the phone durations come from. A1 uses the durations measured from the real
recording by the forced aligner. A2 uses durations predicted by our model from text alone. The
phoneme templates, the vocoder, and the clip's total length are identical. So the quality gap
between them is caused by duration prediction error and nothing else.

**"Isn't it cheating that A2 knows the clip's true total length?"**
Yes, partly, and we should own this. Our synthesis step rescales the predicted durations so the
clip ends up exactly as long as the real recording. That is oracle information a real
text-to-speech system would not have — it would have to predict total length too. What our A1 vs
A2 comparison isolates is therefore the *relative* distribution of durations within a clip, not
total-length prediction. We chose this because it makes the comparison clean: both systems produce
spectrograms of identical length, so the vocoder sees the same conditions. A fully autonomous
version is the obvious next experiment.

**"Why hold out whole books instead of a random split?"**
Because a random split would put sentences from the same book, recorded in the same session, on
both sides. Holding out all 299 clips of book `LJ005` means no test sentence shares its source
text with anything in training. With a single speaker we cannot do speaker-disjoint evaluation,
so this is the strongest honest alternative.

**"Why Word Error Rate and not a human listening test (MOS)?"**
We have no listening panel. Whisper gives us a reproducible, automatic stand-in. But be clear
about what it measures: **intelligibility, not naturalness.** Our audio can be understandable and
still sound robotic, and WER will not detect that.

### About the machine learning

**"Why is your baseline just the per-phone average?"**
Because it is the strongest *trivial* predictor, so beating it proves the context features earn
their place. A model that cannot beat "every `S` is 7 frames long" has learned nothing. Ours beats
it by a large margin, which is the result.

**"Why predict the logarithm of duration?"**
Durations are heavily right-skewed: most phones are around 7 frames, silences reach 481. With
squared error on raw frame counts, the model would spend almost all its capacity on a few long
pauses and get the ordinary sounds wrong. Working in log space makes the error relative instead of
absolute. FastSpeech 2's duration predictor does the same.

**"How do you know your features don't leak the answer?"**
Two ways. First, by construction: every feature comes from the text — the phone, its neighbours,
stress, position, distance to the nearest pause — and none from the audio or the clip's true
length. Second, by test: `test_no_feature_is_built_from_the_clips_true_length` multiplies every
duration by three and asserts the feature table comes back byte-identical. If anyone later adds a
leaking feature, that test fails.

**"Why gradient-boosted trees rather than a neural network?"**
The data is tabular with strong categorical features, which is exactly where boosted trees are
hard to beat; they train in about a minute on a laptop with no GPU and no tuning ritual. A neural
duration predictor is what FastSpeech 2 uses and would be the natural comparison, but it would
not have changed the conclusion of *this* experiment, which is about duration accuracy rather than
model architecture.

**"Why keep the stress digit, giving 70 phone labels instead of 39?"**
Because stress is one of the strongest duration signals in the data: stressed vowels average
108 ms against 67 ms for unstressed. Collapsing `AH0` and `AH1` into `AH` would throw that away.
We measured it before deciding.

### About the audio

**"What is a mel spectrogram, and why 80 bands?"**
It is a picture of the sound: time across, frequency up, brightness as energy — with the frequency
axis warped to match how human hearing resolves pitch. 80 bands is what the pre-trained HiFi-GAN
vocoder expects; we did not choose it freely, we matched it.

**"What is a vocoder and why didn't you train one?"**
A spectrogram throws away phase information, so you cannot simply invert it back to a waveform. A
vocoder is a model that reconstructs a plausible waveform from the spectrogram. Training one needs
far more compute than we have, and it would also *change between our systems* — which would ruin
the experiment. We use a frozen pre-trained HiFi-GAN precisely so the acoustic back end is
identical everywhere.

**"How do you know your spectrograms match what the vocoder expects?"**
We checked numerically rather than assuming. Our mel function and SpeechBrain's own agree to
**mean absolute error 0.000000**. This matters because a mismatch produces audio that sounds
roughly right but is subtly degraded — the kind of bug you never find by listening.

**"How do you know the forced alignment is correct?"**
A test with a built-in control. At each speech/silence boundary, loudness should change in the
expected direction; it does for **94.6%** of boundaries. Then we shifted every boundary by five
frames and re-ran it: **53.7%**, which is roughly chance. A broken aligner would score the same
both times.

**"One frame is 11.61 ms — where does that come from?"**
From the 256-sample hop at 22,050 Hz: 256 / 22050 = 0.01161 s. Every duration in the project is
counted in these frames, and the conversion lives in exactly one place (`src/config.py`) so no two
members can round it differently.

### If they push on weaknesses

**"Your synthetic speech is much worse than the real recording. Isn't that a failure?"**
No — it is the expected consequence of a deliberately simple acoustic model, and the project was
never trying to win on audio quality. One averaged spectrum per phone discards coarticulation and
all prosody except duration. The experiment is the *comparison* between A1 and A2, and that
comparison is only meaningful because the acoustic model is held fixed and simple.

**"What would you do with more time?"**
Three things, in order: predict total clip length as well so A2 becomes fully autonomous; replace
the averaged templates with a learned acoustic model (this is where a diffusion model would go);
and run a small human listening test, because WER cannot see naturalness.
## 13. References

- Ito, K. & Johnson, L. *The LJ Speech Dataset*, 2017. <https://keithito.com/LJ-Speech-Dataset/>
- McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., Sonderegger, M. *Montreal Forced Aligner:
  Trainable text-speech alignment using Kaldi.* Interspeech 2017.
- Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., Liu, T.-Y. *FastSpeech 2: Fast and
  High-Quality End-to-End Text to Speech.* ICLR 2021. <https://arxiv.org/abs/2006.04558>
- Kong, J., Kim, J., Bae, J. *HiFi-GAN: Generative Adversarial Networks for Efficient and High
  Fidelity Speech Synthesis.* NeurIPS 2020. <https://arxiv.org/abs/2010.05646>
- Popov, V., Vovk, I., Gogoryan, V., Sadekova, T., Kudinov, M. *Grad-TTS: A Diffusion
  Probabilistic Model for Text-to-Speech.* ICML 2021. <https://arxiv.org/abs/2105.06337>
- Liu, H. et al. *AudioLDM: Text-to-Audio Generation with Latent Diffusion Models.* ICML 2023.
- Radford, A. et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
  ICML 2023.
