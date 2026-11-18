# Listen to the project

Everything here was produced by the code in this repository. Play them in the order below —
it is the story of the project in about three minutes.

---

## 1. `spoken/` — sentences nobody ever recorded

**Start here. This is the thing worth showing.**

These are not clips from the dataset rebuilt. They are sentences we typed, converted to phonemes,
given durations by the model, drawn as spectrograms by the model, and turned into sound. **No
recording of these sentences exists anywhere.**

| File | What we typed | What Whisper heard back |
| --- | --- | --- |
| `01_intro.wav` | Hello, this is our advanced machine learning project. | Hello, this is our Advanced Machine Learning project. |
| `02_topic.wav` | We generate speech from text using phoneme durations and a neural network. | *(exact)* |
| `03_numbers.wav` | We trained on 13100 clips, about 24 hours of audio. | We trained on 13,100 clips, about 24 hours of audio. |
| `04_pauses.wav` | First we align the audio, then we predict durations, and finally we draw the spectrogram. | *(exact)* |
| `05_question.wav` | Can a machine still understand what this model is saying? | *(exact)* |

**All five come back word-perfect**, including `13100` being spoken as "thirteen thousand one
hundred".

Two things to listen for:

- **`04_pauses.wav`** has three clauses. The pauses are not inserted by hand — punctuation becomes
  a silence phone, and the model decides how long each one lasts.
- **These are the only fully autonomous outputs in the project.** Everywhere else, predicted
  durations are rescaled so the clip matches the real recording's length, which is information a
  deployed system would not have. Here there is no recording, so the model's own durations decide
  the length of the audio. Nothing was corrected.

Make your own, or run it live in front of someone:

```bash
python demo.py                      # speaks, then prompts: type a sentence, hear it
python demo.py "Say this instead."  # one-shot
```

`demo.py` loads the model once (a few seconds) and then generates each sentence in about one
second, playing it automatically. The `SENTENCES` list at the top of the file is what it opens
with — edit that before a presentation.

---

## 2. `comparison/` — the same sentence through every system

This is the experiment. Each system changes **one** thing from the one above it, so the
difference you hear is caused by that one change.

| System | Acoustic model | Durations |
| --- | --- | --- |
| `R0` | — the real recording | — |
| `R1` | the real spectrogram | real |
| `A1` | 70 averaged templates (phase 1) | measured by the aligner |
| `A2` | 70 averaged templates (phase 1) | predicted |
| `D1` | **neural decoder (phase 2)** | predicted |
| `D2` | mel diffusion — **failed** | predicted |

### `good_LJ005-0066_*` — a 10-second sentence D1 gets perfectly right

> *"the new jails," as howard says, "having pompous fronts, appear like palaces to the lower class"*

Play `R0` → `A1` → `D1` back to back. R0 is the real reading. A1 is phase 1: intelligible if you
concentrate, but flat and buzzy, because every instance of a sound is literally the same averaged
picture. D1 is phase 2 — **0% word error rate on this clip.**

### `hard_LJ005-0213_*` — the honest one

> *"newgate through all these years continued a bye-word with the society."*

We included a failure case deliberately. But listen to `R0` first, because the interesting part is
that **Whisper gets 25% wrong on the real human recording**:

| System | WER | What Whisper heard |
| --- | --- | --- |
| R0 *(real!)* | 25.0% | "…continue to buy word with the…" |
| R1 | 16.7% | "…continued a byword…" |
| **D1** | **33.3%** | "nougat through all these years continue to by word…" |
| A1 | 58.3% | "a new date from all these years…" |
| D2 | 116.7% | "we need to keep it well but now we are going to do something" |

"Newgate" and "bye-word" are archaic, and the recogniser mishears them **in the original audio**.
D1 is only 8 points worse than a real human recording here. Most of this clip's difficulty is the
measuring instrument's, not the model's — which is exactly why we validated Whisper on real
recordings before trusting any number.

`D2` is worth playing once so the failure is concrete: it produces confident, speech-shaped sound
with entirely invented content. It learned what speech *looks like* without learning to follow the
phonemes it was given.

---

## The numbers behind what you are hearing

100 clips per system, from a book held out of training entirely.

| System | WER | CER | MCD dB |
| --- | --- | --- | --- |
| R0 real recording | 1.85% | 0.71% | — |
| D1 + measured durations | 2.37% | 0.63% | — |
| R1 real spectrogram → vocoder | 2.48% | 0.95% | — |
| **D1 neural decoder** | **3.74%** | **1.05%** | **30.14** |
| A2 templates + predicted | 21.15% | 10.02% | 63.04 |
| A1 templates + measured | 23.65% | 11.83% | 43.86 |
| D2 diffusion | 123.17% | 93.18% | 67.57 |
| *chance (unrelated clips)* | | | *86.14* |

**Replacing the averaged templates with a learned decoder took word error rate from 21.15% to
3.74%** (p = 2×10⁻¹⁵) — within 1.3 points of what the vocoder alone can achieve.

One caveat we state rather than hide: **D1 is undertrained.** Training and validation loss were
still falling when the budget ran out on a laptop GPU, so 3.74% is a floor, not a ceiling.

Full write-up: [`docs/report/main.pdf`](../docs/report/main.pdf) ·
plain-English version: [`docs/DEEP_LEARNING_EXPLAINED.md`](../docs/DEEP_LEARNING_EXPLAINED.md)
