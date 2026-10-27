# Phonetic Guided Speech Generation

Advanced Machine Learning course project. A small text-to-speech experiment on LJSpeech that compares
**explicit phoneme alignment** (an HMM aligner + a classical duration model) with a **learned acoustic generator**.

**Current stage:** AML baseline. Montreal Forced Aligner (MFA) → phoneme durations → classical duration
predictor → phoneme-template mel spectrograms → HiFi-GAN vocoder → speech, scored with duration MAE and Whisper WER.
The neural (DL) model comes next.

---

## Requirements

| What | Why | Who |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.11 and all packages into `.venv` | Everyone |
| `make` | One-command setup and pipeline shortcuts (built into macOS/Linux) | Everyone (Windows: use WSL) |
| ~15 GB free disk | Dataset, alignments, cached mels | Everyone |
| [Miniforge](https://github.com/conda-forge/miniforge) | MFA only installs through conda | Member 2 only |
| NVIDIA GPU | Optional, makes Whisper and the vocoder faster | Optional |

Install uv once:

```bash
brew install uv                                            # macOS (Homebrew)
curl -LsSf https://astral.sh/uv/install.sh | sh            # macOS / Linux without Homebrew
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

---

## Quick start (new team member)

```bash
# 1. Get the code
git clone https://github.com/<your-username>/phonetic-tts.git
cd phonetic-tts

# 2. Get the dataset archive (pick one)
#    a) copy LJSpeech-1.1.tar.bz2 from the team pen drive / Drive into data/downloads/
#    b) do nothing: step 3 downloads it automatically (2.6 GB)

# 3. Set everything up: Python 3.11 venv, packages, dataset check + extraction, import test
make setup
#    On an NVIDIA laptop, install the GPU build of PyTorch instead:
#    make setup TORCH_INDEX=<the --index-url shown by the selector on pytorch.org>

# 4. Activate the environment whenever you work
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

You are ready when `make setup` ends with:

```
wav files: 13100  (expected 13100)
torch 2.x.x | GPU: True/False
```

---

## Dataset

**LJ Speech 1.1**: <https://keithito.com/LJ-Speech-Dataset/>
Direct download: <https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2>

- 13,100 short clips of one female speaker reading non-fiction books (~24 hours)
- Mono, 16-bit WAV at 22,050 Hz, 1.1–10.1 seconds per clip
- `metadata.csv`: one line per clip, separated by `|`: `ID | raw transcript | normalised transcript`
- Licence: public domain

**Why the dataset is not in this repo:** the archive is 2.6 GB and GitHub rejects files over 100 MB.
The repo stores only `data/checksums.txt` (its SHA-256 fingerprint), so every copy can be verified.
**Never commit anything inside `data/` except `README.md`, `checksums.txt`, `sample/` and M2's
`cache/phones.csv` + `cache/excluded.csv`** (committed so nobody else has to install MFA).

After `make data`, the folder looks like this:

```
data/
├── downloads/LJSpeech-1.1.tar.bz2     # the archive (git-ignored)
└── raw/LJSpeech-1.1/
    ├── wavs/                          # 13,100 .wav files
    ├── metadata.csv
    └── README
```

### Data findings (verified 2026-10-27)

| Check | Result |
| --- | --- |
| Rows in `metadata.csv` / unique IDs | 13,100 / 13,100 |
| wav files vs IDs | 13,100 — 0 missing, 0 extra |
| Total audio | 23.92 h |
| Clip length | 1.11 s min · 6.76 s median · 10.10 s max |
| Mel frames per clip | 96 min · 870 max |
| Rows where raw ≠ normalised text | 1,505 (11.5%) — numbers and money spelled out, so we use column 3 |
| Digits left after cleaning | 0 |
| Non-ASCII left after cleaning | 0 |
| Punctuation left after cleaning | `, . - ' " ; : ? !` — MFA strips these itself |
| Source-text prefixes | 50, between 108 and 518 clips each |

Cleaning decisions these checks settled:

- **`St.` → `saint`.** All 16 occurrences are places (St. Albans, Bury St. Edmunds,
  St. Sepulchre, St. James's). Exactly one, `LJ007-0230` "Giltspur St. Compter", is really
  *Street*; one clip in 13,100 is not worth a special case.
- **`ft.` → `fort`** is correct: the only occurrence is "Dallas-Ft. Worth" (`LJ045-0055`).
- **`[square brackets]` appear in only 7 clips**, all Warren Commission transcriber notes
  (`[Marina]`, `[Oswald]`, `[the lunchroom door]`). These are almost certainly not spoken,
  so `REMOVE_SQUARE_BRACKET_CONTENT` should become `True` after a listening check.
- **`(round brackets)` appear in 151 clips** and are genuine spoken asides, so the words stay
  and only the bracket characters go.
- **`messrs.` appears in 17 clips** and is *not* expanded. That is correct: the MFA dictionary
  has `messrs` as `M EH1 S ER0 Z` ("messers"), which is how the reader says it.
- **366 other words are missing from the MFA dictionary** (names, compounds, acronyms such as
  "FBI"), in 1,179 clips. `configs/oov.dict` gives each a pronunciation before alignment, so
  all 13,100 clips align and none is excluded. Details in [TEAM.md](TEAM.md), M2 section.

### Splits

Test is whole source texts (`LJ005`), so no source text appears in both train and test.
With a single speaker, speaker-disjoint evaluation is impossible, so this is the honest
alternative.

| Split | Clips | Hours |
| --- | --- | --- |
| train | 12,501 | 22.83 |
| val | 300 | 0.54 |
| test (`LJ005`) | 299 | 0.55 |
| test_100 | 100 | 0.19 |

### Manual download and extraction (without make)

```bash
mkdir -p data/downloads data/raw
curl -L -o data/downloads/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
shasum -a 256 -c data/checksums.txt                 # must print: OK
tar -xjf data/downloads/LJSpeech-1.1.tar.bz2 -C data/raw
ls data/raw/LJSpeech-1.1/wavs | wc -l                # must print: 13100
```

Windows (Command Prompt): `curl` and `tar` work the same way. Check the file with
`certutil -hashfile data\downloads\LJSpeech-1.1.tar.bz2 SHA256` and compare the result with `data/checksums.txt` by eye.

---

## MFA setup (Member 2 only)

The Montreal Forced Aligner needs conda, so it lives in its own `aligner` environment, separate from `.venv`.

1. Install [Miniforge](https://github.com/conda-forge/miniforge) (macOS: `brew install --cask miniforge`) and open a new terminal.
   An existing Anaconda install works too.
2. Run:

```bash
make mfa-setup          # creates the "aligner" env and downloads the english_us_arpa models
conda run -n aligner mfa version
```

The `make align-*` targets call MFA through `conda run`, so you never need both environments active at once.
They align with `data/lexicon/english_us_arpa_lj.dict`: the MFA dictionary plus `configs/oov.dict`,
merged automatically by `make lexicon`. Re-run G2P for new OOVs with `python -m src.data.build_lexicon --refresh`.
MFA docs: <https://montreal-forced-aligner.readthedocs.io/>

---

## Running the pipeline

Run these in order from the repo root with `.venv` active. Each target works once its script exists.

| Step | Command | Owner | Produces |
| --- | --- | --- | --- |
| 1 | `make metadata` | M1 | `data/metadata_clean.csv`, `filelists/*.txt` |
| 2 | `make corpus` | M1 | `data/mfa_pilot/`, `data/mfa_core/`, `data/mfa_rest/` |
| 3 | `make align-pilot` / `align-core` / `align-rest` | M2 | `data/textgrids/` (`JOBS=8` uses 8 CPU cores; builds `data/lexicon/` first) |
| 4 | `make phones` | M2 | `data/cache/phones.csv`, `data/cache/excluded.csv` |
| 5 | `make mels` | M4 | `data/cache/mels/*.npy` |
| 6 | `make durations` | M3 | `outputs/durations/test_pred.csv`, `outputs/results/mae.csv` |
| 7 | `make templates` | M4 | `data/cache/templates.npz` |
| 8 | `make synth` | M4 | `outputs/audio/{R1,A1,A2}/` |
| 9 | `make wer` | M5 | `outputs/results/wer_*.csv` |
| 10 | `make results` | M5 | results table + figures in `outputs/` |
| — | `make test` | All | quick sanity tests |

Systems compared: **R0** real recording · **R1** real mel → HiFi-GAN · **A1** templates + true MFA durations ·
**A2** templates + predicted durations.

---

## Repository layout

```
configs/base.yaml     shared settings (sample rate 22,050, hop 256, 80 mels, paths)
configs/oov.dict      pronunciations for the 366 words MFA's dictionary lacks (M2)
filelists/            train / val / test / test_100 clip IDs (committed)
data/                 dataset, alignments, caches (git-ignored)
src/audio.py          load/save wav, wav → mel, mel → wav (HiFi-GAN)
src/text/             text cleaning, phonemes for demo text
src/data/             download, metadata, splits, MFA corpus, OOV lexicon, TextGrid → phones, mel extraction
src/baseline/         duration features + models, phoneme templates, baseline synthesis
src/eval/             duration MAE, Whisper WER, results table
notebooks/            data exploration, alignment checks, demo
tests/                sanity tests
outputs/              results and figures (audio git-ignored)
samples/              a few final wavs per system to listen to
```

---

## Team workflow

- Work on a branch named after your part (`data`, `alignment`, `durations`, `sound`, `eval`), merged by pull request.
- Every PR is reviewed by the next member: M1 → M2 → M3 → M4 → M5 → M1.
- All settings come from `configs/base.yaml`; never hard-code a number that lives there.
- Random seed is 42 everywhere.

| Member | Role |
| --- | --- |
| M1 | Data: download, cleaning, splits, MFA corpus |
| M2 | Alignment: MFA, phoneme durations |
| M3 | AML duration model: features, models, MAE |
| M4 | Sound: mels, vocoder, phoneme templates, baseline audio |
| M5 | Repo and evaluation: WER, results, demo |

**Who does what, in detail, and in what order: [TEAM.md](TEAM.md).**

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Makefile: missing separator` | Indented lines in the Makefile must start with a Tab, not spaces |
| `uv: command not found` | Install uv (see Requirements) and open a new terminal |
| `python3.11: command not found` | Not needed; `uv venv --python 3.11` downloads Python 3.11 itself |
| `shasum: ... FAILED` | The archive is incomplete or corrupted: delete it and copy or download again |
| SpeechBrain error mentioning `list_audio_backends` or `use_auth_token` | Only if the vocoder fails to load: `uv pip install torch==2.8.0 torchaudio==2.8.0 "huggingface_hub<1.0"`. Verified working without it on torch 2.14.1 / torchaudio 2.11.0 / speechbrain 1.1.2 |
| `GPU: False` on the NVIDIA laptop | Reinstall PyTorch with the CUDA command from pytorch.org |
| `make: command not found` (Windows) | Use WSL, or run the commands from the Manual download section and the steps in the Makefile by hand |

---

## References

- Dataset: K. Ito & L. Johnson, [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) (2017)
- McAuliffe et al., [Montreal Forced Aligner](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html), Interspeech 2017
- Kong et al., [HiFi-GAN](https://arxiv.org/abs/2010.05646), NeurIPS 2020 (vocoder: [speechbrain/tts-hifigan-ljspeech](https://huggingface.co/speechbrain/tts-hifigan-ljspeech))
- Ren et al., [FastSpeech 2](https://arxiv.org/abs/2006.04558), ICLR 2021
- Popov et al., [Grad-TTS](https://arxiv.org/abs/2105.06337), ICML 2021
- Radford et al., [Whisper](https://github.com/openai/whisper), used for WER evaluation