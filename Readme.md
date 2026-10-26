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
**Never commit anything inside `data/` except `README.md`, `checksums.txt` and `sample/`.**

After `make data`, the folder looks like this:

```
data/
├── downloads/LJSpeech-1.1.tar.bz2     # the archive (git-ignored)
└── raw/LJSpeech-1.1/
    ├── wavs/                          # 13,100 .wav files
    ├── metadata.csv
    └── README
```

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
2. Run:

```bash
make mfa-setup          # creates the "aligner" env and downloads the english_us_arpa models
conda run -n aligner mfa version
```

The `make align-*` targets call MFA through `conda run`, so you never need both environments active at once.
MFA docs: <https://montreal-forced-aligner.readthedocs.io/>

---

## Running the pipeline

Run these in order from the repo root with `.venv` active. Each target works once its script exists.

| Step | Command | Owner | Produces |
| --- | --- | --- | --- |
| 1 | `make metadata` | M1 | `data/metadata_clean.csv`, `filelists/*.txt` |
| 2 | `make corpus` | M1 | `data/mfa_pilot/`, `data/mfa_core/`, `data/mfa_rest/` |
| 3 | `make align-pilot` / `align-core` / `align-rest` | M2 | `data/textgrids/` (`JOBS=8` uses 8 CPU cores) |
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
filelists/            train / val / test / test_100 clip IDs (committed)
data/                 dataset, alignments, caches (git-ignored)
src/audio.py          load/save wav, wav → mel, mel → wav (HiFi-GAN)
src/text/             text cleaning, phonemes for demo text
src/data/             download, metadata, splits, MFA corpus, TextGrid → phones, mel extraction
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

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Makefile: missing separator` | Indented lines in the Makefile must start with a Tab, not spaces |
| `uv: command not found` | Install uv (see Requirements) and open a new terminal |
| `python3.11: command not found` | Not needed; `uv venv --python 3.11` downloads Python 3.11 itself |
| `shasum: ... FAILED` | The archive is incomplete or corrupted: delete it and copy or download again |
| SpeechBrain error mentioning `list_audio_backends` or `use_auth_token` | `uv pip install torch==2.8.0 torchaudio==2.8.0 "huggingface_hub<1.0"` |
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