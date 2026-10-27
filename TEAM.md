# Work division — AML baseline (2 days)

Five parts that run in parallel. **Everything hangs off two contracts:**
`configs/base.yaml` (numbers) and the shared file formats below (data).
Never change either without telling the team in the group chat.

Rules for everyone:
- Run scripts from the repo root: `python -m src.data.split`, never `python src/data/split.py`.
- Import settings, never retype them: `from src.config import SAMPLE_RATE, HOP_LENGTH, MEL_KWARGS`.
- Use `n_frames()` and `sec_to_frame()` from `src/config.py`. Do not re-derive them —
  if two members round differently, durations and mels silently disagree.
- Seed is 42, from `src.config.SEED`.
- Read audio with `soundfile`/`librosa`, **never** `torchaudio.load`.
- Branch per part (`data`, `alignment`, `durations`, `sound`, `eval`), merge by PR.

---

## Dependency order — who is blocked by whom

```
M1 data  ──> M2 alignment ──> M3 durations ──┐
   │                 │                       ├──> M5 evaluation
   └──> M4 sound ────┴───────────────────────┘
```

- **M2 is the critical path.** MFA on ~4,600 clips is the longest single wait, so M2 starts
  the moment the pilot exists (it already does — see below).
- **M4 does not need M2** for `src/audio.py` or cached mels. Start there immediately.
- **M3 cannot start modelling** until `phones.csv` exists, but *can* write and unit-test the
  feature builder against a hand-made fake `phones.csv` today.
- **M5 does not need anything** to set up the repo, Whisper harness and results scaffolding.
  Test the WER harness on **R0** (the real recordings) — that works on day one and gives the
  floor every other system is compared against.

---

## M1 — Data  ✅ DONE, merged

Owns: `src/text/normalize.py`, `src/data/prepare_metadata.py`, `src/data/split.py`,
`src/data/prepare_mfa_corpus.py`, `configs/base.yaml`, `src/config.py`, `src/utils.py`.

Produced and committed:
- `data/metadata_clean.csv` — 13,100 rows, `id, text_raw, text_clean, n_samples, n_frames, seconds`
- `filelists/train.txt` (12,501) · `val.txt` (300) · `test.txt` (299) · `test_100.txt` (100)
- `data/mfa_pilot/LJ/` — 200 clips, wav + lab, **ready for M2 now**

Regenerate on any machine with:
```bash
make metadata          # prepare_metadata + split
make corpus            # all three MFA folders  (or: python -m src.data.prepare_mfa_corpus pilot)
```

Still open for M1 (small, not blocking):
- Listen to 2 of the 7 `[square bracket]` clips, then set `REMOVE_SQUARE_BRACKET_CONTENT`
  in `src/text/normalize.py`. All 7 look like unspoken transcriber notes, so this will
  probably become `True`. Re-run `make metadata corpus` after flipping it.
- Report figures: duration histogram, clips per prefix, speaking-rate outliers → `outputs/figures/`.

---

## M2 — Alignment  ✅ DONE, on branch `alignment`

Owns: `src/data/textgrid_to_phones.py`, `src/data/build_lexicon.py`, `configs/oov.dict`,
`notebooks/02_alignment.ipynb`.

Produced (all 13,100 clips aligned, **0 excluded**):
- `data/cache/phones.csv` — 949,856 phones, 34 MB. Every split complete:
  train 12,501 · val 300 · test 299 · **test_100 100/100**, so A1/A2 exist for every clip.
- `data/cache/excluded.csv` — header only; the code path is there if a re-run produces failures.
- `data/textgrids/{pilot,core,rest}/` — MFA output, plus MFA's `alignment_analysis.csv` per corpus.
- `outputs/figures/alignment_examples.png`, `alignment_energy_by_class.png`, `phone_durations.png`.

`data/` is git-ignored, so **M3 and M4 get `phones.csv` from the team Drive** (or align
themselves: `make mfa-setup lexicon align-core align-rest phones`, ~15 min on 8 cores).

### What M3 and M4 need to know about `phones.csv`
- **`phone` keeps MFA's stress digit** (`AH0`, `AH1`): 69 labels + `sil`. Stressed vowels
  average 108 ms vs 67 ms unstressed, so stress is a real duration feature. For the 39
  base phones use `phone.str.rstrip("012")` (M4's templates may want this: fewer, fuller classes).
- **Silence rows:** `phone == "sil"`, `word == "sil"`, `word_idx == -1`, `idx_in_word == 0`,
  `word_len == 1`. Neighbouring pauses are merged, so there are never two `sil` rows in a row.
- `word_idx` counts spoken words only, from 0. `start_frame` is cumulative from 0 with no gaps:
  `start_frame + n_frames` is the next row's `start_frame`, and a clip's rows sum to its `n_frames`.
- `textgrid_to_phones.check()` asserts all of this on every run; `make test` covers the edge cases.

### Did the boundaries sit on the sound? (step 2, `notebooks/02_alignment.ipynb`)
- Visual check on 3 pilot clips, mel + boundaries: silences on the gaps, fricatives on the
  high-frequency noise, stops on their closures. The 17-frame `CH` in "chiswick" that looked
  long is closure + frication, which is correct for an affricate.
- Measured on 300 clips: **94.6%** of silence↔speech boundaries sit on an energy change in the
  right direction, vs 53.7% (chance) when every boundary is shifted by 5 frames.
- Only 0.024% of non-silence phones exceed 400 ms (drawn-out "the", hesitations, and
  `LJ001-0033` whose text says "mss" where the reader says "manuscripts"). No test_100 clip is
  a likelihood outlier. Section 5 of the notebook lists the worst clips for M5's failure log.

### Decisions made on the way
- **`messrs` needed no change.** It *is* in the MFA dictionary as `M EH1 S ER0 Z` ("messers"),
  which is how the reader says it. Expanding it to "messieurs" would have given the French
  pronunciation. M1's cleaner is untouched.
- **The real OOV problem was bigger: 366 words in 1,179 clips (9%)** — Warren Commission names
  (`hosty`, `connally`), compounds (`motorcade`, `palmprint`), acronyms (`fbi` ×168). Excluding
  them would have lost 9% of the data, skewed toward a few books, and one test_100 clip. Instead,
  `build_lexicon.py` gives each a pronunciation **before** alignment: 9 acronyms spelled letter
  by letter, 41 hand-corrected (G2P had "lineup" as *lin-ee-up*, "jebb" with no vowel), the rest
  from MFA's G2P model. The result is committed as `configs/oov.dict`; `make align-*` merges it
  into the dictionary automatically. `spn` would still be excluded, not patched — there just is none.
- `make mfa-setup` downloads pinned model files (acoustic + dictionary v3.0.0, G2P v2.0.0) instead
  of `mfa model download`, which pages the GitHub API and hit its 60-an-hour limit on the first call.
- Works with Anaconda as well as Miniforge (`--override-channels` keeps `defaults` out).
- Fixed `make corpus`: it called the script without its required `all` argument.

Timings on an 8-core M-series Mac, `JOBS=6`: pilot 47 s, core 4.4 min, rest 7.4 min, `make phones` 10 s.

The original task list, for reference:

1. `make mfa-setup`, then `conda run -n aligner mfa version`.
2. `make align-pilot`, then check 2–3 TextGrids against the sound. **Do not skip this.**
3. `make align-core JOBS=<cores>`, then `align-rest` if time allows.
4. `textgrid_to_phones.py` → `data/cache/phones.csv`
   (`clip_id, idx, phone, start_frame, n_frames, word, word_idx, idx_in_word, word_len`):
   times via `sec_to_frame()`, each clip sums exactly to `n_frames` (last phone absorbs the
   remainder), zero-frame phones borrow from the longer neighbour, silence is `"sil"`.
5. `data/cache/excluded.csv`: `clip_id, reason` — no TextGrid, `spn`, or mismatch > 2 frames.

---

## M3 — Duration model

Owns: `src/baseline/duration_features.py`, `src/baseline/duration_models.py`, `src/eval/duration_mae.py`.

Input `data/cache/phones.csv`, output `outputs/durations/test_pred.csv`:
`clip_id, idx, phone, n_frames, pred_mean, pred_ridge, pred_gbt, pred_best`.

1. **Today, before M2 finishes:** hand-write a 20-row fake `phones.csv` with the exact
   columns above and build the feature table against it. You lose a day if you wait.
2. Features per phoneme: the phone identity (one-hot), previous and next phone, position in
   word (`idx_in_word`, `word_len`), word length in phones, word index in the sentence,
   sentence length in phones, whether it is `sil`, and whether it is word-initial/final.
3. Three models, trained on `train`, tuned on `val`, reported on `test`:
   - **phone mean** — the mean duration of each phone type. The baseline to beat.
   - **ridge** — `sklearn.linear_model.Ridge` on the one-hot features.
   - **boosted trees** — `sklearn.ensemble.HistGradientBoostingRegressor` (fast, handles
     categoricals natively via `categorical_features=`).
4. Metric: **duration MAE in milliseconds per phoneme** = `mae_frames * 11.61`
   (use `SECONDS_PER_FRAME` from `src.config`). Report overall and per-phone.
5. Predictions must be positive integers that **sum to the clip's `n_frames`**, because M4
   builds a mel of exactly that length. Round, clamp to ≥1, then fix the largest phone to
   absorb the rounding error. Put that logic in one function and give it to M4.

---

## M4 — Sound  ← START NOW, nothing blocks you

Owns: `src/audio.py`, `src/data/extract_mels.py`, `src/baseline/phoneme_templates.py`,
`src/baseline/synthesize_baseline.py`.

1. `src/audio.py` with four functions:
   - `load_wav(path)` → float32 mono at 22,050 Hz, via `soundfile.read(path, dtype="float32")`
   - `save_wav(path, wav)` → `soundfile.write(path, wav, SAMPLE_RATE)`
   - `wav_to_mel(wav)` → `torch.log(MelSpectrogram(**MEL_KWARGS)(wav).clamp(min=LOG_CLAMP_MIN))`
   - `mel_to_wav(mel)` → HiFi-GAN `decode_batch`
2. `make mels` → `data/cache/mels/<id>.npy`, float16, shape `80 x T`.
   **Assert `mel.shape[1] == n_frames(n_samples)`** against `metadata_clean.csv` for every clip.
3. Phoneme templates: for each phone, average its frames across all training occurrences in
   **3 states** (begin / middle / end third of each occurrence) → `80 x 3` per phone.
   Save to `data/cache/templates.npz`.
4. Synthesis: for a phone with duration `d` frames, split `d` into three nearly equal parts
   and tile the matching state column. Concatenate across phones → `80 x T` → `mel_to_wav`.
   - **A1** uses the true MFA durations from `phones.csv`.
   - **A2** uses `pred_best` from M3's `test_pred.csv`.
   - **R1** is copy synthesis: the real cached mel straight through the vocoder.
5. Write `outputs/audio/{R1,A1,A2}/<id>.wav` for all 100 `test_100` IDs.

Verified during setup, so you do not have to rediscover it:
- The vocoder **loads and works on the current torch** (2.14.1 / torchaudio 2.11.0 /
  speechbrain 1.1.2). No downgrade needed.
- Our mel recipe is **byte-identical** to SpeechBrain's own `mel_spectrogram()` — MAE
  `0.000000`. Just use `MEL_KWARGS`; do not invent your own front end.
- Log-mel values land in roughly **−11.5 … +1.5**. That is correct, not a bug.
- **The vocoder pads its output.** `hyperparams.yaml` has `inference_padding: 5`, so a mel of
  `T` frames returns `(T + 10) * 256` samples. **Trim to the original `n_samples`**
  (or `T * 256`) or every clip gains ~116 ms of junk at the ends and WER suffers.
- Re-analysing vocoded audio gives a log-mel MAE of ~1.3 vs the original. That is normal GAN
  behaviour, **not** a recipe mismatch. Do not spend a day chasing it.
- First vocoder call downloads ~55 MB to `pretrained_models/` (git-ignored).

---

## M5 — Repo + evaluation  ← START NOW, nothing blocks you

Owns: `src/eval/whisper_wer.py`, `src/eval/make_results.py`, `notebooks/03_demo.ipynb`,
the failure log, the report and the presentation.

1. Whisper harness: `whisper.load_model(WHISPER_MODEL)` (`small.en` from config),
   transcribe a folder of wavs, score with `jiwer` against `text_clean` from
   `metadata_clean.csv`. Apply Whisper's own `EnglishTextNormalizer` to **both** sides —
   otherwise you measure punctuation, not intelligibility.
2. **Run it on R0 today.** R0 is just `data/raw/LJSpeech-1.1/wavs/<id>.wav` for the 100
   `test_100` IDs, which already exist. R0's WER is the floor; if it is not near 0, the
   harness is wrong and you want to know now, not on day 2.
3. `outputs/results/wer_<system>.csv`: `clip_id, system, ref, hyp, wer, cer`.
   Results table: one row per system (R0, R1, A1, A2) with mean WER, mean CER, duration MAE.
4. Failure log: the 10 worst clips per system, each tagged with a guessed cause —
   *flat prosody*, *wrong duration*, *slurred transition*, *misaligned phone*, *vocoder artefact*.
   This is an explicit deliverable, not an afterthought.
5. Small listening study: 5 clips × 4 systems, each teammate rates naturalness 1–5.
6. `samples/` gets a few wavs per system so the repo is listenable without running anything.

---

## What "done" looks like for the baseline

| System | What it is | Expected WER |
| --- | --- | --- |
| R0 | real recording | near 0 — sanity check on the harness |
| R1 | real mel → HiFi-GAN | near R0 — proves the mel recipe and vocoder |
| A1 | templates + true MFA durations | high — the point of the experiment |
| A2 | templates + predicted durations | highest — adds duration error on top |

The story the report tells: **R1 ≈ R0** proves the audio path is sound, so the gap at **A1**
is caused by templates having no coarticulation or prosody, and **A2 − A1** isolates the cost
of predicting durations. That gap is the motivation for the DL model.
