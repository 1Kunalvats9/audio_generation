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

Status: **M1, M2, M3 and M5's harness are done.** The only work left is M4 running
`make mels templates synth`, then M5 re-running `make wer results`.

- **M4 is now the critical path.** `src/audio.py`, `extract_mels.py`, `phoneme_templates.py`
  and `synthesize_baseline.py` are all written, but none has been *run*: `data/cache/` has no
  mels and `outputs/audio/` has only the R1 probe clips. Everything else is waiting on that.
- **`phones.csv` and `excluded.csv` are committed** — M2 added a narrow `.gitignore` exception,
  so M3 and M4 just `git pull`: no MFA, no Drive. The rest of `data/` stays ignored, so cached
  mels, TextGrids and the lexicon do not travel through git — M4 rebuilds mels with `make mels`.
  A fresh clone still runs `make metadata corpus` first.
- **M5's summary table hides duration MAE until A1/A2 audio exists.** `build_summary_table` in
  `src/eval/make_results.py` only reads `mae.csv` inside its `if sys_name in wer_dfs` branch, so
  the column shows `-` until `wer_A1.csv` / `wer_A2.csv` are written. `mae.csv` itself is correct
  and is read fine (`{'A1': 0.0, 'A2': 20.13}`) — no action needed unless M5 wants the number
  visible earlier.

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

**`phones.csv` and `excluded.csv` are committed** (an exception in `.gitignore`), so M3 and M4
just `git pull` — no MFA needed. To regenerate them instead:
`make mfa-setup lexicon align-core align-rest phones` (~15 min on 8 cores).

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

## M3 — Duration model  ✅ DONE

Owns: `src/baseline/duration_features.py`, `src/baseline/duration_models.py`, `src/eval/duration_mae.py`.

Run with `make durations`. Input `data/cache/phones.csv`, outputs:
- `outputs/durations/test_pred.csv` — `clip_id, idx, phone, n_frames, pred_mean, pred_ridge, pred_gbt, pred_best`
  (299 test clips, 22,696 phones). **`pred_best` is what M4's A2 reads.**
- `outputs/durations/val_pred.csv` — same columns, used for model selection and metrics.
- `outputs/results/duration_models.csv` — the model comparison table for the report.
- `outputs/results/mae.csv` — `system, model, mae_ms, mae_frames`; the A1/A2 rows M5 joins on.

### Results (test split, 299 held-out LJ005 clips)

| model | MAE frames | MAE ms | MAE ms (no sil) | RMSE frames |
|---|---|---|---|---|
| phone mean (baseline) | 2.637 | 30.61 | 26.88 | 4.34 |
| ridge | 2.338 | 27.14 | 23.31 | 3.93 |
| **boosted trees (`pred_best`)** | **1.693** | **19.65** | **16.84** | **3.08** |

`pred_best = gbt`, chosen on **val** (1.702 frames vs ridge 2.414, mean 2.738) and then reported
on test — never selected on test. GBT beats the per-phone mean by **35.8%**. Early stopping ends
training at **1052 iterations**, so `max_iter=1800` is a ceiling the model never reaches — the
stopping criterion decides, not the budget.

Every number is given twice, including and excluding `sil`: silence is 5.3% of rows but averages
14.6 frames against 7.4 for real phones (max 481), so it dominates the headline MAE. The
no-sil column is the honest measure of learned phone duration; the incl-sil column is what A2
actually depends on.

### Decisions worth knowing

- **Target is `log(n_frames)`, not frames.** Durations are strongly right-skewed, so squared
  error on raw frames would be spent almost entirely on long pauses. FastSpeech 2's duration
  predictor does the same, which keeps us comparable to the anchor paper.
- **Predictions are written as unrounded floats.** `adjust_durations` in `src/audio.py` (M4)
  rounds, clamps to ≥ 1 and makes each clip sum to its true `n_frames`. Verified on all 100
  A2 clips: every one sums exactly to target with no phone below 1 frame.
- **No feature is derived from the clip's true length.** That would be a leak, and it would buy
  nothing: A2 rescales to `target_T` anyway, so a clip-level tempo term is divided straight back
  out. The model only has to get durations right *relative to each other* within a clip.
  `tests/test_duration_features.py` guards this — tripling every duration must leave the feature
  table byte-identical.
- Because log-space regression returns a geometric mean, predicted clip totals run about **4.3%
  short** (mean predicted/true = 0.957). Harmless: the rescaling absorbs it.
- **Pre-pausal lengthening is in the features** as `dist_sil_left` / `dist_sil_right`, the phone
  distance to the nearest silence. It is a real phonetic effect and a reportable finding rather
  than a feature dump.

### One fix landed in M4's `src/audio.py`

`adjust_durations` had a one-line bug in its residual redistribution:

```python
add = min(remaining, d[i] - 1) if remaining < 0 else remaining   # was
add = max(remaining, -(d[i] - 1)) if remaining < 0 else remaining  # now
```

When the rounded predictions overshoot the clip, `remaining` is negative and a phone can give up
at most `d[i] - 1` frames. `min()` on a negative number returned the whole deficit, so the first
phone took all of it and went negative — tripping the function's own `(d >= 1)` assert.

It broke **27 of the 299 test clips, 8 of the 100 A2 synthesises** (worst case LJ005-0225:
108 phones, rounded sum 835 against a target of 745, longest phone only 25 frames). It stayed
hidden because the existing tests only used small or positive residuals, and because the earlier
under-trained model under-predicted lengths so the residual was usually positive.

Fixed with a regression test and a 25-case property test in `tests/test_audio.py`; the property
test fails 15 ways against the old code. All 599 val+test clips now pass. **M4: this is in your
file — please pull before running `make synth`.**

### Features (21)

Categorical: `phone` (stress digit kept — M2 measured 108 ms stressed vs 67 ms unstressed),
`prev_phone`, `next_phone` (`<bos>`/`<eos>` at clip edges, grouped by `clip_id` so context never
crosses a clip), `cls` (silence/vowel/fricative/stop/nasal/other, same classes as M2's figure),
`stress`.

Numeric: `idx_in_word`, `word_len`, `rel_in_word`, `is_word_start`, `is_word_end`,
`word_idx_filled`, `n_words`, `rel_word`, `idx`, `n_phones`, `rel_phone`, `is_sil`,
`prev_is_sil`, `next_is_sil`, `dist_sil_left`, `dist_sil_right`.

**Trap for anyone touching this file:** silence rows in `phones.csv` carry sentinels —
`word_idx = -1`, `word = "sil"`, `word_len = 1`, `idx_in_word = 0`. So never use `word_idx` raw as
a numeric feature, count words with `word_idx.max() + 1` and not `nunique()`, and remember
`is_word_start` and `is_word_end` are both trivially true on every silence.

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
