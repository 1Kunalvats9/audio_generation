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

## M2 — Alignment  ← START NOW, you are the critical path

Owns: `src/data/textgrid_to_phones.py`.

1. `make mfa-setup` (needs Miniforge), then `conda run -n aligner mfa version`.
2. `make align-pilot` on the 200 clips already in `data/mfa_pilot/`. Open 2–3 TextGrids in
   Praat and check the phone boundaries sit on the sound. **Do not skip this** — if the
   pilot is wrong, the 4,600-clip run wastes hours.
3. `make align-core` with `JOBS=<your cpu cores>`. Then `align-rest` if time allows;
   core alone (test + val + 4,000 train) is enough for the baseline.
4. Write `textgrid_to_phones.py` → `data/cache/phones.csv`:
   `clip_id, idx, phone, start_frame, n_frames, word, word_idx, idx_in_word, word_len`
   - Parse with `praatio` (already installed): `praatio.textgrid.openTextgrid(path, False)`.
   - Convert times with `sec_to_frame()` from `src.config` — not your own rounding.
   - **Durations of each clip must sum exactly to its `n_frames`** from
     `data/metadata_clean.csv`. Fix the last phone to absorb the remainder.
   - Zero-frame phones borrow 1 frame from the **longer** neighbour.
   - Silence is `"sil"` (`src.config.SIL`).
5. `data/cache/excluded.csv`: `clip_id, reason` — no TextGrid, contains `spn`, or
   length mismatch > 2 frames.

Known traps found during setup:
- **17 clips contain `messrs.`**, which our cleaner does not expand. MFA will likely mark it
  OOV. Either add `"messrs": "messieurs"` to `ABBREVIATIONS` (tell M1) or let those clips
  land in `excluded.csv`. Decide before `align-core`, not after.
- `spn` means MFA could not identify a sound. Those clips must be excluded, not patched.

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
