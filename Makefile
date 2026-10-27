# ---------- settings ----------
VENV        := .venv
PY          := $(VENV)/bin/python
MFA         := conda run -n aligner mfa
LEXICON     := data/lexicon/english_us_arpa_lj.dict
AM          := $(LEXICON) english_us_arpa
MODELS      := https://github.com/MontrealCorpusTools/mfa-models/releases/download
JOBS        ?= 4
TORCH_INDEX ?=
URL         := https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
ARCHIVE     := data/downloads/LJSpeech-1.1.tar.bz2
LJ          := data/raw/LJSpeech-1.1

.PHONY: help setup install check data clean-data mfa-setup \
        metadata corpus lexicon align-pilot align-core align-rest phones mels \
        durations templates synth wer results test

help:
	@echo "make setup       venv + packages + dataset + import check (new members start here)"
	@echo "make data        verify and extract the dataset only"
	@echo "make check       test that all packages import"
	@echo "make mfa-setup   Member 2 only: MFA environment + models (needs Miniforge)"
	@echo "make lexicon     Member 2 only: MFA dictionary + configs/oov.dict (make align-* runs it)"
	@echo "make clean-data  delete the extracted dataset (the archive stays)"

# ---------- setup ----------
setup: install data check

$(PY):
	uv venv --python 3.11 $(VENV)

install: $(PY)
	uv pip install --python $(PY) torch torchaudio $(if $(TORCH_INDEX),--index-url $(TORCH_INDEX))
	uv pip install --python $(PY) -r requirements.txt

check:
	$(PY) -c "import torch, torchaudio, librosa, soundfile, speechbrain, whisper, jiwer, sklearn, praatio, yaml, pandas; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| GPU:', torch.cuda.is_available())"
	$(PY) -m src.config

# ---------- dataset ----------
$(ARCHIVE):
	@mkdir -p data/downloads
	@echo "No archive in data/downloads - downloading 2.6 GB (faster: copy it here from the pen drive)"
	curl -L --fail -o $@.part $(URL)
	mv $@.part $@

$(LJ)/metadata.csv: $(ARCHIVE)
	shasum -a 256 -c data/checksums.txt
	@mkdir -p data/raw
	@echo "Extracting, takes a few minutes..."
	tar -xjf $(ARCHIVE) -C data/raw
	@touch $@

data: $(LJ)/metadata.csv
	@echo "wav files: $$(ls $(LJ)/wavs | wc -l | tr -d ' ')  (expected 13100)"

clean-data:
	rm -rf $(LJ)

# ---------- MFA (Member 2 only) ----------
mfa-setup:
	conda create -n aligner -c conda-forge --override-channels montreal-forced-aligner -y
# Pinned release files, not `mfa model download`: that pages through the GitHub
# API and hits its 60-requests-an-hour limit after one or two calls.
	@mkdir -p data/downloads/mfa/acoustic data/downloads/mfa/dictionary data/downloads/mfa/g2p
	curl -L --fail -o data/downloads/mfa/acoustic/english_us_arpa.zip $(MODELS)/acoustic-english_us_arpa-v3.0.0/english_us_arpa.zip
	curl -L --fail -o data/downloads/mfa/dictionary/english_us_arpa.dict $(MODELS)/dictionary-english_us_arpa-v3.0.0/english_us_arpa.dict
	curl -L --fail -o data/downloads/mfa/g2p/english_us_arpa.zip $(MODELS)/g2p-english_us_arpa-v2.0.0/english_us_arpa.zip
	$(MFA) model save acoustic data/downloads/mfa/acoustic/english_us_arpa.zip --name english_us_arpa --overwrite
	$(MFA) model save dictionary data/downloads/mfa/dictionary/english_us_arpa.dict --name english_us_arpa --overwrite
	$(MFA) model save g2p data/downloads/mfa/g2p/english_us_arpa.zip --name english_us_arpa --overwrite

# ---------- pipeline (each works once its script exists) ----------
metadata:    ; $(PY) -m src.data.prepare_metadata && $(PY) -m src.data.split
corpus:      ; $(PY) -m src.data.prepare_mfa_corpus all
lexicon:     ; $(PY) -m src.data.build_lexicon
$(LEXICON): configs/oov.dict src/data/build_lexicon.py ; $(PY) -m src.data.build_lexicon
align-pilot: $(LEXICON) ; $(MFA) align data/mfa_pilot $(AM) data/textgrids/pilot -j $(JOBS) --clean
align-core:  $(LEXICON) ; $(MFA) align data/mfa_core  $(AM) data/textgrids/core -j $(JOBS) --clean
align-rest:  $(LEXICON) ; $(MFA) align data/mfa_rest  $(AM) data/textgrids/rest -j $(JOBS) --clean
phones:      ; $(PY) -m src.data.textgrid_to_phones
mels:        ; $(PY) -m src.data.extract_mels
durations:   ; $(PY) -m src.baseline.duration_models && $(PY) -m src.eval.duration_mae
templates:   ; $(PY) -m src.baseline.phoneme_templates
synth:       ; $(PY) -m src.baseline.synthesize_baseline
wer:         ; $(PY) -m src.eval.whisper_wer --all
results:     ; $(PY) -m src.eval.make_results
test:        ; $(PY) -m pytest tests