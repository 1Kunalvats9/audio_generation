# ---------- settings ----------
VENV        := .venv
PY          := $(VENV)/bin/python
MFA         := conda run -n aligner mfa
AM          := english_us_arpa english_us_arpa
JOBS        ?= 4
TORCH_INDEX ?=
URL         := https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
ARCHIVE     := data/downloads/LJSpeech-1.1.tar.bz2
LJ          := data/raw/LJSpeech-1.1

.PHONY: help setup install check data clean-data mfa-setup \
        metadata corpus align-pilot align-core align-rest phones mels \
        durations templates synth wer results test

help:
	@echo "make setup       venv + packages + dataset + import check (new members start here)"
	@echo "make data        verify and extract the dataset only"
	@echo "make check       test that all packages import"
	@echo "make mfa-setup   Member 2 only: MFA environment + models (needs Miniforge)"
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
	conda create -n aligner -c conda-forge montreal-forced-aligner -y
	$(MFA) model download acoustic english_us_arpa
	$(MFA) model download dictionary english_us_arpa
	$(MFA) model download g2p english_us_arpa

# ---------- pipeline (each works once its script exists) ----------
metadata:    ; $(PY) -m src.data.prepare_metadata && $(PY) -m src.data.split
corpus:      ; $(PY) -m src.data.prepare_mfa_corpus
align-pilot: ; $(MFA) align data/mfa_pilot $(AM) data/textgrids/pilot --clean
align-core:  ; $(MFA) align data/mfa_core  $(AM) data/textgrids/core -j $(JOBS) --clean
align-rest:  ; $(MFA) align data/mfa_rest  $(AM) data/textgrids/rest -j $(JOBS) --clean
phones:      ; $(PY) -m src.data.textgrid_to_phones
mels:        ; $(PY) -m src.data.extract_mels
durations:   ; $(PY) -m src.baseline.duration_models && $(PY) -m src.eval.duration_mae
templates:   ; $(PY) -m src.baseline.phoneme_templates
synth:       ; $(PY) -m src.baseline.synthesize_baseline
wer:         ; $(PY) -m src.eval.whisper_wer --all
results:     ; $(PY) -m src.eval.make_results
test:        ; $(PY) -m pytest tests