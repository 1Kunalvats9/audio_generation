"""Synthesis contract: denormalise, preserve identity, trim waveform length."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import N_MELS
from src.dl.mel_stats import MelStats
from src.dl.synthesize import prepare_clips, synthesize
from src.dl.train import AcousticModel
from src.dl.vocab import PhoneVocab


def test_vocoder_receives_denormalised_mel_and_output_is_trimmed(tmp_path, monkeypatch):
    from src import audio
    seen = {}
    def vocode(mel):
        seen["mel"] = mel.clone()
        return np.ones(1000, dtype=np.float32)
    monkeypatch.setattr(audio, "mel_to_wav", vocode)
    monkeypatch.setattr(audio, "save_wav", lambda path, wav: seen.update(path=path, wav=wav))
    model = AcousticModel(4, hidden=16).eval()
    for p in model.generator.parameters():
        p.data.zero_()
    clips = [dict(clip_id="LJ_fake", ids=[1, 2], durations=np.array([2, 3]),
                  n_frames=5, n_samples=800)]
    stats = MelStats(np.full(N_MELS, -5, np.float32), np.full(N_MELS, 2, np.float32), 1, 5)
    synthesize(model, clips, stats, tmp_path)
    assert seen["mel"].shape == (N_MELS, 5)
    assert (seen["mel"] == -5).all()
    assert len(seen["wav"]) == 800
    assert seen["path"].name == "LJ_fake.wav"


def test_prediction_join_uses_phone_indices_and_rejects_mismatch(tmp_path):
    phones = pd.DataFrame(dict(clip_id=["a", "a"], idx=[0, 1], phone=["AA0", "sil"], n_frames=[2, 3]))
    meta = pd.DataFrame(dict(id=["a"], n_frames=[5], n_samples=[1000]))
    pred = phones.iloc[::-1].assign(pred_best=[2.5, 1.5])
    paths = {"phones_csv": tmp_path / "phones.csv", "metadata_csv": tmp_path / "meta.csv",
             "predictions_csv": tmp_path / "pred.csv"}
    phones.to_csv(paths["phones_csv"], index=False)
    meta.to_csv(paths["metadata_csv"], index=False)
    pred.to_csv(paths["predictions_csv"], index=False)
    vocab = PhoneVocab(("<pad>", "AA0", "sil"))
    clips = prepare_clips(["a"], vocab, "gbt", **paths)
    assert clips[0]["ids"] == [1, 2]
    assert clips[0]["durations"].sum() == 5
    pred["phone"] = "wrong"
    pred.to_csv(paths["predictions_csv"], index=False)
    with pytest.raises(ValueError, match="do not match"):
        prepare_clips(["a"], vocab, "gbt", **paths)


def test_missing_neural_predictor_fails_before_writing(tmp_path):
    model = AcousticModel(4, hidden=16)
    with pytest.raises(ValueError, match="jointly"):
        synthesize(model, [], None, tmp_path / "new", duration_source="neural")
    assert not (tmp_path / "new").exists()
