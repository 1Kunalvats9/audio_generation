"""Exercise actual optimisation, validation and interrupted training on fake data."""
import csv
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import N_MELS
from src.dl.dataset import collate
from src.dl.train import (AcousticModel, evaluate, fit, load_checkpoint,
                          save_checkpoint, seed_everything)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def loader():
    generator = torch.Generator().manual_seed(10)
    items = [dict(clip_id=f"fake_{i}", phone_ids=torch.tensor([1, 2, 3]),
                  durations=torch.tensor([2, 1, 3]),
                  mel=torch.randn(N_MELS, 6, generator=generator)) for i in range(6)]
    return DataLoader(items, batch_size=2, collate_fn=collate)


def test_teacher_forced_model_composes_and_backpropagates():
    model = AcousticModel(8, hidden=16)
    loss = model.loss(next(iter(loader())))
    loss.backward()
    assert model.encoder.embed.weight.grad.abs().sum() > 0
    assert model.generator.proj.weight.grad.abs().sum() > 0


def test_checkpoint_restores_optimizer_rng_and_predictions(tmp_path):
    seed_everything()
    model = AcousticModel(8, hidden=16)
    opt = torch.optim.AdamW(model.parameters())
    batch = next(iter(loader()))
    model.loss(batch).backward()
    opt.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, 7, 2, 1)
    expected = (torch.rand(3), np.random.rand(3), random.random())
    clone = AcousticModel(8, hidden=16)
    other_opt = torch.optim.AdamW(clone.parameters())
    _, state = load_checkpoint(path, clone, other_opt, restore_random=True)
    assert state["step"] == 7 and state["batch_offset"] == 1
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]
    assert len(other_opt.state) == len(opt.state) > 0
    model.eval()
    clone.eval()
    assert torch.equal(model.loss(batch), clone.loss(batch))


def test_resume_matches_uninterrupted_training_across_epoch(tmp_path):
    def run(folder, steps, resume=None):
        model = AcousticModel(8, hidden=16)
        fit(model, loader(), loader(), max_steps=steps, checkpoint_dir=folder,
            log_path=folder / "log.csv", resume=resume, warmup_steps=3,
            log_every=1, val_every=3, ckpt_every=2)
        return model
    seed_everything()
    full = run(tmp_path / "full", 7)
    seed_everything()
    run(tmp_path / "split", 2)
    resumed = run(tmp_path / "split", 7, tmp_path / "split/latest.pt")
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    with (tmp_path / "split/log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(1, 8))
    assert float(rows[-1]["val_loss"]) > 0
    assert float(rows[0]["lr"]) < float(rows[-1]["lr"])


def test_checkpoint_rejects_changed_contract(tmp_path):
    model = AcousticModel(8, hidden=16)
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 1)
    state = torch.load(path, weights_only=False)
    state["contracts"]["configs/base.yaml"] = "changed"
    torch.save(state, path)
    with pytest.raises(ValueError, match="frozen"):
        load_checkpoint(path)


def test_validation_restores_training_mode():
    model = AcousticModel(8, hidden=16)
    model.train()
    assert evaluate(model, loader(), "cpu") > 0
    assert model.training


def test_end_to_end_model_overfits_ten_synthetic_examples(tmp_path):
    """Wiring check only: real LJSpeech overfit remains a separate CLI gate."""
    seed_everything()
    lookup = torch.randn(31, N_MELS)
    items = []
    for i in range(10):
        ids = torch.tensor([3 * i + 1, 3 * i + 2, 3 * i + 3])
        durations = torch.tensor([2, 3, 4])
        mel = lookup[ids - 1].repeat_interleave(durations, dim=0).T.contiguous()
        items.append(dict(clip_id=f"synthetic_{i}", phone_ids=ids,
                          durations=durations, mel=mel))
    data = DataLoader(items, batch_size=10, collate_fn=collate)
    model = AcousticModel(32, hidden=32)
    initial = evaluate(model, data, "cpu")
    fit(model, data, data, max_steps=400, checkpoint_dir=tmp_path,
        log_path=tmp_path / "synthetic.csv", lr=0.003, warmup_steps=20,
        log_every=100, val_every=100, ckpt_every=400)
    final = evaluate(model, data, "cpu")
    assert final < initial * 0.1
    # A synthetic test must never create a checkpoint claiming a real-data gate.
    _, state = load_checkpoint(tmp_path / "latest.pt")
    assert not state["metadata"].get("gate_passed", False)
