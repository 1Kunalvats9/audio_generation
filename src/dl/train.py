"""M4 training entry point: teacher-forced acoustics, checkpoint/resume and CSV logs.

Run ``python -m src.dl.train --help``. No duration predictor is required for
D1: MFA durations condition training. M3's predictor can be jointly trained
with --joint-durations once their module lands, without editing their files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import N_MELS, ROOT
from src.dl.config import (BATCH_SIZE, CHECKPOINTS, DL, GRAD_CLIP, HIDDEN, LR,
                           MAX_STEPS, MEL_STATS_PATH, SEED, TRAIN_LOG, VOCAB_PATH,
                           WARMUP_STEPS, WEIGHT_DECAY, device)
from src.dl.decoder import MelDecoder, mel_loss
from src.dl.encoder import LengthRegulator, PhonemeEncoder


class AcousticModel(nn.Module):
    """Encoder -> true-duration regulator -> D1 decoder (or D2 diffusion)."""

    def __init__(self, vocab_size: int, system: str = "D1", joint_durations: bool = False,
                 hidden: int = HIDDEN):
        super().__init__()
        if system not in ("D1", "D2"):
            raise ValueError("system must be D1 or D2")
        self.spec = dict(vocab_size=vocab_size, system=system,
                         joint_durations=joint_durations, hidden=hidden)
        self.encoder = PhonemeEncoder(vocab_size, hidden=hidden)
        self.regulator = LengthRegulator()
        if system == "D1":
            self.generator = MelDecoder(hidden=hidden)
        else:
            from src.dl.diffusion import MelDiffusion
            self.generator = MelDiffusion(hidden=hidden)
        self.duration_predictor = None
        if joint_durations:
            from src.dl.duration_predictor import DurationPredictor
            self.duration_predictor = DurationPredictor(hidden=hidden)

    def condition(self, batch):
        enc = self.encoder(batch["phone_ids"], batch["phone_mask"])
        frames, mask = self.regulator(enc, batch["durations"], max_len=batch["mel"].shape[-1])
        if not torch.equal(mask, batch["frame_mask"]):
            raise ValueError("duration expansion disagrees with the dataset frame mask")
        return enc, frames, mask

    def loss(self, batch):
        enc, frames, mask = self.condition(batch)
        if self.spec["system"] == "D1":
            loss = mel_loss(self.generator(frames, mask), batch["mel"], mask)
        else:
            loss = self.generator.loss(batch["mel"], frames, mask)
        if self.duration_predictor is not None:
            from src.dl.duration_predictor import duration_loss
            loss = loss + duration_loss(self.duration_predictor(enc, batch["phone_mask"]),
                                        batch["durations"], batch["phone_mask"])
        return loss


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, target_device):
    return {k: v.to(target_device) if torch.is_tensor(v) else v for k, v in batch.items()}


def contract_hashes():
    """Reject accidental changes to the frozen inputs when resuming/inferencing."""
    paths = [ROOT / "configs/base.yaml", ROOT / "configs/dl.yaml", VOCAB_PATH, MEL_STATS_PATH]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def rng_state():
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, model, optimizer, step, epoch=0, batch_offset=0, **metadata):
    """Atomic replace prevents a disconnect from leaving a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), spec=model.spec, optimizer=optimizer.state_dict(),
                 step=step, epoch=epoch, batch_offset=batch_offset, rng=rng_state(),
                 contracts=contract_hashes(), config=DL, metadata=metadata)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model=None, optimizer=None, target_device="cpu", restore_random=False):
    # Checkpoints contain numpy/Python RNG state: load only trusted local/team files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["contracts"] != contract_hashes():
        raise ValueError("checkpoint's frozen configuration/vocabulary/mel statistics changed")
    if model is None:
        model = AcousticModel(**state["spec"])
    elif model.spec != state["spec"]:
        raise ValueError("checkpoint model architecture differs from the requested model")
    model.to(target_device)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        # map_location=cpu keeps RNG tensors valid; move optimiser moments explicitly.
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != "step":
                    values[key] = value.to(target_device)
    if restore_random:
        restore_rng(state["rng"])
    return model, state


@torch.no_grad()
def evaluate(model, loader, target_device):
    was_training = model.training
    state = rng_state()
    model.eval()
    total = count = 0
    try:
        for batch in loader:
            batch = to_device(batch, target_device)
            weight = int(batch["frame_mask"].sum()) * N_MELS
            total += model.loss(batch).item() * weight
            count += weight
        if not count:
            raise ValueError("validation loader has no real frames")
        return total / count
    finally:
        # Validation sampling/iterator creation must not change training dropout
        # or diffusion noise, including when a resumed run validates more often.
        model.train(was_training)
        restore_rng(state)


def prepare_log(path, resume_step=None):
    """Start a new CSV or discard rows newer than the resumed checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if resume_step is not None and path.exists():
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["step"]) <= resume_step]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)


def fit(model, train_loader, val_loader, *, max_steps=MAX_STEPS, target_device="cpu",
        checkpoint_dir=CHECKPOINTS / "D1", log_path=TRAIN_LOG, resume=None,
        lr=LR, warmup_steps=WARMUP_STEPS, log_every=None, val_every=None,
        ckpt_every=None, metadata=None):
    """Train to an absolute step count; resume restores optimiser, RNG and cursor.

    The project bucket sampler uses seed + epoch, allowing reconstruction of
    the same minibatch order without serialising a live DataLoader iterator.
    """
    if max_steps <= 0 or len(train_loader) == 0:
        raise ValueError("max_steps and training loader length must be positive")
    log_every = DL["train"]["log_every"] if log_every is None else log_every
    val_every = DL["train"]["val_every"] if val_every is None else val_every
    ckpt_every = DL["train"]["ckpt_every"] if ckpt_every is None else ckpt_every
    if min(log_every, val_every, ckpt_every) <= 0:
        raise ValueError("logging, validation and checkpoint intervals must be positive")
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    step = epoch = offset = 0
    metadata = dict(metadata or {})
    if resume:
        _, state = load_checkpoint(resume, model, optimizer, target_device, restore_random=True)
        step, epoch, offset = state["step"], state["epoch"], state["batch_offset"]
        metadata = {**state["metadata"], **metadata}
        options = metadata.get("fit_options")
        if options and options != dict(lr=lr, warmup_steps=warmup_steps):
            raise ValueError("resume must use the checkpoint's learning rate/warmup settings")
    metadata["fit_options"] = dict(lr=lr, warmup_steps=warmup_steps)
    prepare_log(log_path, step if resume else None)
    checkpoint_dir = Path(checkpoint_dir)
    model.train()
    window_total = window_count = 0
    last_val = None
    while step < max_steps:
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "epoch"):
            sampler.epoch = epoch
        # Iterator creation consumes a CPU RNG seed even with num_workers=0.
        # Isolate it, so rebuilding a mid-epoch iterator cannot shift dropout.
        with torch.random.fork_rng(devices=[]):
            iterator = iter(train_loader)
        for index, batch in enumerate(iterator):
            if index < offset:
                continue
            batch = to_device(batch, target_device)
            step += 1
            rate = lr * min(1.0, step / max(1, warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
            optimizer.step()
            weight = int(batch["frame_mask"].sum()) * N_MELS
            window_total += loss.item() * weight
            window_count += weight
            offset = index + 1
            if step % val_every == 0 or step == max_steps:
                last_val = evaluate(model, val_loader, target_device)
            else:
                last_val = None
            if step % log_every == 0 or last_val is not None:
                train_loss = window_total / window_count
                with Path(log_path).open("a", newline="") as f:
                    csv.writer(f).writerow([step, train_loss,
                                            "" if last_val is None else last_val, rate])
                print(f"step {step}: train={train_loss:.5f} val={last_val} lr={rate:.3g}", flush=True)
                window_total = window_count = 0
            if step % ckpt_every == 0 or step == max_steps:
                save_checkpoint(checkpoint_dir / f"step_{step:06d}.pt", model, optimizer,
                                step, epoch, offset, **metadata)
                save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer,
                                step, epoch, offset, **metadata)
            if step == max_steps:
                break
        epoch += 1
        offset = 0
    return {"step": step, "val_loss": last_val, "checkpoint": checkpoint_dir / "latest.pt"}


def training_loader(split, batch_size, limit=None, shuffle=True):
    from src.dl.dataset import LengthBucketSampler, PhonemeMelDataset, collate
    from src.utils import read_filelist
    ids = read_filelist(split)
    if limit:
        ids = ids[:limit]
    ds = PhonemeMelDataset(ids)
    actual = [item[0] for item in ds.items]
    if set(actual) != set(ids):
        raise RuntimeError(f"{split}: only {len(actual)}/{len(ids)} requested clips available; "
                           "restore cached mels or run make mels before training/synthesis")
    return DataLoader(ds, batch_sampler=LengthBucketSampler(ds, batch_size, shuffle),
                      collate_fn=collate, num_workers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["D1", "D2"], default="D1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=device(), choices=["cpu", "mps", "cuda"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--joint-durations", action="store_true")
    parser.add_argument("--overfit", action="store_true", help="train on the same 10 training clips for sanity checking")
    parser.add_argument("--overfit-ratio", type=float, default=0.1, help="pass when final/initial loss <= this ratio")
    parser.add_argument("--gate", type=Path, help="passed ten-clip checkpoint required before a full training run")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or not 0 < args.overfit_ratio < 1:
        parser.error("steps/batch size must be positive and overfit ratio must be between 0 and 1")
    seed_everything()
    from src.dl.vocab import PhoneVocab
    model = AcousticModel(PhoneVocab.load().size, args.system, args.joint_durations).to(args.device)
    directory = args.checkpoint_dir or CHECKPOINTS / args.system / ("overfit" if args.overfit else "full")
    if not args.overfit:
        gate = args.gate or CHECKPOINTS / args.system / "overfit" / "latest.pt"
        _, state = load_checkpoint(gate, target_device="cpu")
        if not state["metadata"].get("gate_passed") or state["spec"] != model.spec:
            raise ValueError("first pass the ten-clip --overfit gate for this model")
    split = "debug_200" if args.overfit else args.split
    train_loader = training_loader(split, args.batch_size, limit=10 if args.overfit else None)
    val_loader = (training_loader(split, args.batch_size, limit=10, shuffle=False)
                  if args.overfit else training_loader("val", args.batch_size, shuffle=False))
    if args.resume:
        _, resumed = load_checkpoint(args.resume, model, target_device=args.device)
        initial = resumed["metadata"].get("initial_loss")
    else:
        initial = None
    if initial is None:
        initial = evaluate(model, val_loader, args.device)
    meta = dict(initial_loss=initial, overfit=args.overfit, split=split,
                clip_ids=[x[0] for x in train_loader.dataset.items])
    result = fit(model, train_loader, val_loader, max_steps=args.steps, target_device=args.device,
                 checkpoint_dir=directory, log_path=args.log or (
                     directory / "train_log.csv" if args.overfit or args.system == "D2" else TRAIN_LOG),
                 resume=args.resume, metadata=meta)
    if args.overfit:
        final = evaluate(model, val_loader, args.device)
        passed = math.isfinite(final) and final <= initial * args.overfit_ratio
        state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        state["metadata"].update(gate_passed=passed, final_loss=final, overfit_ratio=args.overfit_ratio)
        temp = result["checkpoint"].with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(result["checkpoint"])
        print(f"ten-clip gate: initial={initial:.6f} final={final:.6f}, passed={passed}")
        return 0 if passed else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
