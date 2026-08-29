"""Resuming a run has to continue the schedule, not restart it.

A 30-epoch run at ~1.3 h an epoch is ~40 hours, which will not survive a session
boundary, a reboot or a kill. Twice already a run lost completed epochs because
nothing was written until the end; per-epoch `history.json` fixed the reporting, but
the weights alone are not a resume.

Three things have to come back or the continuation is a different experiment:

  optimiser   AdamW's exp_avg / exp_avg_sq. Without them the first few hundred
              steps after a resume run on rebuilt moment estimates.
  scheduler   the step counter. The LR is a linear warmup into a cosine whose
              length is `steps_per_epoch * epochs`, so a fresh scheduler restarts
              warmup and jumps the LR back to its peak -- a warm restart, not a
              resume.
  history     the per-epoch records, or the saved history forfeits every epoch before
              the resume.

And because the cosine's shape is a function of `epochs`, `batch_size`, `n_frames`
and `accum_steps`, resuming with any of those changed would continue a *different*
schedule from the one the stored step counter refers to. That is checked rather
than trusted.
"""

from __future__ import annotations

import pytest
import torch

from src.model.train import (
    TrainConfig,
    check_resumable,
    load_checkpoint,
    save_checkpoint,
)


def _fixture(lr: float = 0.1):
    model = torch.nn.Linear(4, 2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: 1.0)
    return model, optimiser, scheduler


def _advance(model, optimiser, scheduler, steps: int) -> None:
    for _ in range(steps):
        model(torch.randn(2, 4)).sum().backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        scheduler.step()


def test_the_scheduler_step_count_survives(tmp_path) -> None:
    """The one that matters: a fresh scheduler would restart warmup."""
    model, optimiser, scheduler = _fixture()
    _advance(model, optimiser, scheduler, 37)
    save_checkpoint(tmp_path / "last.pt", model=model, optimiser=optimiser,
                    scheduler=scheduler, epoch=3, history=[], config={},
                    steps_per_epoch=100)

    _, _, fresh_sched = _fixture()
    assert fresh_sched.last_epoch == 0
    state = load_checkpoint(tmp_path / "last.pt")
    fresh_sched.load_state_dict(state["scheduler"])
    assert fresh_sched.last_epoch == scheduler.last_epoch == 37


def test_the_optimiser_moments_survive(tmp_path) -> None:
    model, optimiser, scheduler = _fixture()
    _advance(model, optimiser, scheduler, 12)
    save_checkpoint(tmp_path / "last.pt", model=model, optimiser=optimiser,
                    scheduler=scheduler, epoch=1, history=[], config={},
                    steps_per_epoch=100)

    _, fresh_opt, _ = _fixture()
    fresh_opt.load_state_dict(load_checkpoint(tmp_path / "last.pt")["optimiser"])
    restored = next(iter(fresh_opt.state.values()))
    original = next(iter(optimiser.state.values()))
    assert torch.equal(restored["exp_avg"], original["exp_avg"])
    assert torch.equal(restored["exp_avg_sq"], original["exp_avg_sq"])
    assert int(restored["step"]) == 12


def test_the_next_epoch_and_history_survive(tmp_path) -> None:
    model, optimiser, scheduler = _fixture()
    history = [{"epoch": 0, "train_loss": 3.37}, {"epoch": 1, "train_loss": 2.56}]
    save_checkpoint(tmp_path / "last.pt", model=model, optimiser=optimiser,
                    scheduler=scheduler, epoch=2, history=history, config={"lr": 1e-3},
                    steps_per_epoch=13418)
    state = load_checkpoint(tmp_path / "last.pt")
    assert state["epoch"] == 2
    assert state["history"] == history
    assert state["steps_per_epoch"] == 13418


# --- the safeguard against continuing a different schedule ------------------------


def test_a_matching_config_is_resumable() -> None:
    cfg = TrainConfig(epochs=30, batch_size=4, n_frames=300)
    check_resumable({"epochs": 30, "batch_size": 4, "n_frames": 300,
                     "accum_steps": 1}, cfg, 13418, 13418)


@pytest.mark.parametrize("field,value", [
    ("epochs", 6),
    ("batch_size", 8),
    ("n_frames", 160),
    ("accum_steps", 2),
])
def test_a_changed_schedule_input_is_refused(field, value) -> None:
    """Each of these changes the cosine's length, so the stored step count no
    longer refers to the schedule being rebuilt."""
    cfg = TrainConfig(epochs=30, batch_size=4, n_frames=300)
    saved = {"epochs": 30, "batch_size": 4, "n_frames": 300, "accum_steps": 1}
    saved[field] = value
    with pytest.raises(RuntimeError, match=field):
        check_resumable(saved, cfg, 13418, 13418)


def test_a_changed_step_count_is_refused() -> None:
    """A different manifest gives a different steps_per_epoch and so a different
    cosine, even when every config field matches."""
    cfg = TrainConfig(epochs=30, batch_size=4, n_frames=300)
    saved = {"epochs": 30, "batch_size": 4, "n_frames": 300, "accum_steps": 1}
    with pytest.raises(RuntimeError, match="steps"):
        check_resumable(saved, cfg, 13418, 9000)
