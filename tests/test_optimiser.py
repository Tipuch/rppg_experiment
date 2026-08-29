"""The optimiser: which parameters get weight decay, and the shape of the schedule.

Both are the kind of thing that invisibly costs a run. A learning-rate schedule
stepped per epoch instead of per step is a completely different curve when an epoch
is 109 steps in one configuration and 25,000 in another. Weight decay on the
Gaussian band's centre frequency is a prior on heart rate wearing a regulariser's
clothes.
"""

from __future__ import annotations

import pytest

from src.model.cfmamba import CFMambaPhys
from src.model.train import TrainConfig, _optimiser


def _built(**kwargs) -> tuple:
    cfg = TrainConfig(**kwargs)
    model = CFMambaPhys(dim=32, depth=2, n_frames=64)
    optimiser, scheduler = _optimiser(model, cfg, steps_per_epoch=100)
    return model, cfg, optimiser, scheduler


def _group_of(optimiser, parameter) -> dict:
    for group in optimiser.param_groups:
        if any(p is parameter for p in group["params"]):
            return group
    raise AssertionError("parameter is not in any group")


# --- weight decay ------------------------------------------------------------

def test_the_band_frequency_is_exempt_from_weight_decay() -> None:
    """theta_fc and theta_bw are 0-dim, and decay pulls them toward zero -- which
    after Eq. 14's sigmoid is the *midpoint* of the physiological range, 1.625 Hz
    or 97.5 bpm. That is not regularisation, it is a prior on the answer.
    """
    model, _, optimiser, _ = _built()
    mask = model.blocks[0].ffn.pts.mask
    assert _group_of(optimiser, mask.theta_fc)["weight_decay"] == 0.0
    assert _group_of(optimiser, mask.theta_bw)["weight_decay"] == 0.0


def test_mamba_state_parameters_are_exempt_even_though_the_biases_are_3d() -> None:
    """Mamba-3 marks `dt_bias` and `D` `_no_weight_decay` itself. It does not mark
    the B and C biases, and those are 3-D -- so a rule based on dimension alone
    would decay them. Both are initialised to 1 and hold B and C away from zero
    after the RMS norm, so decay there pulls the layer toward a degenerate
    initialisation rather than toward a simpler function; mamba_layer.py sets the
    flag and that is what is respected.
    """
    model, _, optimiser, _ = _built()
    scan = model.blocks[0].mamba.mamba
    for parameter in (scan.dt_bias, scan.D, scan.B_bias, scan.C_bias):
        assert getattr(parameter, "_no_weight_decay", False)
        assert _group_of(optimiser, parameter)["weight_decay"] == 0.0
    assert scan.B_bias.ndim == 3


def test_norms_and_biases_are_exempt_and_weights_are_not() -> None:
    model, cfg, optimiser, _ = _built()
    block = model.blocks[0]
    assert _group_of(optimiser, block.norm1.weight)["weight_decay"] == 0.0
    assert _group_of(optimiser, block.norm1.bias)["weight_decay"] == 0.0
    assert _group_of(optimiser, block.mamba.gate.weight)["weight_decay"] == cfg.weight_decay
    assert _group_of(optimiser, block.ffn.proj_in.weight)["weight_decay"] == cfg.weight_decay


def test_every_trainable_parameter_lands_in_exactly_one_group() -> None:
    """A parameter missing from the optimiser trains not at all, and nothing cautions."""
    model, _, optimiser, _ = _built()
    grouped = [p for g in optimiser.param_groups for p in g["params"]]
    ids = [id(p) for p in grouped]
    assert len(ids) == len(set(ids)), "a parameter is in two groups"
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == set(ids)


def test_adamw_defaults_are_used_where_the_papers_are_silent() -> None:
    _, cfg, optimiser, _ = _built()
    assert (cfg.betas, cfg.eps) == ((0.9, 0.999), 1e-8)
    assert optimiser.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimiser.param_groups[0]["eps"] == 1e-8


# --- the schedule ------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step()`")
def test_the_schedule_warms_up_then_decays() -> None:
    _, cfg, optimiser, scheduler = _built(epochs=10, warmup_frac=0.1)
    total = 100 * 10
    seen = []
    for _ in range(total):
        seen.append(optimiser.param_groups[0]["lr"])
        scheduler.step()
    warmup = int(total * 0.1)
    assert seen[0] < seen[warmup - 1], "the warmup is not rising"
    assert seen[warmup - 1] == pytest.approx(cfg.lr, rel=1e-6), "warmup should reach lr"
    assert seen[-1] < seen[warmup], "the cosine is not falling"


def test_the_first_step_is_not_the_full_learning_rate() -> None:
    """The point of the warmup: the largest and noisiest gradients of the run
    arrive first, and a plain cosine schedule meets them at maximum lr."""
    _, cfg, optimiser, _ = _built()
    assert optimiser.param_groups[0]["lr"] < 0.2 * cfg.lr


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step()`")
def test_the_floor_is_a_fraction_of_lr_not_zero() -> None:
    """Decaying to exactly zero wastes the last epoch."""
    _, cfg, optimiser, scheduler = _built(epochs=5, min_lr_frac=0.01)
    for _ in range(100 * 5 + 50):
        scheduler.step()
    final = optimiser.param_groups[0]["lr"]
    assert final == pytest.approx(cfg.lr * cfg.min_lr_frac, rel=1e-3)


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step()`")
def test_the_schedule_is_scaled_by_total_steps_not_epochs() -> None:
    """An epoch is 109 steps on UBFC and ~25,000 on the full corpus. A per-epoch
    schedule would be a different curve for each."""
    model = CFMambaPhys(dim=32, depth=1, n_frames=64)
    lrs = {}
    for steps in (100, 1000):
        cfg = TrainConfig(epochs=4)
        optimiser, scheduler = _optimiser(model, cfg, steps_per_epoch=steps)
        for _ in range(200):
            scheduler.step()
        lrs[steps] = optimiser.param_groups[0]["lr"]
    # 200 steps is halfway through the short run and 5% into the long one, so the
    # learning rate must differ -- if it did not, the schedule is ignoring length.
    assert lrs[100] < lrs[1000]


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step()`")
def test_both_groups_share_the_schedule() -> None:
    _, _, optimiser, scheduler = _built()
    for _ in range(50):
        scheduler.step()
    assert optimiser.param_groups[0]["lr"] == pytest.approx(
        optimiser.param_groups[1]["lr"]
    )
