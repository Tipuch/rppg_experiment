"""Multi-temporal constraint Mamba on a Mamba-3 core, RhythmMamba Section 3.3.

GPU-only: Mamba-3's scan is a Triton kernel with no CPU path. Everything else in
the package is tested on CPU, which is why this seam exists.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.mamba_layer import (
    DEFAULT_CHUNK_SIZE,
    DIRECTIONS,
    MultiTemporalMamba,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3's scan kernel is CUDA-only"
)


@cuda
def test_shape_is_preserved() -> None:
    layer = MultiTemporalMamba(dim=32).cuda()
    x = torch.randn(2, 64, 32, device="cuda")
    assert layer(x).shape == x.shape


@cuda
def test_the_scan_weights_are_shared_across_paths() -> None:
    """"We replace multi-temporal fusion with multi-temporal constraint" -- one set
    of weights seeing three scales, not three sets. A Mamba per path would turn the
    constraint into an ensemble and change what the layer means."""
    layer = MultiTemporalMamba(dim=16, paths=3)
    scans = [m for n, m in layer.named_modules() if n.endswith("mamba")]
    assert len(scans) == 1


@cuda
@pytest.mark.parametrize("paths", [1, 2, 3, 4])
def test_every_path_count_returns_the_full_sequence(paths: int) -> None:
    """Each path re-splits and recombines to length T. Dropping a remainder here
    would shorten the output and desynchronise every target downstream."""
    layer = MultiTemporalMamba(dim=16, paths=paths).cuda()
    assert layer(torch.randn(1, 64, 16, device="cuda")).shape == (1, 64, 16)


@cuda
@pytest.mark.parametrize("n_frames", [40, 63, 100, 160, 300])
def test_a_length_that_does_not_divide_evenly_still_works(n_frames: int) -> None:
    """T=63 splits into 4 as 16/16/16/15; the loop path must cover the remainder."""
    layer = MultiTemporalMamba(dim=16).cuda()
    out = layer(torch.randn(1, n_frames, 16, device="cuda"))
    assert out.shape == (1, n_frames, 16)
    assert torch.isfinite(out).all()


@cuda
@pytest.mark.parametrize("n_frames", [DEFAULT_CHUNK_SIZE + 1, DEFAULT_CHUNK_SIZE,
                                      DEFAULT_CHUNK_SIZE - 1])
def test_a_sequence_shorter_than_a_chunk_still_scans(n_frames: int) -> None:
    """Mamba-3 splits the sequence into chunks and runs a recurrence between them,
    so the chunk width is a boundary the Mamba-1 scan did not have. The quartered
    path is what reaches it first: at T=300 it is 75 frames, and every path of a
    short clip lands under one chunk."""
    layer = MultiTemporalMamba(dim=16).cuda()
    out = layer(torch.randn(1, n_frames, 16, device="cuda"))
    assert out.shape == (1, n_frames, 16)
    assert torch.isfinite(out).all()


def _last_frame_influence(direction: str) -> torch.Tensor:
    """|output(x) - output(x with the last frame perturbed)| at every position.

    Probed a few steps back from the perturbation rather than at position 0: a
    state space model's state decays, so by 31 steps the reverse pass's influence
    has fallen to ~1e-4 and an assertion there would be testing the decay rate
    rather than the direction of the scan.
    """
    torch.manual_seed(0)
    layer = MultiTemporalMamba(dim=16, paths=1, direction=direction).cuda().eval()
    x = torch.randn(1, 32, 16, device="cuda")
    y = x.clone()
    y[0, -1] += 5.0
    with torch.no_grad():
        return (layer(x) - layer(y)).abs().amax(dim=2)[0]


@cuda
def test_a_unidirectional_scan_is_exactly_causal() -> None:
    """A forward scan cannot see the future -- not approximately, exactly. Every
    position before the perturbed frame must be exactly equal. If it is not, the
    sequence axis has been transposed and the model is not scanning time at all."""
    influence = _last_frame_influence("none")
    assert float(influence[:-1].max()) == 0.0
    assert float(influence[-1]) > 1e-3


@cuda
@pytest.mark.parametrize("direction", ["shared", "separate"])
def test_a_bidirectional_scan_lets_the_future_reach_the_past(direction: str) -> None:
    """The other half of the pair, and the only direct evidence the reverse pass
    runs at all: the perturbation must appear at positions ahead of it."""
    influence = _last_frame_influence(direction)
    assert float(influence[-4]) > 1e-3
    assert float(influence[:-1].max()) > 1e-3


def test_the_default_is_unidirectional() -> None:
    """Neither mamba_ssm.Mamba nor Mamba3 has a bimamba flag; PhysMamba's came
    from a fork."""
    assert MultiTemporalMamba(dim=16).direction == "none"


# --- the core is Mamba-3 ------------------------------------------------------

def test_there_is_no_short_convolution() -> None:
    """Mamba-1 put a width-4 depthwise conv in front of the scan. Mamba-3 has
    none: the trapezoid rule is itself an implicit width-2 convolution on the
    state-input B_t x_t, and the B/C biases supply the data-independent part. The
    paper measures adding the conv back as worse: 15.72 ppl against 15.85 for
    "Mamba-3 + conv" (Table 5a).
    """
    layer = MultiTemporalMamba(dim=16)
    assert not [name for name, _ in layer.named_modules() if "conv" in name]


def test_b_and_c_are_normalised_and_biased_from_one() -> None:
    """The two additions inside the projection: an RMS norm on B and C, and a
    learnable bias added after it. All-ones init is the paper's choice; results are
    insensitive to the value as long as it stays positive (Appendix F)."""
    scan = MultiTemporalMamba(dim=16).mamba
    assert scan.B_norm is not None and scan.C_norm is not None
    assert torch.equal(scan.B_bias, torch.ones_like(scan.B_bias))
    assert torch.equal(scan.C_bias, torch.ones_like(scan.C_bias))


def test_the_whole_state_is_rotated() -> None:
    """Proposition 2 puts a 2x2 rotation on every pair of state dimensions, so a
    state of N carries N/2 angles. Rotating only half of it -- the reference
    implementation's default -- would leave four angles at d_state=16, too coarse
    a frequency basis to resolve a heart rate."""
    scan = MultiTemporalMamba(dim=16, d_state=16).mamba
    assert scan.num_rope_angles == 16 // 2


def test_the_state_transition_carries_a_phase_not_only_a_decay() -> None:
    """The point of the rotation, given as a measurement: perturb one frame and
    the response down the sequence must change sign rather than only shrink. A
    real decay -- Mamba-1's transition -- can only attenuate, so every downstream
    response would keep one sign."""
    torch.manual_seed(0)
    layer = MultiTemporalMamba(dim=16, paths=1).cuda().eval()
    x = torch.randn(1, 128, 16, device="cuda")
    y = x.clone()
    y[0, 0] += 5.0
    with torch.no_grad():
        response = (layer(y) - layer(x))[0, :, 0]
    assert response.max() > 0 and response.min() < 0


def test_the_mamba_3_core_is_cheaper_than_the_mamba_1_layer_it_replaced() -> None:
    """No short conv, no separate x_proj, and a per-head scalar A in place of
    Mamba-1's (d_inner, d_state) matrix. That is what covers for the rotation
    angles, the trapezoid gate and the B/C norms, and then some."""
    from mamba_ssm import Mamba

    mamba1 = Mamba(d_model=80, d_state=16, d_conv=4, expand=2)
    mamba3 = MultiTemporalMamba(dim=80).mamba
    assert sum(p.numel() for p in mamba3.parameters()) < sum(
        p.numel() for p in mamba1.parameters()
    )


def test_shared_bidirectional_costs_no_parameters() -> None:
    """Which is what keeps the ablation runnable without re-fitting the budget."""
    forward = MultiTemporalMamba(dim=16, direction="none")
    shared = MultiTemporalMamba(dim=16, direction="shared")
    separate = MultiTemporalMamba(dim=16, direction="separate")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(shared) == count(forward)
    assert count(separate) > 1.8 * count(forward)


def test_an_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown direction"):
        MultiTemporalMamba(dim=8, direction="bimamba")


@cuda
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_every_direction_preserves_shape(direction: str) -> None:
    layer = MultiTemporalMamba(dim=16, direction=direction).cuda()
    assert layer(torch.randn(2, 64, 16, device="cuda")).shape == (2, 64, 16)


@cuda
def test_slicing_actually_changes_the_result() -> None:
    """Three paths must not collapse to three copies of the same computation, or
    the multi-temporal constraint has no effect."""
    torch.manual_seed(0)
    one = MultiTemporalMamba(dim=16, paths=1).cuda().eval()
    three = MultiTemporalMamba(dim=16, paths=3).cuda().eval()
    three.load_state_dict(one.state_dict())
    x = torch.randn(1, 64, 16, device="cuda")
    with torch.no_grad():
        assert not torch.allclose(one(x), three(x) / 3.0, atol=1e-3)


@cuda
def test_gradients_reach_every_parameter() -> None:
    layer = MultiTemporalMamba(dim=16).cuda()
    layer(torch.randn(2, 32, 16, device="cuda")).square().mean().backward()
    dead = [n for n, p in layer.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
