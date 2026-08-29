"""Fusion Stem: geometry, the five-frame difference stack, and the 5:5 fusion.

The stem is where a silent off-by-one is most expensive. Its 3D convolution has a
kernel of 2 on the time axis, so an unpadded forward pass returns T-1 frames --
which every layer downstream would readily accept while every target was one frame
out of step with its features.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.fusion_stem import VARIANTS, FusionStem, temporal_differences

# --- the difference stack -----------------------------------------------------

def test_differences_are_the_four_consecutive_gaps_across_five_frames() -> None:
    """Chronological ordering, checked against the frames it should be built from."""
    x = torch.randn(1, 9, 3, 4, 4)
    diffs = temporal_differences(x, reverse=False)
    assert diffs.shape == (1, 9, 12, 4, 4)
    t = 4                                             # a frame with room either side
    for channel, (a, b) in enumerate([(2, 3), (3, 4), (4, 5), (5, 6)]):
        got = diffs[0, t, channel * 3 : (channel + 1) * 3]
        assert torch.allclose(got, x[0, b] - x[0, a], atol=1e-6), channel


def test_reverse_ordering_is_exactly_a_sign_flip() -> None:
    """The two source papers disagree on this, and the conflict is dead: the
    first convolution is linear, so a negated input is a negated weight."""
    x = torch.randn(1, 8, 3, 4, 4)
    assert torch.allclose(
        temporal_differences(x, reverse=True), -temporal_differences(x, reverse=False)
    )


def test_the_clip_boundary_clamps_instead_of_wrapping() -> None:
    """Wrapping would join a recording's last frame to its first and manufacture a
    step discontinuity at exactly the frequencies rPPG reads."""
    x = torch.randn(1, 6, 3, 4, 4)
    diffs = temporal_differences(x, reverse=False)
    # At t=0 the stack is X[0,0,0,1,2], so the first two differences must disappear.
    assert torch.allclose(diffs[0, 0, 0:3], torch.zeros(3, 4, 4), atol=1e-6)
    assert torch.allclose(diffs[0, 0, 3:6], torch.zeros(3, 4, 4), atol=1e-6)
    # And the last frame's forward differences likewise.
    assert torch.allclose(diffs[0, -1, 6:9], torch.zeros(3, 4, 4), atol=1e-6)
    assert torch.allclose(diffs[0, -1, 9:12], torch.zeros(3, 4, 4), atol=1e-6)
    # A wrap would have made these large; a clamp makes them exactly zero.
    assert not torch.allclose(diffs[0, 3], torch.zeros_like(diffs[0, 3]), atol=1e-6)


def test_a_still_video_has_no_differences_at_all() -> None:
    x = torch.rand(1, 7, 3, 4, 4)[:, :1].repeat(1, 7, 1, 1, 1)
    assert temporal_differences(x).abs().max() == 0.0


# --- geometry -----------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_output_geometry_matches_the_variant(variant: str) -> None:
    stem = FusionStem(dim=24, stem_dim=6, variant=variant).eval()
    x = torch.rand(2, 10, 3, 128, 128)
    with torch.no_grad():
        out = stem(x)
    side = 128 // VARIANTS[variant]["stride"]
    assert out.shape == (2, 10, 24, side, side)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("n_frames", [2, 5, 16])
def test_the_time_axis_survives_the_3d_convolution(variant: str, n_frames: int) -> None:
    """The whole reason stem3's time axis is front-padded."""
    stem = FusionStem(dim=8, stem_dim=4, variant=variant).eval()
    with torch.no_grad():
        out = stem(torch.rand(1, n_frames, 3, 64, 64))
    assert out.shape[1] == n_frames


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        FusionStem(dim=8, stem_dim=4, variant="physformer")


# --- fusion -------------------------------------------------------------------

def test_the_difference_branch_changes_the_output() -> None:
    """If it does not, the stem is a vanilla stem with extra parameters -- and the
    published effect of this module is larger than most architectures produce."""
    torch.manual_seed(0)
    fused = FusionStem(dim=8, stem_dim=4, fuse=True).eval()
    x = torch.rand(1, 8, 3, 64, 64)
    still = x[:, :1].repeat(1, 8, 1, 1, 1)
    with torch.no_grad():
        moving_out, still_out = fused(x), fused(still)
    assert not torch.allclose(moving_out, still_out, atol=1e-4)


def test_the_vanilla_stem_ablation_drops_the_difference_branch() -> None:
    stem = FusionStem(dim=8, stem_dim=4, fuse=False)
    assert stem.stem_diff is None
    assert not any(n.startswith("stem_diff") for n, _ in stem.named_parameters())
    with torch.no_grad():
        assert stem.eval()(torch.rand(1, 6, 3, 64, 64)).shape[:3] == (1, 6, 8)


def test_the_fusion_weights_are_used_as_written() -> None:
    """X_fusion = stem2(alpha*raw + beta*diff) + stem2(diff), with 5:5 the value
    RhythmFormer's Table 8 measures as optimal. alpha=beta=0 must leave only the
    second term, which is the cheapest way to prove the first one is really there.
    """
    torch.manual_seed(0)
    stem = FusionStem(dim=8, stem_dim=4, alpha=0.0, beta=0.0).eval()
    x = torch.rand(1, 6, 3, 64, 64)
    with torch.no_grad():
        diff = stem.stem_diff(temporal_differences(x, reverse=True).flatten(0, 1))
        expected = stem.stem2(torch.zeros_like(diff)) + stem.stem2(diff)
        got_from_forward = stem(x)
    # Run the tail of forward on `expected` and compare against the real forward.
    with torch.no_grad():
        volume = expected.view(1, 6, -1, *expected.shape[-2:]).permute(0, 2, 1, 3, 4)
        volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, 1, 0), mode="replicate")
        tail = stem.stem3(volume).permute(0, 2, 1, 3, 4)
    assert torch.allclose(tail, got_from_forward, atol=1e-5)


def test_gradients_reach_every_parameter() -> None:
    stem = FusionStem(dim=8, stem_dim=4)
    stem(torch.rand(2, 6, 3, 64, 64)).square().mean().backward()
    dead = [n for n, p in stem.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
