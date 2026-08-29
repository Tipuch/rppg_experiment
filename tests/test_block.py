"""One backbone layer, CFMamba Sections 3.2-3.3 over RhythmMamba Fig. 3.

The thing worth pinning here is *where* CAM sits. Section 3.2 states it "operates on
the temporal representations produced by the state space model", so it modulates
the Mamba branch before that branch joins the residual stream. Applying it after
the addition would let it rescale the skip connection as well, which Eq. 8 does not
say and which would compound the attenuation across L layers -- a bug that shows up
as a model that trains but never gets anywhere.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.cfmamba.block import ChannelAdaptiveMambaBlock
from src.model.cfmamba.cam import ChannelAdaptiveModulation
from src.model.cfmamba.df_ffn import DualFrequencyFFN
from src.model.cfmamba.vanilla_ffn import VanillaFFN

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3's scan kernel is CUDA-only"
)


def _block(dim: int = 16, **kwargs) -> ChannelAdaptiveMambaBlock:
    return ChannelAdaptiveMambaBlock(dim, VanillaFFN(dim, dim * 2), **kwargs)


def test_cam_modulates_the_mamba_branch_not_the_residual_stream() -> None:
    """Structural, so it keeps without CUDA: forward must read
    norm1(x + cam(mamba(x))), never cam(norm1(x + mamba(x)))."""
    import inspect

    source = inspect.getsource(ChannelAdaptiveMambaBlock.forward)
    assert "mixed = self.cam(mixed)" in source
    assert "self.norm1(x + mixed)" in source
    assert "self.cam(x" not in source


def test_the_ffn_is_injected_so_ablations_swap_one_module() -> None:
    dim = 16
    assert isinstance(_block(dim).ffn, VanillaFFN)
    assert isinstance(
        ChannelAdaptiveMambaBlock(dim, DualFrequencyFFN(dim, dim * 2)).ffn,
        DualFrequencyFFN,
    )
    assert isinstance(ChannelAdaptiveMambaBlock(dim, nn.Identity()).ffn, nn.Identity)


def test_cam_can_be_removed_for_the_ablation() -> None:
    assert _block(use_cam=False).cam is None
    assert isinstance(_block(use_cam=True).cam, ChannelAdaptiveModulation)


def test_both_residual_branches_are_post_normed() -> None:
    """RhythmMamba Fig. 3 draws Add & Norm twice, after each sublayer."""
    block = _block()
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)


@cuda
def test_shape_is_preserved() -> None:
    block = _block(32).cuda()
    x = torch.randn(2, 64, 32, device="cuda")
    assert block(x).shape == x.shape


@cuda
def test_removing_cam_changes_the_output() -> None:
    """If it does not, the ablation would come out flat for the wrong reason."""
    torch.manual_seed(0)
    with_cam = _block(16, use_cam=True).cuda().eval()
    without = _block(16, use_cam=False).cuda().eval()
    without.load_state_dict(
        {k: v for k, v in with_cam.state_dict().items() if not k.startswith("cam.")}
    )
    x = torch.randn(1, 32, 16, device="cuda")
    with torch.no_grad():
        assert not torch.allclose(with_cam(x), without(x), atol=1e-3)


@cuda
def test_the_output_is_layer_normalised() -> None:
    block = _block(32).cuda().eval()
    with torch.no_grad():
        out = block(torch.randn(2, 48, 32, device="cuda"))
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert (out.std(dim=-1, unbiased=False) - 1.0).abs().max() < 0.05


@cuda
def test_gradients_reach_every_parameter() -> None:
    block = ChannelAdaptiveMambaBlock(16, DualFrequencyFFN(16, 32, n_frames=32)).cuda()
    block(torch.randn(2, 32, 16, device="cuda")).square().mean().backward()
    dead = [n for n, p in block.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
