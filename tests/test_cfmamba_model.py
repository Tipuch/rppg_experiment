"""The assembled model, CFMamba Fig. 2.

What only the assembly can get wrong: the order of the stages, T remaining from
input to output, and whether each ablation flag actually reaches anything.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.fusion_stem import FusionStem
from src.model.cfmamba.model import CFMambaPhys
from src.model.cfmamba.pga import PhysiologyGuidedAttention

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3's scan kernel is CUDA-only"
)


def _small(**kwargs) -> CFMambaPhys:
    return CFMambaPhys(dim=16, depth=1, n_frames=32, **kwargs)


# --- structure, no CUDA needed ------------------------------------------------

def test_the_stages_are_in_the_paper_s_order() -> None:
    model = _small()
    assert isinstance(model.stem, FusionStem)
    assert isinstance(model.pga, PhysiologyGuidedAttention)
    assert len(model.blocks) == 1


def test_every_ablation_flag_reaches_something() -> None:
    """A flag that is accepted and ignored is worse than no flag: the ablation
    comes back flat and reads as evidence the module does not matter."""
    assert _small(use_pga=False).pga is None
    assert _small(use_cam=False).blocks[0].cam is None
    assert _small(fuse_stem=False).stem.stem_diff is None
    assert type(_small(ffn="vanilla").blocks[0].ffn).__name__ == "VanillaFFN"
    assert type(_small(ffn="none").blocks[0].ffn).__name__ == "Identity"


def test_an_unknown_ffn_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown ffn"):
        _small(ffn="spectral")


def test_the_ffn_latent_is_wider_than_the_stream_by_default() -> None:
    model = CFMambaPhys(dim=96, depth=1)
    assert model.ffn_hidden > model.dim


def test_describe_accounts_for_every_parameter() -> None:
    """The budget is read off this, so it must not lose a module."""
    model = _small()
    total = sum(
        sum(p.numel() for p in m.parameters())
        for m in (model.stem, model.blocks, model.predictor)
    )
    assert total == model.parameter_count()


# --- forward ------------------------------------------------------------------

@cuda
def test_a_clip_becomes_one_value_per_frame() -> None:
    model = _small().cuda().eval()
    frames = torch.rand(2, 32, 3, 128, 128, device="cuda")
    skin = torch.ones(2, 128, 128, device="cuda")
    with torch.no_grad():
        out = model(frames, skin)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


@cuda
@pytest.mark.parametrize("n_frames", [32, 96, 160])
def test_the_frame_count_survives_end_to_end(n_frames: int) -> None:
    """One prediction per input frame, or the waveform loss compares a prediction
    to a target it is not aligned with."""
    model = CFMambaPhys(dim=16, depth=1, n_frames=n_frames).cuda().eval()
    with torch.no_grad():
        out = model(torch.rand(1, n_frames, 3, 128, 128, device="cuda"))
    assert out.shape == (1, n_frames)


@cuda
def test_diagonal_mode_generalises_to_an_unseen_clip_length() -> None:
    """RhythmMamba's arbitrary-length property (Supplementary C) survives here."""
    model = CFMambaPhys(dim=16, depth=1, n_frames=64, pts_mode="diagonal").cuda().eval()
    with torch.no_grad():
        assert model(torch.rand(1, 150, 3, 128, 128, device="cuda")).shape == (1, 150)


@cuda
def test_the_skin_mask_changes_the_prediction() -> None:
    """PGA is wired to the mask, not only given it."""
    torch.manual_seed(0)
    model = _small().cuda().eval()
    frames = torch.rand(1, 32, 3, 128, 128, device="cuda")
    left = torch.zeros(1, 128, 128, device="cuda")
    left[:, 40:80, 8:48] = 1.0
    right = torch.zeros(1, 128, 128, device="cuda")
    right[:, 40:80, 80:120] = 1.0
    with torch.no_grad():
        assert not torch.allclose(model(frames, left), model(frames, right), atol=1e-4)


@cuda
def test_a_still_clip_and_a_moving_clip_do_not_predict_the_same_thing() -> None:
    """The stem's difference branch has to reach the output."""
    torch.manual_seed(0)
    model = _small().cuda().eval()
    frames = torch.rand(1, 32, 3, 128, 128, device="cuda")
    still = frames[:, :1].repeat(1, 32, 1, 1, 1)
    with torch.no_grad():
        assert not torch.allclose(model(frames), model(still), atol=1e-4)


@cuda
def test_backward_is_finite_under_bf16_autocast() -> None:
    model = _small().cuda()
    frames = torch.rand(1, 32, 3, 128, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(frames).square().mean()
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
