"""The published cost budget, used as the arbiter for what the papers omit.

CFMamba states two numbers and no widths: **0.91M parameters** and **80.82M per
frame at 128x128** (Table 4). Between them they are two equations in several
unknowns -- dim, depth, stem width, FFN expansion, CAM's hidden width -- so they
cannot pin the architecture alone. They can and do rule configurations out, and
that is what this file is for. It runs before training, so a change that moves
the model away from the published cost fails here rather than three GPU-hours
later.

**The unit is MACs, not FLOPs.** Table 4 is named FLOPs, but its entries for
PhysNet (438.24), TS-CAN (744.45) and DeepPhys (744.45) are identical to the MACs
column of RhythmMamba's Table 5. The table was brought over and relabelled. Torch's
FlopCounterMode reports 2 per MAC, hence the division below; reading the label
literally would put the target at half the real figure and shrink the model into
something that cannot fit the data.

**Table 4 also resolves Eq. 17.** The cost was measured on a 900-frame clip while
training used 160-frame segments (Section 4.1), so the model must accept both. A
(T, T) complex weight over the frequency axis -- the literal reading of Eq. 17's
text -- is fixed to one length and could not have produced that measurement. The
channel-axis reading, which is both what Eq. 17's own reference to Eqs. 10-11
implies and what FreTS (CFMamba [33]) actually does, has no such problem.

Two module shapes were wrong before FreTS and CMamba were read, both in the
direction of too few parameters:

  - PTS-FFN's projection is (N, N) over channels, not a (T, T) frequency matrix
    and not a per-bin gain (pts_ffn.py).
  - CAM's hidden layer is set by an *expansion* rate, CMamba's term, not by a
    squeeze-and-excitation bottleneck ratio (cam.py).

Correcting both moved the fit from -2.9%/+2.1% to -0.6%/-2.0% under the Mamba-1
scan. The current configuration, on Mamba-3, reproduces the parameter count to
+2.5% and the MAC count to -2.0%.

The fit does not establish a unique configuration: several land inside tolerance.
What held across every configuration search, before and after the corrections, is
depth=4 and a stem width of 16.

**Where the budget cannot help at all.** CAM is the only module in the model that
is parameter-heavy and compute-free -- Eq. 6 pools over T first, so its two MLPs
run once per clip on a (B, C) vector rather than per frame. Measured, MACs per
frame are identical to the second decimal at every expansion rate from 0.25 to 4.0,
while parameters move from 0.879M to 1.265M. So the FLOP budget cannot constrain
that control and the parameter budget fully determines it, which is why it is the one
place a textual argument was allowed to win.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba import CFMambaPhys

PUBLISHED_PARAMS = 0.91e6
PUBLISHED_MACS_PER_FRAME = 80.82e6
RESOLUTION = 128
# Measured at the training length. MACs per frame are length-invariant except for
# the FFTs, which grow as log T -- a few percent between 160 and the paper's 900,
# and 900 frames at 128x128 does not fit an 8 GB card.
N_FRAMES = 160          # the length the published cost was measured against
DEFAULT_N_FRAMES = 300  # the length this project trains at; see the test below

# 4%, down from the 6% the Mamba-1 scan needed. CAM's expansion rate of 1.0 is a
# textual choice costing +5.04%, and the gate had to fund it; Mamba-3 gave most of
# that back -- 0.9559M to 0.9327M, +2.49% against the published 0.91M -- by
# dropping the short conv and the x_proj and reducing A to one scalar per head.
PARAM_TOLERANCE = 0.04

# The scan is invisible to this count, and always was: FlopCounterMode decomposes
# aten operators and a fused kernel is opaque to it. Measured, MACs/frame moved
# 79.19M -> 79.15M across the swap. So this bounds the stem, PGA, CAM, the FFNs
# and the predictor, and does not cover the recurrence between them.
MACS_TOLERANCE = 0.10

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-3's scan kernel is CUDA-only"
)


def macs_per_frame(model: CFMambaPhys, n_frames: int = N_FRAMES) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    model = model.cuda().eval()
    frames = torch.rand(1, n_frames, 3, RESOLUTION, RESOLUTION, device="cuda")
    skin = torch.ones(1, RESOLUTION, RESOLUTION, device="cuda")
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(frames, skin)
    return counter.get_total_flops() / 2 / n_frames


def test_parameter_count_matches_the_paper() -> None:
    count = CFMambaPhys().parameter_count()
    error = abs(count - PUBLISHED_PARAMS) / PUBLISHED_PARAMS
    assert error <= PARAM_TOLERANCE, (
        f"{count / 1e6:.4f}M parameters against a published 0.91M "
        f"({100 * error:+.1f}%)"
    )


@cuda
def test_macs_per_frame_matches_the_paper() -> None:
    measured = macs_per_frame(CFMambaPhys())
    error = abs(measured - PUBLISHED_MACS_PER_FRAME) / PUBLISHED_MACS_PER_FRAME
    assert error <= MACS_TOLERANCE, (
        f"{measured / 1e6:.2f}M MACs/frame against a published 80.82M "
        f"({100 * error:+.1f}%)"
    )


def test_the_default_clip_length_departs_from_the_paper_deliberately() -> None:
    """This project trains on 300-frame (10 s) windows, not the paper's 160.

    A deliberate departure, pinned here so it cannot drift back or be mistaken for
    the published configuration. Section 4.1 uses 160, and RhythmFormer's Table 11
    measures 160 as the optimum (3.07 MAE against 3.53 at 80 and 3.86 at 320), so
    **numbers produced at 300 are not directly comparable with published ones**.

    What 300 buys is spectral resolution: FFT bins fall from 11.25 bpm to 6.0 bpm.
    The cost budget below is still measured at the paper's 160, because that is
    the length Table 4 was measured against.
    """
    assert CFMambaPhys().n_frames == DEFAULT_N_FRAMES
    assert 30.0 * 60 / N_FRAMES == pytest.approx(11.25)
    assert 30.0 * 60 / DEFAULT_N_FRAMES == pytest.approx(6.0)


def test_the_model_still_accepts_the_length_the_cost_was_measured_at() -> None:
    """Changing the default must not fix the model to one length.

    Table 4's cost was measured on a 900-frame clip while training used 160, so a
    module whose weights are sized by T would have been impossible all along --
    and the failure is silent, since the tensors keep their shape.
    """
    assert CFMambaPhys(n_frames=N_FRAMES).n_frames == N_FRAMES
    assert CFMambaPhys(n_frames=900).n_frames == 900


def test_the_default_pts_mode_can_accept_the_length_the_cost_was_measured_at() -> None:
    """A (T, T) complex weight could not have produced Table 4's 900-frame run."""
    model = CFMambaPhys()
    assert model.blocks[0].ffn.pts_mode == "channel"


def test_cam_expands_rather_than_squeezes() -> None:
    """CMamba (CFMamba [32]) parameterises GDD-MLP by an expansion rate and cautions
    that too small a value underfits. 1.0 is the smallest setting that is actually
    an expansion, making the hidden layer as wide as the stream.

    CMamba never prints the value it used: its A.2 gives an expansion rate of 1 for
    a *different* module (M-Mamba's linear), and the GDD-MLP one appears only in a
    rasterised figure. So this is not recoverable from the paper, and the choice is
    made on the word "expansion" rather than on a number.
    """
    model = CFMambaPhys()
    cam = model.blocks[0].cam
    assert cam.expansion == pytest.approx(1.0)
    assert cam.scale[0].out_features == model.dim


@cuda
def test_cam_costs_parameters_but_not_compute() -> None:
    """Why the FLOP budget cannot arbitrate the expansion rate: Eq. 6 pools over T
    before the MLPs, so they run once per clip rather than once per frame."""
    lean = macs_per_frame(CFMambaPhys(cam_expansion=0.25))
    rich = macs_per_frame(CFMambaPhys(cam_expansion=4.0))
    assert rich == pytest.approx(lean, rel=1e-3)
    assert (CFMambaPhys(cam_expansion=4.0).parameter_count()
            > 1.3 * CFMambaPhys(cam_expansion=0.25).parameter_count())


@cuda
def test_the_full_pts_reading_is_ruled_out_by_the_900_frame_measurement() -> None:
    """Kept as an implemented alternative, and shown here to be inconsistent with
    the paper's own cost table rather than only disfavoured."""
    model = CFMambaPhys(pts_mode="full", n_frames=160).cuda().eval()
    with pytest.raises(ValueError, match="cannot accept"), torch.no_grad():
        model(torch.rand(1, 900, 3, 8, 8, device="cuda"))


def test_the_stem_is_not_where_the_budget_goes() -> None:
    """A 7x7 stem2 at 32x32 costs more MACs than the entire backbone. The published
    budget is only reachable once that is fixed, which is what selected k2=5."""
    model = CFMambaPhys()
    stem = sum(p.numel() for p in model.stem.parameters())
    assert stem < 0.2 * model.parameter_count()


@cuda
def test_peak_memory_is_in_the_paper_s_range() -> None:
    """Table 4 reports 2.85 MB per frame, CUDA context excluded. A model an order
    of magnitude off that is not the model in the paper, whatever it scores."""
    model = CFMambaPhys().cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(torch.rand(1, N_FRAMES, 3, RESOLUTION, RESOLUTION, device="cuda"),
              torch.ones(1, RESOLUTION, RESOLUTION, device="cuda"))
    per_frame_mb = torch.cuda.max_memory_allocated() / 1e6 / N_FRAMES
    assert 0.3 < per_frame_mb < 30.0, f"{per_frame_mb:.2f} MB/frame against 2.85"
