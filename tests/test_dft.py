"""The DFT-as-matmul path, from src/model/cfmamba/dft.py.

`spectral="matmul"` exists so DF-FFN can run where there is no FFT. It has to be
the *same* layer, not an approximation of it, so every test here is a parity test
against the `torch.fft` path at float32 tolerance.

The load-compatibility tests carry as much weight as the numerical ones: the DFT
matrices are registered non-persistently precisely so a checkpoint trained on the
FFT path loads into an export-shaped model with `strict=True`. If that ever
regressed, the export would silently need `strict=False` -- which would mean it is
computing something the checkpoint did not train.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.cs_ffn import ChannelSpectralFFN
from src.model.cfmamba.df_ffn import DualFrequencyFFN
from src.model.cfmamba.dft import FixedDFT
from src.model.cfmamba.pts_ffn import PhysiologyTemporalSpectralFFN

# The shipped widths: DF-FFN's latent is 160 and a clip is 300 frames.
HIDDEN, FRAMES, DIM = 160, 300, 80


def test_analyse_matches_torch_fft_on_real_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 7, FRAMES)
    expected = torch.fft.fft(x, dim=-1)
    real, imag = FixedDFT(FRAMES).analyse(x)
    scale = float(expected.abs().max())
    assert (expected.real - real).abs().max() / scale < 1e-6
    assert (expected.imag - imag).abs().max() / scale < 1e-6


def test_analyse_matches_torch_fft_on_complex_input() -> None:
    """The four-matmul branch, which the real-input case skips."""
    torch.manual_seed(0)
    x = torch.complex(torch.randn(3, FRAMES), torch.randn(3, FRAMES))
    expected = torch.fft.fft(x, dim=-1)
    real, imag = FixedDFT(FRAMES).analyse(x.real, x.imag)
    scale = float(expected.abs().max())
    assert (expected.real - real).abs().max() / scale < 1e-6
    assert (expected.imag - imag).abs().max() / scale < 1e-6


def test_synthesise_real_inverts_the_transform() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, FRAMES)
    dft = FixedDFT(FRAMES)
    assert torch.allclose(dft.synthesise_real(*dft.analyse(x)), x, atol=1e-5)


def test_synthesise_real_matches_the_real_part_of_ifft() -> None:
    """The inverse deliberately computes no imaginary output, because both callers
    discard it. That is only sound if the real part it does compute is the right one
    for a spectrum that is *not* conjugate symmetric -- which is exactly what a
    complex linear produces."""
    torch.manual_seed(0)
    spectrum = torch.complex(torch.randn(2, FRAMES), torch.randn(2, FRAMES))
    expected = torch.fft.ifft(spectrum, dim=-1).real
    got = FixedDFT(FRAMES).synthesise_real(spectrum.real, spectrum.imag)
    assert (expected - got).abs().max() / float(expected.abs().max()) < 1e-5


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda spectral: ChannelSpectralFFN(HIDDEN, spectral=spectral),
            id="cs_ffn",
        ),
        pytest.param(
            lambda spectral: PhysiologyTemporalSpectralFFN(
                HIDDEN, n_frames=FRAMES, mode="channel", spectral=spectral),
            id="pts_ffn_channel",
        ),
        pytest.param(
            lambda spectral: PhysiologyTemporalSpectralFFN(
                HIDDEN, n_frames=FRAMES, mode="diagonal", spectral=spectral),
            id="pts_ffn_diagonal",
        ),
        pytest.param(
            lambda spectral: PhysiologyTemporalSpectralFFN(
                HIDDEN, n_frames=FRAMES, mode="none", spectral=spectral),
            id="pts_ffn_none",
        ),
    ],
)
def test_the_two_spectral_paths_agree(build) -> None:
    torch.manual_seed(0)
    fft_module = build("fft").eval()
    matmul_module = build("matmul").eval()
    matmul_module.load_state_dict(fft_module.state_dict(), strict=True)
    x = torch.randn(2, FRAMES, HIDDEN)
    with torch.no_grad():
        expected, got = fft_module(x), matmul_module(x)
    assert (expected - got).abs().max() / float(expected.abs().max()) < 1e-5


def test_df_ffn_agrees_end_to_end() -> None:
    torch.manual_seed(0)
    fft_module = DualFrequencyFFN(DIM, HIDDEN, n_frames=FRAMES, spectral="fft").eval()
    matmul_module = DualFrequencyFFN(
        DIM, HIDDEN, n_frames=FRAMES, spectral="matmul").eval()
    matmul_module.load_state_dict(fft_module.state_dict(), strict=True)
    x = torch.randn(2, FRAMES, DIM)
    with torch.no_grad():
        expected, got = fft_module(x), matmul_module(x)
    assert (expected - got).abs().max() / float(expected.abs().max()) < 1e-5


def test_the_matrices_stay_out_of_the_state_dict() -> None:
    """Both directions, because a checkpoint has to move either way."""
    fft_module = DualFrequencyFFN(DIM, HIDDEN, n_frames=FRAMES, spectral="fft").eval()
    matmul_module = DualFrequencyFFN(
        DIM, HIDDEN, n_frames=FRAMES, spectral="matmul").eval()
    assert set(fft_module.state_dict()) == set(matmul_module.state_dict())
    matmul_module.load_state_dict(fft_module.state_dict(), strict=True)
    fft_module.load_state_dict(matmul_module.state_dict(), strict=True)


def test_the_frozen_band_is_the_same_filter_as_the_computed_one() -> None:
    """`from_freqs` on the frozen grid must equal `forward`'s own grid at that fps.

    The frozen grid is what fixes the export to one frame rate. This asserts the
    freezing itself introduces no error -- so any fps sensitivity that shows up
    later is the frame rate, not a bug here.
    """
    module = PhysiologyTemporalSpectralFFN(
        HIDDEN, n_frames=FRAMES, mode="channel", spectral="matmul", fps=30.0).eval()
    with torch.no_grad():
        frozen = module.mask.from_freqs(module.frozen_freqs)
        computed = module.mask(FRAMES, torch.device("cpu"), torch.float32)
    assert torch.allclose(frozen, computed, atol=1e-7)


def test_a_length_mismatch_is_refused() -> None:
    """The transform length is baked in, so a wrong T must raise rather than
    silently transform the wrong axis length."""
    module = PhysiologyTemporalSpectralFFN(
        HIDDEN, n_frames=FRAMES, mode="channel", spectral="matmul").eval()
    with pytest.raises(ValueError, match="was built for 300 frames"):
        module(torch.randn(1, 160, HIDDEN))


def test_mode_full_is_refused_on_the_matmul_path() -> None:
    with pytest.raises(ValueError, match="does not support mode='full'"):
        PhysiologyTemporalSpectralFFN(
            HIDDEN, n_frames=FRAMES, mode="full", spectral="matmul")


def test_the_dft_tables_are_shared_across_blocks() -> None:
    """One instance per length, not one per block.

    Each block builds its own and they are identical: at depth 4 the duplicates
    come to 3.70 MB of host memory, as much as the whole model's parameters. The
    exported file is unaffected -- see `share_fixed_dfts`.
    """
    from src.model.cfmamba.model import CFMambaPhys

    # hidden is 2 * dim, so 32 and 48 are two different transform lengths.
    model = CFMambaPhys(dim=16, depth=3, n_frames=48, spectral="matmul")
    channel = {id(block.ffn.cs.dft) for block in model.blocks}
    temporal = {id(block.ffn.pts.dft) for block in model.blocks}
    assert len(channel) == 1
    assert len(temporal) == 1
    assert channel != temporal
    assert model.blocks[0].ffn.cs.dft.length == 32
    assert model.blocks[0].ffn.pts.dft.length == 48
    # The point of the sharing: one pair of tables in memory, not three.
    tables = {id(b) for b in model.buffers() if b.dim() == 2}
    assert len(tables) == 4


def test_sharing_leaves_the_state_dict_alone() -> None:
    """The whole model, not just one FFN: sharing must not move a single key."""
    from src.model.cfmamba.model import CFMambaPhys

    fft_model = CFMambaPhys(dim=16, depth=3, n_frames=48, spectral="fft")
    matmul_model = CFMambaPhys(dim=16, depth=3, n_frames=48, spectral="matmul")
    assert set(fft_model.state_dict()) == set(matmul_model.state_dict())
    matmul_model.load_state_dict(fft_model.state_dict(), strict=True)


def test_an_unknown_spectral_is_refused_on_every_ffn() -> None:
    """`spectral` reaches the FFN only when ffn="df", so the model checks it too.

    Without the check at that level a typo on a vanilla-FFN ablation is accepted
    in silence and exports the FFT path.
    """
    from src.model.cfmamba.model import CFMambaPhys

    with pytest.raises(ValueError, match="unknown spectral"):
        CFMambaPhys(dim=16, depth=1, n_frames=32, ffn="vanilla", spectral="matmull")
    with pytest.raises(ValueError, match="unknown spectral"):
        ChannelSpectralFFN(HIDDEN, spectral="matmull")
