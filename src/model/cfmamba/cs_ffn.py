"""Channel-spectral feed-forward network. CFMamba Eqs. 9-12, Fig. 4 stage 1.

After PGA reduces space into channels, each channel is a pure time series and
the *set* of channels is a spatial code. Mixing them pointwise in the time domain
treats each instant independently. Mixing them in the channel-frequency domain
instead asks which combinations of channels co-vary, which is what distinguishes a
pulse spread coherently across skin regions from noise that is not.

This stage is RhythmMamba's frequency-domain feed-forward unchanged -- CFMamba
keeps it and adds PTS-FFN after it. Its feasibility for spectral MLPs is the
result both papers cite from FreTS (Yi et al., NeurIPS 36).
"""

from __future__ import annotations

import torch
from torch import nn

from .complex_linear import ComplexLinear, complex_activation, complex_activation_parts
from .dft import SPECTRAL, FixedDFT


class ChannelSpectralFFN(nn.Module):
    """(B, T, N) -> (B, T, N). Complex mixing along the channel axis.

    The transform runs over channels, not time, so the same weights apply at every
    timestamp -- "the learnable parameters are shared across all temporal
    positions" (after Eq. 11). That is what keeps this stage length-agnostic.
    """

    def __init__(
        self, hidden: int, activation: str | None = "gelu", spectral: str = "fft"
    ) -> None:
        super().__init__()
        if spectral not in SPECTRAL:
            raise ValueError(f"unknown spectral {spectral!r}, expected one of {SPECTRAL}")
        self.hidden = hidden
        self.activation = activation
        self.spectral = spectral
        self.linear = ComplexLinear(hidden, hidden)
        # "matmul" replaces the transform pair with two constant matrices, for the
        # runtimes that have no FFT. The transform axis here is `hidden`, which was
        # already fixed at construction, so this costs nothing in generality --
        # see dft.py. `CFMambaPhys` collapses these onto one instance per length
        # afterwards (`share_fixed_dfts`): the tables are identical in every block.
        self.dft = FixedDFT(hidden) if spectral == "matmul" else None

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, activation={self.activation}, spectral={self.spectral}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dft is not None:
            return self._forward_matmul(x)
        # Eq. 9: full FFT over the channel axis, as written. rfft would halve the
        # weight matrix, but the paper specifies W in C^(N x N), and the parameter
        # budget is what has to decide between the two readings.
        spectrum = torch.fft.fft(x, dim=-1)
        # FreMLP = complex linear then an activation on each component (FreTS
        # Eq. 7). Section 3.3 omits the activation; see complex_linear.py.
        spectrum = complex_activation(self.linear(spectrum), self.activation)
        # Eq. 12. A complex linear does not preserve conjugate symmetry, so the
        # inverse transform is complex. The imaginary leftover is the part of the
        # learned map that no real-valued signal can express; discarding it is what
        # makes this a real-to-real layer, and is what FreTS does.
        return torch.fft.ifft(spectrum, dim=-1).real

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The same three steps with the parts carried separately. No complex tensors.

        The channel axis is already last, so no transpose is needed.
        """
        real, imag = self.dft.analyse(x)
        real, imag = self.linear.forward_parts(real, imag)
        real, imag = complex_activation_parts(real, imag, self.activation)
        return self.dft.synthesise_real(real, imag)
