"""Channel-spectral feed-forward network. CFMamba Eqs. 9-12, Fig. 4 stage 1.

After PGA collapses space into channels, each channel is a pure time series and
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

from .complex_linear import ComplexLinear, complex_activation


class ChannelSpectralFFN(nn.Module):
    """(B, T, N) -> (B, T, N). Complex mixing along the channel axis.

    The transform runs over channels, not time, so the same weights apply at every
    timestamp -- "the learnable parameters are shared across all temporal
    positions" (after Eq. 11). That is what keeps this stage length-agnostic.
    """

    def __init__(self, hidden: int, activation: str | None = "gelu") -> None:
        super().__init__()
        self.hidden = hidden
        self.activation = activation
        self.linear = ComplexLinear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 9: full FFT over the channel axis, as written. rfft would halve the
        # weight matrix, but the paper specifies W in C^(N x N), and the parameter
        # budget is what has to decide between the two readings.
        spectrum = torch.fft.fft(x, dim=-1)
        # FreMLP = complex linear then an activation on each component (FreTS
        # Eq. 7). Section 3.3 omits the activation; see complex_linear.py.
        spectrum = complex_activation(self.linear(spectrum), self.activation)
        # Eq. 12. A complex linear does not preserve conjugate symmetry, so the
        # inverse transform is complex. The imaginary residue is the part of the
        # learned map that no real-valued signal can express; discarding it is what
        # makes this a real-to-real layer, and is what FreTS does.
        return torch.fft.ifft(spectrum, dim=-1).real
