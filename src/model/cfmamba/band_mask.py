"""Learnable Gaussian cardiac band mask. CFMamba Eqs. 14-15.

This is the module that makes the DF-FFN "physiology-aware". A pulse concentrates
its energy in a narrow band and everything else -- motion, illumination drift,
sensor noise -- spreads across the spectrum, so a filter that keeps one band and
attenuates the rest is a prior worth building in rather than learning from
scratch. What the network chooses is *where* inside the human range that band sits
and how wide it is; it cannot choose to leave the range.

Two details the equations depend on and neither paper spells out:

**Frequencies are Hz, never bin indices.** FFT bin k means k*fps/T, so a mask
built from bin numbers means a different filter at every clip length -- and
nothing raises, because the tensor keeps its shape. This is the same bug that
shipped silently in the model this one replaces, where a fixed bin list meant
45-202 bpm at T=160 and 24-108 bpm at T=300.

**Eq. 15 is two Gaussians, not one.** A real signal's spectrum is conjugate
symmetric, so the cardiac peak appears at +f_c and again at -f_c. A single-lobe
mask would keep one and discard the other, halving the energy and leaving the
inverse transform complex. That is also why the full `fft` is used downstream
rather than `rfft`.
"""

from __future__ import annotations

import torch
from torch import nn

# CFMamba Eq. 14. 0.75-2.5 Hz is 45-150 bpm, the range both papers cite as
# physiologically plausible; the bandwidth range keeps the filter from collapsing
# onto a single bin or opening up into an all-pass.
FC_MIN_HZ, FC_MAX_HZ = 0.75, 2.5
BW_MIN_HZ, BW_MAX_HZ = 0.2, 1.0


class GaussianBandMask(nn.Module):
    """A soft, differentiable, symmetric band-pass over the temporal spectrum.

    Both parameters are unconstrained scalars squashed into a physiological range,
    so no setting of the weights can produce a filter centred outside 0.75-2.5 Hz.
    The constraint is the point: it is what makes this a prior rather than one more
    free parameter that can be trained onto a motion artefact.
    """

    def __init__(self, fps: float = 30.0) -> None:
        super().__init__()
        self.fps = fps
        # Zero-initialised, so training starts at the midpoint of each range:
        # f_c = 1.625 Hz (97.5 bpm), b_w = 0.6 Hz. Roughly the centre of the adult
        # distribution, which is a better starting point than either edge.
        self.theta_fc = nn.Parameter(torch.zeros(()))
        self.theta_bw = nn.Parameter(torch.zeros(()))

    @property
    def centre_hz(self) -> torch.Tensor:
        return FC_MIN_HZ + torch.sigmoid(self.theta_fc) * (FC_MAX_HZ - FC_MIN_HZ)

    @property
    def bandwidth_hz(self) -> torch.Tensor:
        return BW_MIN_HZ + torch.sigmoid(self.theta_bw) * (BW_MAX_HZ - BW_MIN_HZ)

    def forward(
        self, n_frames: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """(T,) real mask aligned to `torch.fft.fft`'s frequency ordering."""
        freqs = torch.fft.fftfreq(n_frames, d=1.0 / self.fps, device=device, dtype=dtype)
        centre = self.centre_hz.to(device=device, dtype=dtype)
        bandwidth = self.bandwidth_hz.to(device=device, dtype=dtype)
        two_var = 2.0 * bandwidth * bandwidth
        return (
            torch.exp(-((freqs - centre) ** 2) / two_var)
            + torch.exp(-((freqs + centre) ** 2) / two_var)
        )

    def extra_repr(self) -> str:
        return (f"fps={self.fps}, f_c in [{FC_MIN_HZ}, {FC_MAX_HZ}] Hz, "
                f"b_w in [{BW_MIN_HZ}, {BW_MAX_HZ}] Hz")
