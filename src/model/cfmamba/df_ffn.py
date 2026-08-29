"""Physiology-aware dual-frequency feed-forward network. CFMamba Section 3.3.

A conventional FFN mixes channels pointwise in the time domain, which cannot see a
periodic structure spread over a whole clip. A pulse is periodic, so the DF-FFN
does its mixing in the frequency domain instead, in the two stages Fig. 4 names:

  CS-FFN   cross-channel interaction in the channel-frequency domain (cs_ffn.py)
  PTS-FFN  band-masked temporal-frequency mixing, the physiological prior
           (pts_ffn.py, band_mask.py)

This file is only the wrapper: expand C -> N, run the two stages, project back.
CFMamba Table 5 measures what each part is worth on UBFC-rPPG -- 0.36 MAE with the
whole thing, 0.45 with no frequency processing at all, and 0.59 with a vanilla
time-domain FFN, which is the paper's argument that unweighted channel mixing
fights the periodic structure rather than only failing to use it.
"""

from __future__ import annotations

import torch
from torch import nn

from .cs_ffn import ChannelSpectralFFN
from .pts_ffn import PhysiologyTemporalSpectralFFN


class DualFrequencyFFN(nn.Module):
    """(B, T, C) -> (B, T, C) through an N-wide spectral latent, N > C."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        fps: float = 30.0,
        pts_mode: str = "channel",
        n_frames: int = 160,
        pts_bins: int = 64,
        activation: str | None = "gelu",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.pts_mode = pts_mode
        # Section 3.3: "we first apply a linear expansion to project it into a
        # higher-dimensional latent space", N > C, before any spectral work. The
        # expansion is what gives the spectral stages room to separate the
        # physiological component from the noise sharing its channels.
        self.proj_in = nn.Linear(dim, hidden)
        self.activation = activation
        self.cs = ChannelSpectralFFN(hidden, activation=activation)
        self.pts = PhysiologyTemporalSpectralFFN(
            hidden, fps=fps, mode=pts_mode, n_frames=n_frames, n_bins=pts_bins,
            activation=activation,
        )
        self.proj_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj_in(x)
        # FFT has no autocast kernel, and complex arithmetic in bf16 forfeits far more
        # than the memory is worth on a 0.9M-parameter model, so the whole spectral
        # section runs in float32 whatever the surrounding autocast context.
        with torch.autocast(x.device.type, enabled=False):
            spectral = self.pts(self.cs(projected.float()))
        # No activation here. Section 3.3 puts none anywhere, and the one this
        # module needs now resides inside the two complex linears, where FreTS Eq. 7
        # puts it. An extra GELU at this point was an earlier invention.
        return self.proj_out(spectral.to(x.dtype))
