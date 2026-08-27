"""Physiology-aware temporal-spectral FFN. CFMamba Eqs. 13-18, Fig. 4 stage 2.

Where CS-FFN mixes channels, this stage works along time: transform to the
temporal spectrum, apply the learnable cardiac band mask, mix, transform back.
It is the half of the DF-FFN that carries the physiological prior.

`mode` resolves the one place the paper contradicts itself. Eq. 17's prose says the
complex weight is "applied to each channel individually, with weights shared across
all N channels", which reads as a matrix over the *frequency* axis -- shape (T, T),
fixed to one clip length. But the same sentence points at Eqs. 10-11 for the
operation, and those define W in C^(N x N) and B in C^N: a matrix over the
*channel* axis. Two other sources break the tie:

**FreTS (Yi et al., NeurIPS 36), cited by CFMamba as [33]**, is where the design
comes from, and its frequency temporal learner transforms along time while applying
W in C^(d x d) to a *different* axis, shared across the N channels (its Eq. 4).
The transformed axis is left as a batch dimension. CFMamba folded FreTS's three
axes into two, which is how the prose came to read ambiguously, but the shape it
inherits is the channel one.

**CFMamba Table 4** measured cost on a 900-frame clip while Section 4.1 trained on
160-frame segments. A (T, T) weight cannot do both, so the literal-prose reading is
inconsistent with the paper's own experiment.

"channel" is therefore the default. "full" is kept as the literal-prose alternative
and shown in tests/test_budget.py to be ruled out; "diagonal" and "none" are
cheaper variants. The published budget decides between the survivors.
"""

from __future__ import annotations

import torch
from torch import nn

from .band_mask import GaussianBandMask
from .complex_linear import ComplexLinear, complex_activation

MODES = ("channel", "full", "diagonal", "none")


class PhysiologyTemporalSpectralFFN(nn.Module):
    """(B, T, N) -> (B, T, N). Band-masked complex mixing along the time axis.

      "channel"   (N, N) complex matrix on the channel axis, shared across every
                  temporal frequency bin. The FreTS-consistent reading, and the one
                  Eq. 17's own reference to Eqs. 10-11 implies. Length-agnostic.
      "full"      (T, T) complex matrix, the literal reading of Eq. 17's prose.
                  Couples frequencies, and pins the model to one clip length.
      "diagonal"  one complex gain per frequency, held on a fixed grid in Hz and
                  interpolated to whatever T arrives. Length-agnostic and far
                  cheaper; still a per-frequency linear map, just without the
                  cross-frequency coupling.
      "none"      Eq. 16 alone -- the Gaussian mask with no projection after it.
    """

    def __init__(
        self,
        hidden: int,
        fps: float = 30.0,
        mode: str = "channel",
        n_frames: int = 160,
        n_bins: int = 64,
        activation: str | None = "gelu",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        self.hidden = hidden
        self.fps = fps
        self.mode = mode
        self.activation = activation
        self.n_frames = n_frames
        self.n_bins = n_bins
        self.mask = GaussianBandMask(fps=fps)

        if mode == "channel":
            self.linear = ComplexLinear(hidden, hidden)
        elif mode == "full":
            self.linear = ComplexLinear(n_frames, n_frames)
        elif mode == "diagonal":
            # Held in Hz, not in bin indices, for the reason in band_mask.py: a
            # grid indexed by bin silently means a different filter at every T.
            # Initialised to unit gain and zero phase, so the stage starts as the
            # mask alone and has to earn any deviation from it.
            self.gain_re = nn.Parameter(torch.ones(n_bins))
            self.gain_im = nn.Parameter(torch.zeros(n_bins))

    def frequency_gain(self, freqs: torch.Tensor) -> torch.Tensor:
        """Interpolate the learned complex gain onto `freqs`, given in Hz.

        The gain magnitude is a function of |f|, because a real signal's spectrum
        is conjugate symmetric and both halves must be filtered alike. Its *phase*
        is not: a real linear filter satisfies H(-f) = conj(H(f)), so the real part
        is even in f and the imaginary part is odd.

        That sign is not cosmetic. With an even imaginary part the filtered
        spectrum is no longer conjugate symmetric, its entire imaginary-gain
        contribution lands in the imaginary half of the inverse transform, and
        `.real` throws every bit of it away -- so `gain_im` receives exactly zero
        gradient and is dead for the whole run. Nothing raises; the tensor has the
        right shape and the loss still falls. It was caught by asserting that a
        gradient reaches every parameter, which is the only way a bug shaped like
        this ever surfaces.
        """
        grid = torch.linspace(
            0.0, self.fps / 2.0, self.n_bins, device=freqs.device, dtype=freqs.dtype
        )
        position = torch.abs(freqs).clamp(grid[0], grid[-1])
        # Linear interpolation, not pooling: this samples a smooth gain curve
        # rather than downsampling a spectrum, so there is no narrow peak to miss.
        index = (position - grid[0]) / (grid[1] - grid[0])
        low = index.floor().clamp(0, self.n_bins - 1).long()
        high = (low + 1).clamp(0, self.n_bins - 1)
        frac = (index - low.to(index.dtype)).clamp(0.0, 1.0)
        real = torch.lerp(self.gain_re.to(freqs.dtype)[low],
                          self.gain_re.to(freqs.dtype)[high], frac)
        imag = torch.lerp(self.gain_im.to(freqs.dtype)[low],
                          self.gain_im.to(freqs.dtype)[high], frac)

        # sign(f) makes the phase odd. Two bins are their own conjugate partner and
        # must therefore stay purely real: DC, which sign() already sends to zero,
        # and -- for even T -- Nyquist, which it does not. Leaving Nyquist complex
        # puts a sliver of the filter where `.real` discards it. It sits at fps/2 =
        # 15 Hz, six times any heart rate and already crushed by the mask, so it
        # changes no number today; it is fixed because a filter that is *nearly*
        # real stops being harmless the moment someone changes fps.
        self_conjugate = freqs.abs() >= self.fps / 2.0 - 1e-6
        phase_sign = torch.where(
            self_conjugate, torch.zeros_like(freqs), torch.sign(freqs)
        )
        return torch.complex(real, imag * phase_sign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = x.shape[1]
        # Eq. 13.
        spectrum = torch.fft.fft(x, dim=1)
        # Eq. 16: element-wise spectral gating by the physiological band.
        spectrum = spectrum * self.mask(n_frames, x.device, x.dtype).view(1, -1, 1)

        if self.mode == "channel":
            # Eq. 17 with the Eq. 10-11 shapes: mixing runs over channels, and the
            # temporal frequency axis rides along as a batch dimension. The mask
            # above is what makes this stage temporal; the projection is what makes
            # it a projection.
            spectrum = complex_activation(self.linear(spectrum), self.activation)
        elif self.mode == "full":
            if n_frames != self.n_frames:
                raise ValueError(
                    f"mode='full' was built for {self.n_frames} frames and cannot "
                    f"accept {n_frames}; use mode='diagonal' for variable-length clips"
                )
            # Eq. 17, per channel over the frequency axis.
            spectrum = complex_activation(
                self.linear(spectrum.transpose(1, 2)), self.activation
            ).transpose(1, 2)
        elif self.mode == "diagonal":
            freqs = torch.fft.fftfreq(
                n_frames, d=1.0 / self.fps, device=x.device, dtype=x.dtype
            )
            spectrum = spectrum * self.frequency_gain(freqs).view(1, -1, 1)
        # Eq. 18.
        return torch.fft.ifft(spectrum, dim=1).real
