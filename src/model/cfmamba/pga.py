"""Physiology-guided attention. CFMamba Section 3.1, Eqs. 1-5.

RhythmMamba flattens each frame's spatial map into channels, which is what lets
Mamba treat the clip as a pure time series -- but a plain average over space
weights the background exactly as heavily as the cheeks. Its own answer was a
learned sigmoid gate. CFMamba's objection is that a purely data-driven gate has
nothing anchoring it to anatomy, so under motion or a lighting change it strays
onto whatever is most visually salient: hair, an edge, the wall (Fig. 3a).

PGA multiplies two maps before pooling:

  M_prior  a Gaussian centred on the skin region. Parameter-free, so it cannot be
           trained away, and smooth, so it does not cut a hard boundary through
           the gradient.
  A_feat   a per-channel divisive gate, which lets different channels specialise
           on different sub-regions.

Removing it costs the paper 0.36 -> 0.50 MAE on UBFC-rPPG and 4.03 -> 5.72 on
VIPL-HR (Table 5).
"""

from __future__ import annotations

import torch
from torch import nn


class PhysiologyGuidedAttention(nn.Module):
    """(B, T, C, H, W) + skin mask -> (B, T, C), space collapsed into channels."""

    def __init__(
        self,
        gamma: float = 1.0,
        eps: float = 1e-6,
        sigma_scale: float = 1.0,
        min_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        # gamma and eps are unstated in the paper. Both are almost dead: Eq. 4
        # renormalises A_tilde by its own L1 norm, so any positive gamma cancels
        # exactly, and eps only has to keep the divide finite on a dead channel.
        self.gamma = gamma
        self.eps = eps
        # sigma is described only as "controls the spatial spread". Deriving it
        # from the mask's own second moment makes it adapt to how much of the frame
        # the face fills, instead of being one more number to guess.
        self.sigma_scale = sigma_scale
        self.min_sigma = min_sigma

    def gaussian_prior(
        self, skin: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Eq. 1. skin (B, Hs, Ws) in [0, 1] -> (B, 1, H, W).

        Centroid and spread are the mask's own first and second moments, so the
        prior tracks the subject rather than assuming a centred face. A clip whose
        mask came back empty falls back to the frame centre with a broad sigma --
        that degrades PGA to a mild centre bias, which is the right failure.
        """
        if skin.dim() != 3:
            raise ValueError(f"expected skin (B, H, W), got {tuple(skin.shape)}")
        # Resample the mask onto the feature grid. area interpolation gives the
        # fraction of each cell that was skin; nearest would throw that away.
        resized = torch.nn.functional.interpolate(
            skin.unsqueeze(1).float(), size=(height, width), mode="area"
        ).squeeze(1)

        rows = torch.arange(height, device=skin.device, dtype=torch.float32)
        cols = torch.arange(width, device=skin.device, dtype=torch.float32)
        grid_h = rows.view(1, height, 1)
        grid_w = cols.view(1, 1, width)

        total = resized.sum(dim=(1, 2), keepdim=True)
        empty = total <= 1e-6
        weights = resized / total.clamp_min(1e-6)
        centre_h = (weights * grid_h).sum(dim=(1, 2), keepdim=True)
        centre_w = (weights * grid_w).sum(dim=(1, 2), keepdim=True)
        centre_h = torch.where(empty, torch.full_like(centre_h, (height - 1) / 2), centre_h)
        centre_w = torch.where(empty, torch.full_like(centre_w, (width - 1) / 2), centre_w)

        squared = (grid_h - centre_h) ** 2 + (grid_w - centre_w) ** 2
        # Mean per-axis variance, so sigma is a radius rather than a diameter.
        variance = (weights * squared).sum(dim=(1, 2), keepdim=True) / 2.0
        fallback = torch.full_like(variance, (max(height, width) / 4.0) ** 2)
        variance = torch.where(empty, fallback, variance)
        sigma = (variance.sqrt() * self.sigma_scale).clamp_min(self.min_sigma)

        return torch.exp(-squared / (2.0 * sigma * sigma)).unsqueeze(1)

    def forward(self, x: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
        batch, n_frames, channels, height, width = x.shape
        prior = self.gaussian_prior(skin, height, width).to(x.dtype)   # (B, 1, H, W)

        # Eq. 2: divisive normalisation per channel per frame, not a sigmoid. A
        # sigmoid saturates and forfeits the ratio between two bright regions; this
        # keeps it, so a channel can express "twice as much here as there".
        spatial_mean = x.mean(dim=(3, 4), keepdim=True)
        feat = x / (self.eps + spatial_mean) * self.gamma

        # Eq. 3: the prior is single-channel and broadcasts, so every channel is
        # constrained by anatomy while still free to specialise within it.
        combined = feat * prior.unsqueeze(1)

        # Eq. 4: rescale each (channel, frame) map to a fixed L1 energy. Without
        # it a channel could win the pooled average by sheer magnitude rather than
        # by where it is looking.
        l1 = combined.abs().sum(dim=(3, 4), keepdim=True)
        attention = (height * width) * combined / (2.0 * l1.clamp_min(self.eps))

        # Eq. 5: spatial pooling is what turns the video into a time series. From
        # here on the state transitions run purely along time, which is the whole
        # basis the Mamba backbone depends on.
        return (x * attention).mean(dim=(3, 4)).view(batch, n_frames, channels)
