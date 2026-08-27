"""One backbone layer: Mamba -> CAM -> add & norm -> FFN -> add & norm.

Post-norm, following RhythmMamba Fig. 3, where both residual branches are
normalised after the addition rather than before the sublayer.

Where CAM sits is the substance of CFMamba Section 3.2: it "operates on the
temporal representations produced by the state space model", so it modulates the
Mamba branch *before* that branch joins the residual stream. Putting it after the
addition would let it rescale the skip connection too, which is not what Eq. 8
says and would make a stack of L layers compound the attenuation.

Where the DF-FFN sits is Section 3.3: "positioned after the channel-adaptive
Mamba layer ... before residual aggregation".
"""

from __future__ import annotations

import torch
from torch import nn

from .cam import ChannelAdaptiveModulation
from .mamba_layer import MultiTemporalMamba


class ChannelAdaptiveMambaBlock(nn.Module):
    """(B, T, C) -> (B, T, C). `ffn` is injected so ablations swap one module."""

    def __init__(
        self,
        dim: int,
        ffn: nn.Module,
        use_cam: bool = True,
        cam_expansion: float = 1.0,
        cam_pooling: str = "cmamba",
        **mamba: int,
    ) -> None:
        super().__init__()
        self.mamba = MultiTemporalMamba(dim, **mamba)
        self.cam = (
            ChannelAdaptiveModulation(dim, expansion=cam_expansion, pooling=cam_pooling)
            if use_cam
            else None
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = ffn
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mamba(x)
        if self.cam is not None:
            mixed = self.cam(mixed)
        x = self.norm1(x + mixed)
        return self.norm2(x + self.ffn(x))
