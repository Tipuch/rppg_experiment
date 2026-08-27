"""Channel-adaptive modulation. CFMamba Eqs. 6-8.

The selective state space model at the heart of Mamba runs each channel through
its own recurrence and mixes channels only through linear projections, so it
cannot reweight a channel by how trustworthy that channel is. In rPPG the channels
differ enormously: some carry a coherent pulse, others carry a motion artefact or
an illumination glitch. CAM estimates that reliability from the whole clip at once
and rescales every channel accordingly.

Removing it costs the paper 0.62 -> 0.78 RMSE on UBFC-rPPG and 6.59 -> 8.63 on
VIPL-HR (Table 5).

CFMamba Eqs. 6-8 are a near-verbatim port of CMamba's GDD-MLP (Zeng et al.,
arXiv:2406.05316, Eqs. 5-6), which CFMamba cites as [32]. Two things follow from
reading that source, and neither is inferable from CFMamba alone:

**The hidden layer expands, it does not squeeze.** CMamba calls its width knob an
"expansion rate r" and reports that too small a value underfits. That is the
opposite of the squeeze-and-excitation convention CFMamba cites separately as
[31], which would have put a bottleneck here. Sizing this as a bottleneck costs
roughly 28k parameters per block at dim=96 -- enough to matter against a 0.91M
budget.

**CFMamba deviates from CMamba in where the two poolings are combined.** CMamba
runs the average and max descriptors through a shared MLP and sums the *outputs*;
CFMamba Eq. 6 sums the *descriptors* and applies the MLP once. CMamba's is the
default here, because the two are not equivalent and its version is the one with
an ablation behind it: summing first lets a large average cancel a large max
before either reaches a non-linearity, so a channel with a strong steady level and
a strong transient can end up indistinguishable from a quiet one. Summing after
keeps them separable. Same parameter count either way, and `pooling="cfmamba"`
restores the literal Eq. 6.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAdaptiveModulation(nn.Module):
    """(B, T, C) -> (B, T, C), rescaled by a descriptor pooled over T.

    Average pooling captures a channel's long-term trend; max pooling captures its
    strongest pulsatile excursion. Eq. 6 sums the two rather than concatenating
    them, so the descriptor stays (B, C) and the two MLPs stay small.
    """

    def __init__(
        self, dim: int, expansion: float = 1.0, pooling: str = "cmamba"
    ) -> None:
        super().__init__()
        if pooling not in ("cmamba", "cfmamba"):
            raise ValueError(f"unknown pooling {pooling!r}, expected cmamba or cfmamba")
        self.pooling = pooling
        # CFMamba calls these "lightweight" but gives no width. CMamba, the source
        # of the design, parameterises them by an expansion rate; the value is
        # settled by the published parameter budget in tests/test_budget.py.
        self.expansion = expansion
        hidden = max(1, round(dim * expansion))
        self.scale = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim)
        )
        self.shift = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim)
        )

    def extra_repr(self) -> str:
        return f"expansion={self.expansion}, pooling={self.pooling}"

    def _project(self, head: nn.Module, average: torch.Tensor, peak: torch.Tensor
                 ) -> torch.Tensor:
        """Combine the two pooled descriptors through `head`.

        CMamba Eq. 5 applies the shared MLP to each descriptor and adds the
        outputs; CFMamba Eq. 6 adds the descriptors and applies the MLP once.
        """
        if self.pooling == "cmamba":
            return head(average) + head(peak)
        return head(average + peak)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pooled over the temporal axis, so the descriptor describes the clip
        # rather than any one frame -- that is what makes it robust to the
        # transient noise it is meant to suppress. Average pooling captures a
        # channel's long-term trend, max its strongest pulsatile excursion.
        average, peak = x.mean(dim=1), x.amax(dim=1)
        # Eq. 7: sigmoid bounds both coefficients to (0, 1). The weight can only
        # attenuate, never amplify, which is what keeps the residual stream stable
        # when this is stacked L deep.
        weight = torch.sigmoid(self._project(self.scale, average, peak)).unsqueeze(1)
        bias = torch.sigmoid(self._project(self.shift, average, peak)).unsqueeze(1)
        # Eq. 8, broadcast over T so every time step in a channel is modulated
        # identically -- the point is to reweight channels, not to re-time them.
        return weight * x + bias
