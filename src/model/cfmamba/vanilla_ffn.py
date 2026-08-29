"""Vanilla time-domain FFN. The `--ffn vanilla` ablation, not part of the model.

Kept as its own file because it is a control, and a control that shares a file
with the thing it controls for tends to drift into it.

CFMamba Table 5 puts this at 0.59 MAE on UBFC-rPPG against DF-FFN's 0.36, and 4.70
against 4.03 on VIPL-HR. On UBFC it is *worse* than removing frequency processing
altogether (0.45), which is the paper's evidence that pointwise time-domain mixing
does not only fail to exploit periodicity -- it obscures it.
"""

from __future__ import annotations

import torch
from torch import nn


class VanillaFFN(nn.Module):
    """(B, T, C) -> (B, T, C). Two linear layers, as the ablation specifies."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
