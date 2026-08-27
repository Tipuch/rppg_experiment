"""Multi-temporal constraint Mamba. RhythmMamba Section 3.3, Eq. 4.

Mamba's state transitions are a phase shift along the sequence, which is a good
match for a quasi-periodic pulse. RhythmMamba's addition is that one block should
be constrained by several time scales at once: the same weights see the clip
whole, in halves, and in quarters, so they must explain both the long-range
periodicity and the short-range trend.

The paper is explicit that this is a *constraint* rather than a fusion -- "we
replace multi-temporal fusion with multi-temporal constraint" -- so there is one
set of weights shared across all three paths, not one set per scale. Allocating a
Mamba per path would quietly turn it into a multi-scale ensemble and change what
the layer means, which is why the sharing is asserted in the tests.

**Direction.** Neither CFMamba nor RhythmMamba says whether the scan is
bidirectional, and the evidence is split. RhythmMamba's argument -- that a state
transition *is* the rPPG signal's temporal phase shift -- reads as directional,
and stock `mamba_ssm` 2.3.2 has no `bimamba` flag at all: PhysMamba's came from a
fork. Against that, every vision Mamba since Vim runs both ways, and a pulse is
not causal in any physical sense -- nothing stops frame 40 from informing the
estimate at frame 10, and an offline rPPG model has the whole clip in hand.

Unidirectional is the default, matching what stock `mamba_ssm` provides and what
RhythmMamba's reasoning implies. `direction` is the knob for settling it by
measurement rather than argument:

  "none"      one forward scan. The default.
  "shared"    also scan the reversed sequence with the same weights and sum, as
              Vim does. Adds no parameters, so it stays on the published 0.91M;
              roughly doubles the scan's compute.
  "separate"  give the reverse pass its own scan. Roughly doubles the SSM
              parameter count, so it needs the budget re-checked.

`mamba_ssm`'s selective scan is a CUDA kernel with no CPU path, so this is the one
module in the package that cannot be unit-tested on CPU. Everything above and
below it can, which is why the package is split the way it is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Neither paper states the SSM's internal widths. These are Mamba's own defaults
# (Gu and Dao), which is the most likely thing an author calling the reference
# implementation would have used. tests/test_budget.py is what confirms or refutes
# it against the published 0.91M parameters.
DEFAULT_D_STATE = 16
DEFAULT_D_CONV = 4
DEFAULT_EXPAND = 2
# "For the ith path, the sequence is divided into 2^(i-1) sub-sequences" -- three
# paths gives 1, 2 and 4 slices, so at T=160 the shortest sub-sequence is 40
# frames, or 1.3 s: still longer than one cardiac cycle at 45 bpm.
DEFAULT_PATHS = 3

DIRECTIONS = ("none", "shared", "separate")
# Unidirectional for now: it is what stock mamba_ssm gives without a fork, and
# whether the reverse pass earns its compute is a question for the ablation, not
# for a comment. See the module docstring.
DEFAULT_DIRECTION = "none"


class MultiTemporalMamba(nn.Module):
    """(B, T, C) -> (B, T, C). One Mamba block, constrained at several scales."""

    def __init__(
        self,
        dim: int,
        d_state: int = DEFAULT_D_STATE,
        d_conv: int = DEFAULT_D_CONV,
        expand: int = DEFAULT_EXPAND,
        paths: int = DEFAULT_PATHS,
        direction: str = DEFAULT_DIRECTION,
    ) -> None:
        super().__init__()
        from mamba_ssm import Mamba

        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}, expected one of {DIRECTIONS}")
        self.dim = dim
        self.paths = paths
        self.direction = direction
        # One module, called once per path. The figure draws "Conv -> sigma -> SSM"
        # on each path with "Share Weights" underneath; that triple is exactly
        # Mamba's internals, so a single Mamba reused per path is the faithful
        # reading and the cheapest one.
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # Only "separate" allocates a second scan. "shared" reuses self.mamba,
        # held as a flag rather than an aliased attribute so the module appears
        # once in state_dict and the weight sharing stays unambiguous.
        self.mamba_reverse = (
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            if direction == "separate"
            else None
        )
        self.gate = nn.Linear(dim, dim, bias=False)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """The scan itself, run forward and -- unless disabled -- backward too.

        The two directions are summed, which is what Vim does. Concatenating and
        projecting would be the alternative; summing keeps the residual stream at
        one width and costs nothing.
        """
        forward = self.mamba(x)
        if self.direction == "none":
            return forward
        reverse = self.mamba_reverse if self.mamba_reverse is not None else self.mamba
        return forward + reverse(x.flip(1)).flip(1)

    def _scan_at_scale(self, x: torch.Tensor, slices: int) -> torch.Tensor:
        """Run the shared scan over `slices` contiguous sub-sequences."""
        batch, n_frames, dim = x.shape
        if slices == 1:
            return self._scan(x)
        if n_frames % slices:
            # Uneven split: loop rather than drop the remainder. Dropping it would
            # shorten the output and desynchronise every target, silently.
            sizes = [n_frames // slices] * slices
            for i in range(n_frames - sum(sizes)):
                sizes[i] += 1
            return torch.cat([self._scan(part) for part in x.split(sizes, dim=1)], dim=1)
        # Folding the slices into the batch axis runs all of them in one kernel
        # launch, which matters because this is called `paths` times per layer.
        folded = x.view(batch, slices, n_frames // slices, dim).reshape(
            batch * slices, n_frames // slices, dim
        )
        return self._scan(folded).view(batch, n_frames, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scanned = sum(self._scan_at_scale(x, 2**i) for i in range(self.paths))
        # Eq. 4: the summed paths are gated by a projection of the layer input, so
        # the block can suppress its own output wherever the input carries nothing.
        return scanned * F.silu(self.gate(x))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, paths={self.paths} "
                f"(slices {[2**i for i in range(self.paths)]}), direction={self.direction}")
