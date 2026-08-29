"""Multi-temporal constraint Mamba, on a Mamba-3 core. RhythmMamba Section 3.3, Eq. 4.

Mamba's state transitions are a phase shift along the sequence, which is a good
match for a quasi-periodic pulse. RhythmMamba's addition is that one block should
be constrained by several time scales at once: the same weights see the clip
whole, in halves, and in quarters, so they must explain both the long-range
periodicity and the short-range trend.

The paper is explicit that this is a *constraint* rather than a fusion -- "we
replace multi-temporal fusion with multi-temporal constraint" -- so there is one
set of weights shared across all three paths, not one set per scale. Allocating a
Mamba per path would turn it into a multi-scale ensemble and change what
the layer means, which is why the sharing is checked in the tests.

**The scan is Mamba-3** (Lahoti et al., arXiv:2603.15569), not the Mamba-1
selective scan all three source papers call. Same (B, T, C) in and out; only the
recurrence changes.

    h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t          Prop. 1
    a_t = exp(dt_t A_t)   b_t = (1 - l_t) dt_t a_t   g_t = l_t dt_t

`l_t = sigmoid(trap_t)` is projected from the input like every other coefficient.
Mamba-1 keeps only the `g_t` term, which is Euler's rule; weighing both endpoints
is the trapezoid rule, second-order where Euler is first. A 1-3 Hz fundamental
sampled at 30 Hz gives ten to thirty samples per cycle, so the discretisation
error is not negligible, and it is *phase* error.

Three smaller changes. `A_t` gains an imaginary part, so the transition is a decay
times a data-dependent rotation applied to B and C as a RoPE (Prop. 2-4). A real
decay only attenuates; a rotation also advances a phase. B and C are
RMS-normalised and then offset by a learnable bias initialised to 1. And channels
are grouped into heads of `headdim`, sharing one state, one `dt` and one `A`,
where Mamba-1 gave each of the 160 its own.

There is **no depthwise convolution**: the trapezoid rule is itself an implicit
width-2 convolution on `B_t x_t` (Section 3.1.2), and Table 5a measures adding the
short conv back as worse (15.72 ppl against 15.85). `d_conv` is therefore not a
control here. That, plus a per-head scalar `A` in place of Mamba-1's
`(d_inner, d_state)` matrix, makes this cheaper than what it replaces; see
tests/test_budget.py.

**Direction.** Neither CFMamba nor RhythmMamba states whether the scan is
bidirectional, and the evidence is split. RhythmMamba's argument -- that a state
transition *is* the rPPG signal's temporal phase shift -- reads as directional,
and neither `mamba_ssm.Mamba` nor `Mamba3` has a `bimamba` flag: PhysMamba's came
from a fork. Against that, every vision Mamba since Vim runs both ways, and a pulse
is not causal in any physical sense -- nothing stops frame 40 from informing the
estimate at frame 10, and an offline rPPG model has the whole clip in hand.

Unidirectional is the default, matching what stock `mamba_ssm` provides and what
RhythmMamba's reasoning implies. `direction` is the control for settling it by
measurement rather than argument:

  "none"      one forward scan. The default.
  "shared"    also scan the reversed sequence with the same weights and sum, as
              Vim does. Adds no parameters; roughly doubles the scan's compute.
  "separate"  give the reverse pass its own scan. Roughly doubles the SSM
              parameter count, so it needs the budget checked again.

Mamba-3's scan is a Triton kernel with no CPU path, so this is the one module in
the package that cannot be unit-tested on CPU. Everything above and below it can,
which is why the package is split the way it is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Brought over from the Mamba-1 fit, where the published budget corroborated it.
# Mamba-3's own default of 128 is sized for a 1.5B language model.
DEFAULT_D_STATE = 16
# Channels per head; must divide d_inner (= expand * dim = 160). 32 gives 5 heads.
DEFAULT_HEADDIM = 32
DEFAULT_EXPAND = 2
# R=1 is the SISO recurrence, which is what Mamba-3 uses by default.
DEFAULT_MIMO_RANK = 1
# 32 rather than the reference 64: the shortest multi-temporal path is T/4, 75
# frames at T=300, and a 64-wide chunk leaves it no cross-chunk recurrence to run.
DEFAULT_CHUNK_SIZE = 32
# 1.0 rotates the whole state, giving d_state/2 angles as in Prop. 2. The reference
# default of 0.5 would leave four at d_state=16, too coarse for a heart rate.
DEFAULT_ROPE_FRACTION = 1.0
# "For the ith path, the sequence is divided into 2^(i-1) sub-sequences" -- three
# paths gives 1, 2 and 4 slices, so at T=300 the shortest sub-sequence is 75
# frames, or 2.5 s: still longer than one cardiac cycle at 45 bpm.
DEFAULT_PATHS = 3

DIRECTIONS = ("none", "shared", "separate")
# Unidirectional for now: it is what stock mamba_ssm gives without a fork, and
# whether the reverse pass justifies its compute is a question for the ablation, not
# for a comment. See the module docstring.
DEFAULT_DIRECTION = "none"

# Mamba-3 marks `dt_bias` and `D` itself but not these. Both are offsets, not
# weights: initialised to 1, holding B and C away from zero after the RMS norm, so
# decay pulls the layer toward a degenerate initialisation rather than a simpler
# function. The optimiser's name rule would catch them; the flag states so directly.
NO_DECAY_PARAMETERS = ("B_bias", "C_bias")


class MultiTemporalMamba(nn.Module):
    """(B, T, C) -> (B, T, C). One Mamba-3 block, constrained at several scales."""

    def __init__(
        self,
        dim: int,
        d_state: int = DEFAULT_D_STATE,
        headdim: int = DEFAULT_HEADDIM,
        expand: int = DEFAULT_EXPAND,
        mimo_rank: int = DEFAULT_MIMO_RANK,
        rope_fraction: float = DEFAULT_ROPE_FRACTION,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        paths: int = DEFAULT_PATHS,
        direction: str = DEFAULT_DIRECTION,
    ) -> None:
        super().__init__()
        from mamba_ssm import Mamba3

        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}, expected one of {DIRECTIONS}")
        d_inner = int(expand * dim)
        if d_inner % headdim:
            raise ValueError(
                f"headdim {headdim} does not divide d_inner {d_inner} "
                f"(= expand {expand} x dim {dim}); Mamba-3 groups channels into heads"
            )
        self.dim = dim
        self.paths = paths
        self.direction = direction
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size

        def build() -> nn.Module:
            scan = Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                rope_fraction=rope_fraction,
                is_mimo=mimo_rank > 1,
                mimo_rank=mimo_rank,
                chunk_size=self.chunk_size,
            )
            for name in NO_DECAY_PARAMETERS:
                parameter = getattr(scan, name, None)
                if parameter is not None:
                    parameter._no_weight_decay = True
            return scan

        # One module, called once per path. The figure draws "Conv -> sigma -> SSM"
        # on each path with "Share Weights" underneath; that triple is exactly
        # Mamba's internals, so a single Mamba reused per path is the accurate
        # reading and the cheapest one.
        self.mamba = build()
        # Only "separate" allocates a second scan. "shared" reuses self.mamba,
        # kept as a flag rather than an aliased attribute so the module appears
        # once in state_dict and the weight sharing stays unambiguous.
        self.mamba_reverse = build() if direction == "separate" else None
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
            # shorten the output and desynchronise every target, invisibly.
            sizes = [n_frames // slices] * slices
            for i in range(n_frames - sum(sizes)):
                sizes[i] += 1
            return torch.cat([self._scan(part) for part in x.split(sizes, dim=1)], dim=1)
        # Merging the slices into the batch axis runs all of them in one kernel
        # launch, which matters because this is called `paths` times per layer.
        folded = x.view(batch, slices, n_frames // slices, dim).reshape(
            batch * slices, n_frames // slices, dim
        )
        return self._scan(folded).view(batch, n_frames, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scanned = sum(self._scan_at_scale(x, 2**i) for i in range(self.paths))
        # Eq. 4: the summed paths are controlled by a projection of the layer input, so
        # the block can suppress its own output wherever the input carries nothing.
        return scanned * F.silu(self.gate(x))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, paths={self.paths} "
                f"(slices {[2**i for i in range(self.paths)]}), direction={self.direction}, "
                f"mimo_rank={self.mimo_rank}, chunk_size={self.chunk_size}")
