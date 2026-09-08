"""Mamba-3's SISO scan in plain PyTorch, so it can leave CUDA.

`mamba_ssm.Mamba3` is a Triton kernel with no CPU path: it cannot be traced,
exported or unit-tested off a GPU. This module is the same recurrence written as
ordinary tensor operations, with the same parameter names and shapes, so a
trained checkpoint loads into it with `strict=True`.

It is a *drop-in for inference*, not a second training path. The scan here is
sequential over T, which a Triton kernel is not, so it is slower on a GPU and
allocates no chunk machinery. What it buys is portability.

**Where the algebra comes from.** Reading Proposition 1 off the paper and
implementing it directly is error-prone -- `rishikksh20/mamba3-pytorch`, the only
public port, gets it wrong twice and still scores a plausible-looking Pearson
0.9698 against the kernel. The kernel's own arrangement is used instead, from
`ops/triton/mamba3/mamba3_siso_fwd.py`, because it is the definition:

    a_t     = exp(A_t dt_t)                     the decay
    l_t     = sigmoid(trap_t)                   the trapezoid weight
    gamma_t = dt_t l_t                           this step's own contribution
    scale_t = gamma_t + dt_{t+1} (1 - l_{t+1})   plus the next step's

    S_t   = a_t (S_{t-1} + scale_{t-1} K_{t-1} (x) V_{t-1})
    y_t   = <Q_t, S_t> + (D + gamma_t <Q_t, K_t>) V_t
    out_t = y_t silu(z_t)

`scale_t` is what makes this equivalent to Proposition 1's two-term update rather
than a rewrite of it. Prop. 1 carries the previous step explicitly:

    h_t = a_t h_{t-1} + (1 - l_t) dt_t a_t Bx_{t-1} + l_t dt_t Bx_t

Collect the total weight on Bx_t once it has decayed to step s. Prop. 1 gives it
two paths -- l_t dt_t applied at t, and (1 - l_{t+1}) dt_{t+1} a_{t+1} applied at
t+1 -- and both arrive carrying exp(sum of a from t+1 to s). Their sum is
`scale_t`, so folding them into one scaling of K_t is exact, and it removes the
need to keep Bx_{t-1} around. The diagonal is the one place they differ: at s = t
only the `gamma_t` path has happened yet, which is why the (t, t) term is added
separately with `gamma_t` and the state carries `scale_t`.

**Three details that are invisible from the Python side of `mamba_ssm`.**

1. The RoPE angle is `cumsum(tanh(angle) * pi * dt)`, inclusive.
   `ops/triton/mamba3/angle_dt.py:94` squashes the projection with `tanh` and
   scales by pi *before* the cumsum. A port that feeds the raw projection in
   unbounded leaves every head rotating at the wrong rate, and nothing about the
   output looks structurally wrong.
2. The pairing is interleaved -- (0, 1), (2, 3), ... -- from the kernel's
   `reshape(..., HEADDIM_QK // 2, 2)` then `split`. Split-half pairing, which is
   what most RoPE helpers implement, scores measurably worse.
3. The `D` skip and the diagonal both sit *inside* the `silu(z)` gate, and the
   diagonal is computed on Q and K *before* rotation. That is not an
   approximation: a rotation is orthogonal, so it preserves the dot product, and
   the unrotated form is the better-conditioned way to write it.

The residual disagreement with the kernel is the kernel's own: it calls PTX
`cos.approx.f32`, `sin.approx.f32` and `tanh.approx.f32`, which
`ops/triton/mamba3/utils.py` documents as trading accuracy for speed. Where the
two differ, this module is the more accurate one.

`tests/test_mamba3_export.py` is the gate. It asserts Pearson above 0.9999
against the kernel on the trained checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# `mamba_ssm.Mamba3`'s own defaults, repeated so this module does not import it.
DEFAULT_A_FLOOR = 1e-4
RMS_NORM_EPS = 1e-5


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """`mamba_ssm.ops.triton.layernorm_gated.RMSNorm` with no gate, over the last axis."""
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * scale).to(x.dtype) * weight


def _apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate (..., 2n) by pairing element 2i with 2i+1. `cos`/`sin` are (..., n).

    The kernel reaches this pairing through `reshape(..., n, 2)` then `split`,
    which takes the even indices as one half and the odd as the other. Doing it
    with a reshape here rather than two strided slices keeps the op list short:
    a converter lowers `reshape` and `stack` cleanly, where `x[..., 0::2]`
    becomes a strided slice it may or may not fold.
    """
    pairs = x.unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Mamba3Sequential(nn.Module):
    """`mamba_ssm.Mamba3`'s SISO forward, as ordinary PyTorch. (B, L, d_model) in and out.

    Only the configuration CFMamba-Phys uses is supported: SISO (`is_mimo=False`),
    `ngroups=1`, and no output-projection norm. Each is asserted rather than
    silently ignored, because a checkpoint trained with any of them would load
    here without complaint and then compute something else.

    `chunk_size` is accepted and ignored. It is a parallel-scan tile size; a
    sequential scan has no tiles. It stays in the signature so the two
    implementations are constructed the same way.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = DEFAULT_A_FLOOR,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()
        if is_mimo:
            raise ValueError("Mamba3Sequential implements the SISO scan only")
        if is_outproj_norm:
            raise ValueError("Mamba3Sequential does not implement the output-projection norm")
        if ngroups != 1:
            raise ValueError(f"ngroups must be 1, got {ngroups}")
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be 0.5 or 1.0, got {rope_fraction}")

        factory = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.chunk_size = chunk_size
        self.mimo_rank = 1

        self.d_inner = int(expand * d_model)
        if self.d_inner % headdim:
            raise ValueError(f"headdim {headdim} does not divide d_inner {self.d_inner}")
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups

        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("rope_fraction leaves no angles to rotate")
        # rope_fraction=0.5 rotates only the leading half of the state; 1.0 rotates
        # all of it. CFMamba uses 1.0, so d_state/2 angles as in Proposition 2.
        self.rope_dim = 2 * self.num_rope_angles

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle] -- the same packing the
        # kernel's `in_proj` uses, so one weight matrix loads into either module.
        self._split = [
            self.d_inner,
            self.d_inner,
            d_state * self.num_bc_heads,
            d_state * self.num_bc_heads,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        self.in_proj = nn.Linear(d_model, sum(self._split), bias=False, **factory)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.dt_bias._no_weight_decay = True

        # Shaped (nheads, mimo_rank, d_state) even though mimo_rank is 1, because
        # that is the shape in the checkpoint.
        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, d_state), dtype=torch.float32, device=device)
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, d_state), dtype=torch.float32, device=device)
        )

        # Plain RMS norms, held in modules named `B_norm` and `C_norm` so their
        # `weight` lands at the checkpoint's key.
        self.B_norm = nn.Module()
        self.B_norm.weight = nn.Parameter(torch.ones(d_state, **factory))
        self.C_norm = nn.Module()
        self.C_norm.weight = nn.Parameter(torch.ones(d_state, **factory))

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, nheads={self.nheads}, "
            f"headdim={self.headdim}, rope_angles={self.num_rope_angles}"
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """(batch, seqlen, d_model) -> the same shape."""
        batch, seqlen, _ = u.shape
        heads, headdim, state = self.nheads, self.headdim, self.d_state

        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u), self._split, dim=-1
        )
        z = z.unflatten(-1, (heads, headdim))
        x = x.unflatten(-1, (heads, headdim))

        # `A` is a per-head scalar, negative by construction and held away from
        # zero: a decay of exactly 1 would make the state a running sum.
        A = (-F.softplus(dd_A.float())).clamp(max=-self.A_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        decay = torch.exp(A * dt)                                  # (B, L, H)
        trapezoid = torch.sigmoid(trap)                            # (B, L, H)

        # gamma is this step's weight on its own input; scale adds the weight the
        # next step will put on it. See the module docstring for why one scaling
        # of K reproduces Proposition 1's two terms.
        gamma = dt * trapezoid
        # Past the end of the sequence dt is zero, so the shifted term vanishes and
        # the last step contributes gamma alone. Padding dt rather than special-casing
        # the last index keeps this a single expression.
        dt_next = F.pad(dt[:, 1:], (0, 0, 0, 1))
        trapezoid_next = F.pad(trapezoid[:, 1:], (0, 0, 0, 1))
        scale = gamma + dt_next * (1.0 - trapezoid_next)

        # B and C are shared across heads (ngroups=1); only the bias is per head.
        key = _rms_norm(B, self.B_norm.weight).unsqueeze(2) + self.B_bias.squeeze(1)
        query = _rms_norm(C, self.C_norm.weight).unsqueeze(2) + self.C_bias.squeeze(1)

        # The angle projection is shared across heads too, but dt is not, so the
        # accumulated angle differs per head. Inclusive cumsum, as the kernel's is.
        angle_step = torch.tanh(angles) * math.pi                  # (B, L, R)
        theta = torch.cumsum(angle_step.unsqueeze(2) * dt.unsqueeze(-1), dim=1)
        cos, sin = torch.cos(theta), torch.sin(theta)              # (B, L, H, R)

        # The diagonal term, on unrotated Q and K -- a rotation preserves the dot
        # product, and this is the better-conditioned way to write it.
        diagonal = (query * key).sum(dim=-1) * gamma               # (B, L, H)

        if self.rope_dim == state:
            query_rot = _apply_interleaved_rope(query, cos, sin)
            key_rot = _apply_interleaved_rope(key, cos, sin)
        else:
            # rope_fraction=0.5: rotate the leading rope_dim channels, pass the rest.
            query_rot = torch.cat(
                (_apply_interleaved_rope(query[..., : self.rope_dim], cos, sin),
                 query[..., self.rope_dim:]), dim=-1)
            key_rot = torch.cat(
                (_apply_interleaved_rope(key[..., : self.rope_dim], cos, sin),
                 key[..., self.rope_dim:]), dim=-1)

        key_scaled = key_rot * scale.unsqueeze(-1)                 # (B, L, H, N)
        skip = self.D.view(1, heads, 1) + diagonal.unsqueeze(-1)   # (B, L, H, 1) per step

        # The scan. Sequential by necessity: this is the recurrence the Triton
        # kernel parallelises with a chunked scan, and there is no portable way to
        # express that. At T=300 the state is (B, H, headdim, d_state) -- small.
        carry = u.new_zeros(batch, heads, headdim, state)
        outputs = []
        for step in range(seqlen):
            carry = carry * decay[:, step].view(batch, heads, 1, 1)
            value = x[:, step]                                     # (B, H, P)
            y = torch.einsum("bhpn,bhn->bhp", carry, query_rot[:, step])
            y = y + skip[:, step] * value
            outputs.append(y * F.silu(z[:, step]))
            carry = carry + value.unsqueeze(-1) * key_scaled[:, step].unsqueeze(-2)

        y = torch.stack(outputs, dim=1).flatten(-2)                # (B, L, d_inner)
        return self.out_proj(y.to(u.dtype))
