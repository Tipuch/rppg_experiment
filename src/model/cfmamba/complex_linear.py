"""Complex-valued linear projection. CFMamba Eqs. 10-11.

`complex_activation` resides here too, because it is the other half of what FreTS
calls a FreMLP: a complex linear followed by an activation applied separately to
the real and imaginary parts. CFMamba's Eqs. 10-11 describe only the linear half
and Section 3.3 names no activation at all, which -- tested -- leaves the whole
DF-FFN linear to float32 precision (affine error 2.2e-07 against 6.4e-01 for an
normal two-layer MLP). A "feed-forward network" that is one linear operator, and
whose ablation is worth 0.36 against 0.59 MAE, is an omission rather than a design.
FreTS Eq. 7 is where the activation goes, and FreTS is the work CFMamba references for
the operation.

Both halves of the DF-FFN project in the frequency domain, where a feature is a
complex number bringing magnitude *and* phase. A real-valued Linear applied to the
real and imaginary parts separately would treat them as two unrelated channels and
throw the phase relationship away. Complex multiplication is what couples them:

    (a + ib)(w + iv) = (aw - bv) + i(av + bw)

which is Eqs. 10-11 exactly. Its feasibility for spectral MLPs is the result
CFMamba references from FreTS (Yi et al., NeurIPS 36).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# FreTS writes the activation as a generic sigma and its released implementation
# uses ReLU. GELU is the default here instead: ReLU on an imaginary part zeroes
# half the phase plane directly, and phase is what carries the timing of a pulse
# -- the thing the negative-Pearson term is scored on. GELU is smooth and leaves a
# small negative tail, so a component near zero is attenuated rather than deleted.
# Both are available; the choice is a one-line ablation.
ACTIVATIONS = ("gelu", "relu", None)


def complex_activation(z: torch.Tensor, kind: str | None) -> torch.Tensor:
    """FreTS Eq. 7: apply the activation to the real and imaginary parts separately.

    Not to the magnitude, and not to the complex number as a whole -- FreTS is
    explicit that the two components are computed and activated independently, then
    restacked. That is what makes the layer non-linear while keeping the complex
    multiplication of Eqs. 10-11 intact underneath it.
    """
    if kind is None:
        return z
    if kind == "relu":
        return torch.complex(F.relu(z.real), F.relu(z.imag))
    if kind == "gelu":
        return torch.complex(F.gelu(z.real), F.gelu(z.imag))
    raise ValueError(f"unknown activation {kind!r}, expected one of {ACTIVATIONS}")


class ComplexLinear(nn.Module):
    """(..., in_features) complex -> (..., out_features) complex.

    Weights are stored as two real tensors rather than one `torch.complex`
    parameter: complex parameters are still second-class in autograd and in
    optimiser state, and this form is what the equations are written in anyway.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Var(out) = in_features * std^2 * (Var(x_re) + Var(x_im)), so std^2 =
        # 1/(2*in_features) keeps the output at the input's scale -- the complex
        # analogue of the 1/sqrt(fan_in) a real Linear uses. Getting this wrong is
        # invisible at init and shows up as a dead or exploding spectral branch.
        std = 1.0 / math.sqrt(2.0 * in_features)
        self.weight_re = nn.Parameter(torch.randn(in_features, out_features) * std)
        self.weight_im = nn.Parameter(torch.randn(in_features, out_features) * std)
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_features))
            self.bias_im = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.real, x.imag
        out_re = real @ self.weight_re - imag @ self.weight_im
        out_im = real @ self.weight_im + imag @ self.weight_re
        if self.bias_re is not None:
            out_re = out_re + self.bias_re
            out_im = out_im + self.bias_im
        return torch.complex(out_re, out_im)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias_re is not None}")
