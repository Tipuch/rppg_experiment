"""The vanilla FFN ablation control, CFMamba Table 5.

It must be exactly what the ablation states -- "two linear layers to compose the
FFN" -- because its whole job is to be the thing DF-FFN is compared against. A
control that has acquired a convolution or a norm does not isolate the FFN.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.cfmamba.vanilla_ffn import VanillaFFN


def test_shape_is_preserved() -> None:
    assert VanillaFFN(dim=6, hidden=12)(torch.randn(2, 20, 6)).shape == (2, 20, 6)


def test_it_is_two_linear_layers_and_an_activation_and_nothing_else() -> None:
    kinds = [type(m) for m in VanillaFFN(dim=6, hidden=12).net]
    assert kinds == [nn.Linear, nn.GELU, nn.Linear]


def test_it_is_pointwise_in_time() -> None:
    """No temporal mixing at all: that absence is what makes it the control."""
    torch.manual_seed(0)
    ffn = VanillaFFN(dim=4, hidden=8)
    x = torch.randn(1, 12, 4)
    order = torch.randperm(12)
    with torch.no_grad():
        assert torch.allclose(ffn(x)[:, order], ffn(x[:, order]), atol=1e-6)


def test_gradients_reach_every_parameter() -> None:
    ffn = VanillaFFN(dim=4, hidden=8)
    ffn(torch.randn(2, 10, 4)).square().mean().backward()
    dead = [n for n, p in ffn.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
