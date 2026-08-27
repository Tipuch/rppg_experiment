"""ComplexLinear must be complex matrix multiplication, not two real ones.

Eqs. 10-11 written out by hand are easy to get subtly wrong -- a sign on the
cross term, or the two weight matrices swapped -- and the result still trains,
just without the phase coupling that is the whole reason the layer exists. So the
test is against torch's own complex matmul rather than against a transcription of
the same equations.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.complex_linear import (
    ACTIVATIONS,
    ComplexLinear,
    complex_activation,
)


def test_matches_torch_complex_matmul() -> None:
    torch.manual_seed(0)
    layer = ComplexLinear(6, 4)
    x = torch.complex(torch.randn(3, 5, 6), torch.randn(3, 5, 6))

    weight = torch.complex(layer.weight_re, layer.weight_im)
    bias = torch.complex(layer.bias_re, layer.bias_im)
    assert torch.allclose(layer(x), x @ weight + bias, atol=1e-5)


def test_a_real_input_and_real_weights_reduce_to_a_real_linear() -> None:
    """Degenerate case: with no imaginary parts this is nn.Linear."""
    torch.manual_seed(0)
    layer = ComplexLinear(4, 3)
    with torch.no_grad():
        layer.weight_im.zero_()
        layer.bias_im.zero_()
    real = torch.randn(2, 4)
    x = torch.complex(real, torch.zeros_like(real))
    out = layer(x)
    assert torch.allclose(out.real, real @ layer.weight_re + layer.bias_re, atol=1e-6)
    assert torch.allclose(out.imag, torch.zeros_like(out.imag), atol=1e-6)


def test_multiplying_by_i_rotates_the_output() -> None:
    """The cross terms are what make this a rotation; a sign error breaks it."""
    torch.manual_seed(0)
    layer = ComplexLinear(4, 4, bias=False)
    x = torch.complex(torch.randn(2, 4), torch.randn(2, 4))
    assert torch.allclose(layer(1j * x), 1j * layer(x), atol=1e-5)


def test_initialisation_preserves_signal_scale() -> None:
    """Var(out) = in * std^2 * (Var(re) + Var(im)); std = 1/sqrt(2*in) holds it."""
    torch.manual_seed(0)
    layer = ComplexLinear(256, 256, bias=False)
    x = torch.complex(torch.randn(64, 256), torch.randn(64, 256))
    with torch.no_grad():
        out = layer(x)
    ratio = float(out.abs().pow(2).mean() / x.abs().pow(2).mean())
    assert 0.5 < ratio < 2.0, ratio


def test_gradients_reach_both_weight_halves() -> None:
    layer = ComplexLinear(4, 3)
    x = torch.complex(torch.randn(2, 4), torch.randn(2, 4))
    layer(x).abs().pow(2).mean().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None and parameter.grad.any(), name


def test_bias_can_be_omitted() -> None:
    layer = ComplexLinear(4, 3, bias=False)
    assert layer.bias_re is None and layer.bias_im is None
    assert layer(torch.complex(torch.randn(2, 4), torch.randn(2, 4))).shape == (2, 3)


# --- FreTS Eq. 7's activation ------------------------------------------------

def test_the_activation_hits_the_real_and_imaginary_parts_separately() -> None:
    """FreTS is explicit that the two components are activated independently, then
    restacked. Applying the activation to the magnitude, or to the complex number
    as a whole, would be a different operation."""
    z = torch.complex(torch.tensor([-2.0, 1.0, 3.0]), torch.tensor([1.0, -4.0, 0.5]))
    out = complex_activation(z, "relu")
    assert out.real.tolist() == [0.0, 1.0, 3.0]
    assert out.imag.tolist() == [1.0, 0.0, 0.5]


def test_no_activation_is_the_identity() -> None:
    """ComplexLinear itself stays exactly Eqs. 10-11: the activation is a property
    of FreMLP, not of the complex multiply, so None must be a true passthrough."""
    z = torch.complex(torch.randn(4), torch.randn(4))
    assert complex_activation(z, None) is z


def test_the_activation_is_non_linear() -> None:
    z1 = torch.complex(torch.randn(8), torch.randn(8))
    z2 = torch.complex(torch.randn(8), torch.randn(8))
    for kind in ("relu", "gelu"):
        lhs = complex_activation(2 * z1 - z2, kind)
        rhs = 2 * complex_activation(z1, kind) - complex_activation(z2, kind)
        assert not torch.allclose(lhs, rhs, atol=1e-4), kind


def test_an_unknown_activation_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        complex_activation(torch.complex(torch.randn(2), torch.randn(2)), "swish")
    assert set(ACTIVATIONS) == {"relu", "gelu", None}


def test_complex_linear_has_no_activation_of_its_own() -> None:
    """Eqs. 10-11 are purely linear; the module must stay that way."""
    torch.manual_seed(0)
    layer = ComplexLinear(6, 6, bias=False)
    z1 = torch.complex(torch.randn(3, 6), torch.randn(3, 6))
    z2 = torch.complex(torch.randn(3, 6), torch.randn(3, 6))
    assert torch.allclose(layer(2 * z1 - z2), 2 * layer(z1) - layer(z2), atol=1e-5)
