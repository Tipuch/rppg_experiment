"""The composite waveform loss, CFMamba Eqs. 19-21.

Both terms are minimised by a correct prediction, but they are minimised by
*different* wrong predictions too, and that is the point of having both: the
temporal term is blind to which harmonic was locked onto, and the frequency term
is blind to phase. Tests below pin each blindness and show the other term covers
it.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.losses import (
    BPM_MAX,
    BPM_MIN,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    bpm_candidates,
    composite_loss,
    frequency_loss,
    spectral_power,
)
from src.model.waveform import neg_pearson

FPS = 30.0
N = 300


def _tone(bpm: float, phase: float = 0.0, n: int = N) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / FPS
    return torch.sin(2 * math.pi * (bpm / 60.0) * t + phase).unsqueeze(0)


# --- spectral power -----------------------------------------------------------

@pytest.mark.parametrize("bpm", [50.0, 72.0, 96.0, 145.0])
def test_power_peaks_at_the_tone_that_produced_it(bpm: float) -> None:
    candidates = bpm_candidates()
    peak = candidates[int(spectral_power(_tone(bpm), FPS, candidates).argmax())]
    assert float(peak) == pytest.approx(bpm, abs=1.5)


def test_the_candidate_grid_is_the_physiological_band_at_1_bpm_resolution() -> None:
    """Finer than the 11.25 bpm an unpadded 160-frame DFT would give."""
    candidates = bpm_candidates()
    assert float(candidates[0]) == BPM_MIN
    assert float(candidates[-1]) == BPM_MAX - 1
    assert float(candidates[1] - candidates[0]) == 1.0


def test_power_is_differentiable() -> None:
    signal = _tone(72.0).requires_grad_(True)
    spectral_power(signal, FPS, bpm_candidates()).sum().backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()


def test_power_ignores_a_constant_offset() -> None:
    """A DC shift is skin tone and room lighting, not a pulse."""
    candidates = bpm_candidates()
    plain = spectral_power(_tone(72.0), FPS, candidates)
    offset = spectral_power(_tone(72.0) + 7.0, FPS, candidates)
    assert torch.allclose(plain, offset, rtol=1e-3, atol=1e-2)


# --- frequency term -----------------------------------------------------------

def test_frequency_loss_is_lowest_when_the_rate_is_right() -> None:
    truth = _tone(72.0)
    right = float(frequency_loss(_tone(72.0), truth, FPS))
    near = float(frequency_loss(_tone(80.0), truth, FPS))
    wrong = float(frequency_loss(_tone(120.0), truth, FPS))
    assert right < near < wrong


def test_the_frequency_term_catches_a_harmonic_that_correlation_forgives() -> None:
    """Why Eq. 19 has two terms. A prediction at double the true rate can score a
    respectable correlation while being wrong by 72 bpm; the frequency term is
    what refuses it."""
    truth = _tone(72.0)
    harmonic = _tone(144.0)
    assert float(frequency_loss(harmonic, truth, FPS)) > \
        float(frequency_loss(_tone(72.0), truth, FPS))


def test_the_temporal_term_catches_a_phase_error_that_the_spectrum_forgives() -> None:
    """The blindness in the other direction: an inverted prediction has exactly
    the right spectrum and exactly the wrong waveform."""
    truth = _tone(72.0)
    inverted = _tone(72.0, phase=math.pi)
    assert float(frequency_loss(inverted, truth, FPS)) == \
        pytest.approx(float(frequency_loss(truth, truth, FPS)), abs=1e-4)
    assert float(neg_pearson(inverted, truth)) > 1.5


def test_the_label_comes_from_the_target_waveform() -> None:
    """Not from a label column: five UBFC subjects have a broken HR readout and an
    intact waveform (DATASETS.md)."""
    truth = _tone(93.0)
    losses = [float(frequency_loss(_tone(b), truth, FPS)) for b in (60.0, 93.0, 130.0)]
    assert losses[1] == min(losses)


def test_the_label_carries_no_gradient() -> None:
    """argmax over the target's spectrum is a label, not something to optimise.
    Letting gradient reach it would let the model move the goalposts."""
    predicted = _tone(90.0).requires_grad_(True)
    truth = _tone(72.0).requires_grad_(True)
    frequency_loss(predicted, truth, FPS).backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert truth.grad is None or float(truth.grad.abs().sum()) == 0.0


# --- composite ----------------------------------------------------------------

def test_the_temporal_term_is_weighted_to_be_worth_optimising() -> None:
    """CFMamba states Eq. 19 with alpha and beta as symbols and never gives values.
    RhythmFormer Section 3.4 Table 13 supplied 0.2 and 1.0, and this project ran
    that way -- until REPORT_cfmamba.md finding 2: the temporal term went flat from
    epoch 2 and at alpha=0.2 was ~1.5% of the final loss, so the optimiser had
    almost no reason to fix the waveform it is supposed to be predicting.

    0.8 is a departure from RhythmFormer, on this project's own measurement."""
    assert (DEFAULT_ALPHA, DEFAULT_BETA) == (0.8, 1.0)


def test_the_composite_is_lowest_for_an_exact_prediction() -> None:
    truth = _tone(72.0)
    exact, _ = composite_loss(_tone(72.0), truth, FPS)
    off_rate, _ = composite_loss(_tone(110.0), truth, FPS)
    inverted, _ = composite_loss(_tone(72.0, phase=math.pi), truth, FPS)
    assert float(exact) < float(off_rate)
    assert float(exact) < float(inverted)


def test_the_terms_are_reported_separately() -> None:
    """They fail differently -- a stuck temporal term means an uncorrelated
    waveform, a stuck frequency term means the wrong periodicity -- and one
    summed number cannot tell those apart."""
    total, parts = composite_loss(_tone(80.0), _tone(72.0), FPS)
    assert set(parts) == {"loss", "time", "freq"}
    assert float(parts["loss"]) == pytest.approx(
        DEFAULT_ALPHA * float(parts["time"]) + DEFAULT_BETA * float(parts["freq"]),
        rel=1e-5,
    )
    assert float(parts["loss"]) == pytest.approx(float(total), rel=1e-6)


def test_the_reported_terms_are_detached_tensors_not_floats() -> None:
    """Floats here would be three device-to-host syncs per training step.

    Each one waits for the GPU between the forward pass and `.backward()`, which is
    exactly where the GPU should be running ahead of the dataloader. The caller
    accumulates these on the device and syncs once a logging window instead.
    """
    _, parts = composite_loss(_tone(80.0), _tone(72.0), FPS)
    for name, value in parts.items():
        assert isinstance(value, torch.Tensor), name
        assert not value.requires_grad, name


def test_gradients_flow_to_the_prediction() -> None:
    predicted = _tone(90.0).requires_grad_(True)
    composite_loss(predicted, _tone(72.0), FPS)[0].backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0


def test_a_batch_is_reduced_to_one_number() -> None:
    predicted = torch.cat([_tone(60.0), _tone(90.0), _tone(120.0)])
    truth = torch.cat([_tone(62.0), _tone(88.0), _tone(118.0)])
    total, _ = composite_loss(predicted, truth, FPS)
    assert total.shape == ()
    assert torch.isfinite(total)
