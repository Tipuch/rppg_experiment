"""Tests for waveform targets, loss and heart-rate readout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.waveform import (
    hr_from_waveform,
    load_ppg,
    neg_pearson,
    sample_ppg,
)


def _sine(bpm: float, seconds: float = 10.0, fps: float = 30.0) -> np.ndarray:
    t = np.arange(0, seconds, 1.0 / fps)
    return np.sin(2 * np.pi * (bpm / 60.0) * t)


@pytest.mark.parametrize("bpm", [48.0, 60.0, 72.0, 96.0, 120.0, 180.0])
def test_hr_from_waveform_recovers_a_known_frequency(bpm):
    assert hr_from_waveform(_sine(bpm)) == pytest.approx(bpm, abs=1.0)


def test_hr_from_waveform_is_amplitude_and_offset_invariant():
    wave = _sine(72.0)
    assert hr_from_waveform(wave * 0.001 + 50.0) == pytest.approx(
        hr_from_waveform(wave), abs=1e-6
    )


def test_hr_from_waveform_rejects_degenerate_input():
    assert np.isnan(hr_from_waveform(np.zeros(300)))
    assert np.isnan(hr_from_waveform(np.array([1.0, 2.0])))


def test_hr_from_waveform_survives_a_short_window():
    # 160 frames is the training window; raw bins there are 11.25 bpm apart, so
    # this only passes because the readout zero-pads before locating the peak.
    assert hr_from_waveform(_sine(72.0, seconds=160 / 30)) == pytest.approx(72.0, abs=2.0)


def test_neg_pearson_is_zero_for_an_exact_match():
    x = torch.randn(4, 200)
    assert neg_pearson(x, x).item() == pytest.approx(0.0, abs=1e-5)


def test_neg_pearson_is_two_for_an_inverted_match():
    x = torch.randn(4, 200)
    assert neg_pearson(x, -x).item() == pytest.approx(2.0, abs=1e-5)


def test_neg_pearson_ignores_scale_and_offset():
    x = torch.randn(4, 200)
    y = torch.randn(4, 200)
    baseline = neg_pearson(x, y)
    assert neg_pearson(x * 37.0 + 5.0, y).item() == pytest.approx(baseline.item(), abs=1e-4)


def test_neg_pearson_is_around_one_for_unrelated_signals():
    torch.manual_seed(0)
    value = neg_pearson(torch.randn(64, 500), torch.randn(64, 500)).item()
    assert 0.8 < value < 1.2


def test_neg_pearson_is_differentiable():
    x = torch.randn(2, 100, requires_grad=True)
    neg_pearson(x, torch.randn(2, 100)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_sample_ppg_standardises_and_lands_on_frame_times():
    times = np.arange(0, 10, 1 / 60)
    values = np.sin(2 * np.pi * 1.2 * times) * 400 + 2000
    frame_times = np.arange(0, 9.0, 1 / 30)
    out = sample_ppg(times, values, frame_times)
    assert out.shape == frame_times.shape
    assert out.dtype == np.float32
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-5)
    # Resampling 60 Hz -> 30 Hz must not move the cardiac peak.
    assert hr_from_waveform(out) == pytest.approx(72.0, abs=1.0)


def test_sample_ppg_handles_a_flat_trace():
    times = np.arange(0, 5, 0.01)
    out = sample_ppg(times, np.ones_like(times), np.arange(0, 4, 1 / 30))
    assert np.all(out == 0.0)


def test_load_ppg_reads_both_ubfc_layouts(tmp_path):
    two = tmp_path / "subject1"
    two.mkdir()
    ppg = np.sin(np.linspace(0, 20, 300))
    hr = np.full(300, 72.0)
    stamps = np.arange(300) / 30.0
    np.savetxt(two / "ground_truth.txt", np.vstack([ppg, hr, stamps]))
    times, values = load_ppg(two)
    assert times[0] == 0.0
    assert values == pytest.approx(ppg)

    one = tmp_path / "5-gt"
    one.mkdir()
    ms = np.arange(300) * 16.0
    np.savetxt(one / "gtdump.xmp",
               np.column_stack([ms, hr, np.full(300, 98.0), ppg]), delimiter=",")
    times, values = load_ppg(one)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx((ms[-1] - ms[0]) / 1000.0)
    assert values == pytest.approx(ppg)


def test_load_ppg_returns_none_when_absent(tmp_path):
    assert load_ppg(tmp_path) is None


def test_differencing_a_ppg_target_moves_the_peak_to_the_harmonic():
    """Why WindowDataset._waveform never differences the target.

    A PPG is not a sinusoid: its dicrotic notch puts real energy at 2f. First
    differencing scales the spectrum by |2*pi*f| and doubles the harmonic's weight
    relative to the fundamental, which is enough to flip which one is the peak.
    """
    fps, bpm = 30.0, 72.0
    t = np.arange(0, 10, 1 / fps)
    fundamental = 2 * np.pi * (bpm / 60.0) * t
    ppg = np.sin(fundamental) + 0.6 * np.sin(2 * fundamental)

    assert hr_from_waveform(ppg, fps) == pytest.approx(bpm, abs=1.0)
    assert hr_from_waveform(np.diff(ppg, prepend=ppg[0]), fps) == pytest.approx(
        2 * bpm, abs=2.0
    )


def test_neg_pearson_matches_the_physnet_reference():
    """Cross-check against the PhysNet formulation vendored in tools/rPPG-Toolbox.

    That file (neural_methods/loss/PhysNetNegPearsonLoss.py) is the original
    author's code, written as an explicit sum-of-products over each batch row.
    This is the same quantity in vector form; if they ever disagree, ours is the
    one that is wrong.
    """
    torch.manual_seed(0)
    preds, labels = torch.randn(5, 300), torch.randn(5, 300)

    reference = 0.0
    for i in range(preds.shape[0]):
        x, y = preds[i], labels[i]
        n = preds.shape[1]
        sum_x, sum_y = x.sum(), y.sum()
        pearson = (n * (x * y).sum() - sum_x * sum_y) / torch.sqrt(
            (n * x.pow(2).sum() - sum_x.pow(2)) * (n * y.pow(2).sum() - sum_y.pow(2))
        )
        reference += 1 - pearson
    reference = reference / preds.shape[0]

    assert neg_pearson(preds, labels).item() == pytest.approx(reference.item(), abs=1e-5)


def test_neg_pearson_matches_the_cosine_formulation():
    """The other vendored variant (NegPearsonLoss.py) uses cosine similarity of
    mean-centred signals, which is Pearson correlation by definition."""
    torch.manual_seed(1)
    preds, labels = torch.randn(4, 256), torch.randn(4, 256)
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    reference = torch.mean(
        1
        - cosine(
            preds - preds.mean(dim=1, keepdim=True),
            labels - labels.mean(dim=1, keepdim=True),
        )
    )
    assert neg_pearson(preds, labels).item() == pytest.approx(reference.item(), abs=1e-6)
