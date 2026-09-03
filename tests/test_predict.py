"""Single-video inference: checkpoint discovery, config recovery, stitch, beats.

The forward pass is not covered here -- Mamba-3's scan kernel has no CPU path, so it cannot
run without a card. Everything around it can, and everything around it is where a
silent wrong answer would come from: a checkpoint loaded with the wrong
architecture, or a bpm read off a trace assembled incorrectly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.model.predict import (
    analyse,
    config_from_checkpoint,
    latest_checkpoint,
    stitch,
)

FPS = 30.0


def _tone(bpm: float, n: int, fps: float = FPS, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fps
    return np.sin(2 * math.pi * (bpm / 60.0) * t + phase)


def _run(tmp_path, name: str, file: str = "final.pt"):
    directory = tmp_path / name
    directory.mkdir()
    path = directory / file
    path.write_bytes(b"x")
    return path


def test_latest_checkpoint_takes_the_newest(tmp_path):
    old = _run(tmp_path, "old")
    new = _run(tmp_path, "new", "last.pt")
    import os

    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert latest_checkpoint(tmp_path) == new


def test_latest_checkpoint_ignores_top_level_files(tmp_path):
    """build/runs/best.pt came before the config payload and cannot be rebuilt."""
    stray = tmp_path / "best.pt"
    stray.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(tmp_path)

    inside = _run(tmp_path, "run")
    assert latest_checkpoint(tmp_path) == inside


def test_config_survives_a_checkpoint_from_another_revision():
    """Unknown keys are dropped, absent ones default, and asdict is undone."""
    config = config_from_checkpoint({
        "n_frames": 300,
        "ffn": "vanilla",
        "out_dir": "build/runs/whatever",
        "betas": [0.9, 0.999],
        "sources": ["ubfc"],
        "an_option_that_no_longer_exists": 7,
    })
    assert config.n_frames == 300
    assert config.ffn == "vanilla"
    assert config.out_dir.name == "whatever"
    assert config.betas == (0.9, 0.999)
    assert config.sources == ("ubfc",)
    # Absent: falls back to the dataclass default rather than raising.
    assert config.resolution == 128


def test_stitch_removes_the_amplitude_a_pearson_loss_never_fixed():
    windows = np.stack([_tone(60, 60), 40.0 * _tone(60, 60) + 12.0])
    trace = stitch(windows)
    assert trace.shape == (120,)
    assert np.allclose(trace[:60], trace[60:], atol=1e-9)
    assert abs(trace.mean()) < 1e-9


def test_stitch_survives_a_flat_window():
    trace = stitch(np.stack([_tone(60, 60), np.zeros(60)]))
    assert np.isfinite(trace).all()
    assert np.allclose(trace[60:], 0.0)


def test_analyse_reports_nan_beats_rather_than_dividing_by_nothing():
    result = analyse(np.zeros((1, 300)), FPS)
    assert math.isnan(result["bpm_beats"])
    assert result["seams"] == []


def test_analyse_scores_against_a_contact_trace_when_given_one():
    """The truth goes through the same stitch and band-pass, so the two compare."""
    windows = np.stack([_tone(72.0, 300), _tone(72.0, 300)])
    # Same rate, quarter-cycle late, and forty times the amplitude -- none of which
    # the reported rate may depend on.
    truth = 40.0 * np.stack([_tone(72.0, 300, phase=-math.pi / 2)] * 2)
    result = analyse(windows, FPS, truth=truth)
    assert result["bpm_true"] == pytest.approx(72.0, abs=1.5)
    assert result["truth"].shape == result["trace"].shape
    # MACC is a maximum over lags, so a phase offset does not cost it anything.
    assert result["macc"] > 0.9


def test_analyse_omits_the_truth_terms_when_there_is_none():
    result = analyse(np.stack([_tone(72.0, 300)]), FPS)
    assert "truth" not in result
    assert "bpm_true" not in result


@pytest.mark.parametrize("bpm", [55.0, 72.0, 110.0, 145.0])
def test_both_rates_recover_a_known_tone(bpm):
    """Spectral and inter-beat agree, and both land on the tone that went in."""
    windows = np.stack([
        _tone(bpm, 300, phase=2 * math.pi * (bpm / 60.0) * (i * 300 / FPS))
        for i in range(2)
    ])
    result = analyse(windows, FPS)
    assert result["bpm_fft"] == pytest.approx(bpm, abs=1.5)
    assert result["bpm_beats"] == pytest.approx(bpm, abs=1.0)
    assert result["per_window_bpm"] == pytest.approx(bpm, abs=2.0)
    assert result["seconds"] == pytest.approx(20.0)
    assert result["seams"] == [300]
