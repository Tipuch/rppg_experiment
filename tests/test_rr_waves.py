"""The RR network's inputs (src/vitals/waves.py): per-window z-scoring, and the
pulse-interval channels `tach` and `ls` that carry the breathing."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.vitals import waves

FPS, N = waves.WAVE_FPS, waves.WAVE_SAMPLES


def _rsa_wave(
    hr_hz: float = 1.2, rr_hz: float = 0.25, depth_hz: float = 0.15
) -> np.ndarray:
    """A pulse whose rate swings by `depth_hz` at the breathing rate `rr_hz`:
    respiratory sinus arrhythmia, the modulation `tach` and `ls` are built to show."""
    t = np.arange(N) / FPS
    rate = hr_hz + depth_hz * np.sin(2 * np.pi * rr_hz * t)
    return np.sin(2 * np.pi * np.cumsum(rate) / FPS)


def test_pp_signals_recover_the_breathing_rate_of_a_synthetic_rsa():
    tach, ls = waves.pp_signals(_rsa_wave())
    assert tach.shape == ls.shape == (N,) and tach.dtype == ls.dtype == np.float32
    assert abs(60 * waves.LS_FREQS[ls.argmax()] - 15.0) < 1.0
    # The tachogram swings at the breathing rate: its strongest frequency is 0.25 Hz.
    power = np.abs(np.fft.rfft(tach - tach.mean())) ** 2
    assert abs(np.fft.rfftfreq(N, 1 / FPS)[power.argmax()] - 0.25) < 1 / 30 + 1e-9


def test_a_flat_wave_gives_zeros_not_nan():
    for tach_or_ls in waves.pp_signals(np.zeros(N)):
        assert np.array_equal(tach_or_ls, np.zeros(N, np.float32))


def test_pp_table_matches_pp_signals_row_by_row():
    rows = pl.DataFrame({"clip_id": ["a", "b"], "segment_start_s": [0.0, 30.0],
                         "wave": np.stack([_rsa_wave(), np.zeros(N)]).astype(np.float32)})  # fmt: skip
    table = waves.pp_table(rows)
    assert table.columns == list(waves.PP_SCHEMA)
    for i, wave in enumerate(rows["wave"].to_numpy()):
        tach, ls = waves.pp_signals(wave)
        np.testing.assert_allclose(table["tach"][i].to_numpy(), tach, atol=1e-6)
        np.testing.assert_allclose(table["ls"][i].to_numpy(), ls, atol=1e-6)


def test_segment_waves_z_scores_each_window_apart_in_time_order():
    rng = np.random.default_rng(0)
    scales = [1.0, 50.0, 0.01]
    tagged = pl.DataFrame({
        "clip_id": ["c"] * 3, "segment_start_s": [0.0] * 3,
        # Out of order on purpose: the join must follow window_start_s.
        "window_start_s": [20.0, 0.0, 10.0],
        "ppg": (rng.normal(size=(3, 300)) * np.array(scales)[:, None] + 7).astype(np.float32),
    })  # fmt: skip
    wave = waves.segment_waves(tagged, "ppg")["wave"][0].to_numpy().reshape(3, 300)
    np.testing.assert_allclose(wave.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(wave.std(1), 1, atol=1e-5)
    np.testing.assert_allclose(
        wave[0], waves.zscore(tagged["ppg"][1].to_numpy()), atol=1e-6
    )
    assert np.array_equal(waves.zscore(np.ones(300)), np.zeros(300))
