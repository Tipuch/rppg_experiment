"""The readouts, measured on real predicted waveforms rather than on synthetic ones.

Every other test in this suite feeds the post-processing a signal built from a
formula, because that is the only way to know the answer. It is also how three
decisions in this project came to be made on evidence that did not hold: a beat
detector that read a synthetic notch pulse correctly at every rate from 45 to 160 bpm
lost 1.05 bpm of RMSE on real predictions, a quality metric that separated a clean
tone from white noise could not separate a prediction from white noise, and a marker
misalignment that affected 4 of 13 beats on one clip affected 83% of them across the
split.

So these tests read the cached forward pass instead. They are skipped when it is
absent -- `build/` is not distributed -- and they pin the numbers the sweep in
README.md quotes, so a change that degrades them fails here rather than in a table
nobody re-runs.

Regenerate the dump with `uv run python -m src.cli readout --force` on a machine with
a card.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.model.postprocess import reported_hr, template_match
from src.model.predict import analyse
from src.model.readout import load_dump, score
from src.paths import BUILD_ROOT

FPS = 30.0
DUMP = BUILD_ROOT / "readout_test_s900.npz"

pytestmark = pytest.mark.skipif(
    not DUMP.exists(), reason=f"{DUMP} absent; build/ is not distributed"
)


@pytest.fixture(scope="module")
def dump():
    return load_dump(DUMP)


def test_the_dump_is_the_split_the_readme_quotes(dump):
    """A guard on every number below. If the dump is regenerated over a different
    sample, the thresholds here are measuring something else and should be re-swept
    rather than relaxed."""
    predicted, truth, sources = dump
    assert predicted.shape == truth.shape
    assert len(predicted) == 1569
    assert predicted.shape[1] == 300
    assert set(sources) == {"mcd", "mrnirp", "ubfc"}


def test_the_reported_readout_holds_its_swept_score(dump):
    """The headline numbers, pinned. README.md quotes 3.41 MAE, 7.28 RMSE, 0.834 rho
    for `blend_median` over this split. Small drift is expected from a refactor; a
    regression back toward the 8.7 RMSE the 0.5-8 Hz detection band gave is not."""
    predicted, truth, sources = dump
    row = (
        score(predicted, truth, sources, FPS, variants=["blend_median"])
        .filter(pl.col("source") == "all")
        .row(0, named=True)
    )
    assert row["dropped"] == 0
    assert row["mae"] == pytest.approx(3.41, abs=0.15)
    assert row["rmse"] == pytest.approx(7.28, abs=0.30)
    assert row["rho"] == pytest.approx(0.834, abs=0.02)


def test_every_real_window_yields_a_rate(dump):
    """A readout that returns NaN on the hard windows reports the score of the easy
    ones. `score` counts drops for that reason; this asserts there are none."""
    predicted, _, _ = dump
    rates = np.array([reported_hr(w, FPS) for w in predicted])
    assert np.isfinite(rates).all()
    assert rates.min() > 40.0
    assert rates.max() < 240.0


def test_analyse_marks_real_peaks_on_the_trace_it_returns(dump):
    """The case the synthetic version of this test cannot produce.

    Before the markers were snapped, 3152 of 3785 beats across 300 real windows sat
    off a local maximum of the trace they are drawn on, and every one of the 300
    windows had at least one. On a lightly notched synthetic the same check passed,
    which is why this file exists.
    """
    predicted, _, _ = dump
    good = checked = 0
    for window in predicted[:300]:
        result = analyse(window[None, :], FPS)
        trace = result["trace"]
        inner = result["peaks"]
        inner = inner[(inner > 0) & (inner < trace.size - 1)]
        good += int(
            ((trace[inner] >= trace[inner - 1]) & (trace[inner] >= trace[inner + 1])).sum()
        )
        checked += int(inner.size)
    assert checked == 3785
    # 2 of 3785 are left alone on purpose. `align_to_peaks` searches half the shortest
    # beat interval and no further, so a marker whose nearest maximum is beyond that
    # stays put rather than being moved onto a neighbouring beat. Both residuals sit
    # 12 frames from the nearest maximum of the reporting-band trace, which is the two
    # bands disagreeing about where the beat is rather than which sample of it is
    # highest. A count rather than a ratio, because the split is pinned above.
    assert checked - good <= 2


def test_template_matching_cannot_grade_a_real_prediction(dump):
    """Why there is no TMCC floor. Measured on predictions, not on synthetic noise.

    The predicted and white-noise distributions overlap across their whole range, so
    no threshold admits most predictions while excluding most noise. The metric is not
    broken -- it separates contact PPG, which is the signal Charlton et al. 2025
    measured it on -- it is the predicted waveform it cannot grade.
    """
    predicted, truth, _ = dump
    rng = np.random.default_rng(0)
    def spread(arr):
        q = np.array([template_match(w, FPS) for w in arr])
        return np.percentile(q[np.isfinite(q)], [5, 50, 95])
    low_pred, mid_pred, high_pred = spread(predicted)
    low_noise, mid_noise, high_noise = spread(
        [rng.standard_normal(300) for _ in range(300)]
    )
    _, mid_truth, _ = spread(truth)
    # Predictions sit barely above noise, and the ranges cover each other.
    assert mid_pred - mid_noise < 0.05
    assert low_pred < high_noise and low_noise < high_pred
    # Contact PPG is the case it does separate, by an order more margin.
    assert mid_truth - mid_noise > 4 * (mid_pred - mid_noise)


def test_the_split_barely_samples_a_tachycardia(dump):
    """The stated limit on what this split can measure.

    0.75-4 Hz was chosen over 0.75-2.5 for headroom above 150 bpm, against a sweep
    that scored it worse. This records why the sweep cannot price that choice: almost
    none of the split is up there. If a regenerated dump ever does sample it, the
    band comparison is worth re-running rather than inherited.
    """
    _, truth, _ = dump
    rates = np.array([reported_hr(w, FPS) for w in truth])
    rates = rates[np.isfinite(rates)]
    assert np.percentile(rates, 99) < 130.0
    assert (rates > 150.0).mean() < 0.01
