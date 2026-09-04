"""Aggregation of per-window results into the five reported metrics.

The test that matters most here is the constant-predictor one. Every previous
attempt on this project stabilised to predicting a constant, and every aggregate
metric except rho reports that as a mediocre-but-credible result.
"""

from __future__ import annotations

import math

import pytest

from src.model.evaluate import format_metrics, per_subject, summarise


def _row(pred: float, true: float, subject: str = "s0", **extra) -> dict:
    return {"hr_pred": pred, "hr_true": true, "subject_id": subject,
            "clip_id": f"{subject}-c", "snr": 1.0, "macc": 0.5, **extra}


def test_a_perfect_prediction_scores_zero_error_and_unit_correlation() -> None:
    rows = [_row(hr, hr) for hr in (60.0, 72.0, 90.0, 110.0)]
    m = summarise(rows)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mape"] == pytest.approx(0.0)
    assert m["rho"] == pytest.approx(1.0)


def test_a_constant_predictor_is_exposed_by_rho_and_nothing_else() -> None:
    """The failure mode this project has hit three times. MAE looks like a result;
    rho is what states the prediction carries no information."""
    truths = [60.0, 72.0, 84.0, 96.0, 108.0]
    mean = sum(truths) / len(truths)
    m = summarise([_row(mean, t) for t in truths])
    assert m["mae"] < 20.0                       # credible-looking
    assert math.isnan(m["rho"])                  # and undefined, not "zero"
    assert m["hr_pred_std"] == pytest.approx(0.0)
    assert m["hr_true_std"] > 10.0


def test_rmse_punishes_one_large_error_more_than_mae_does() -> None:
    """Reported together because they answer different questions: MAE is the
    typical miss, RMSE is how bad the outliers are. With three exact windows and
    one 60 bpm miss out of four, MAE is e/4 and RMSE is e/2 -- exactly double.
    That factor is the whole reason the paper reports both."""
    error = 60.0
    rows = [_row(t, t) for t in (60.0, 70.0, 80.0)] + [_row(80.0 + error, 80.0)]
    m = summarise(rows)
    assert m["mae"] == pytest.approx(error / 4)
    assert m["rmse"] == pytest.approx(error / 2)
    assert m["rmse"] == pytest.approx(2 * m["mae"])


def test_mape_is_a_percentage_of_the_true_rate() -> None:
    m = summarise([_row(110.0, 100.0)])
    assert m["mape"] == pytest.approx(10.0)


def test_unusable_windows_are_dropped_and_counted() -> None:
    """Invisibly dropping them would let a model that fails on the hard half of the
    data report the easy half's score."""
    rows = [_row(72.0, 72.0), _row(float("nan"), 80.0), _row(90.0, float("nan"))]
    m = summarise(rows)
    assert m["windows"] == 3
    assert m["dropped"] == 2
    assert m["mae"] == pytest.approx(0.0)


def test_all_unusable_returns_a_count_rather_than_a_fake_zero() -> None:
    m = summarise([_row(float("nan"), float("nan")) for _ in range(3)])
    assert m["dropped"] == 3
    assert "mae" not in m
    assert "no usable windows" in format_metrics("dead", m)


def test_rho_is_undefined_for_a_single_window() -> None:
    assert math.isnan(summarise([_row(72.0, 74.0)])["rho"])


def test_per_subject_puts_the_worst_first() -> None:
    """An aggregate over 42 people hides which ones the model fails on, and is
    itself moved by any single bad subject."""
    rows = [
        _row(72.0, 72.0, "good"), _row(80.0, 80.0, "good"),
        _row(120.0, 70.0, "bad"), _row(130.0, 75.0, "bad"),
    ]
    ranked = per_subject(rows)
    assert ranked[0][0] == "bad"
    assert ranked[0][1]["mae"] > ranked[1][1]["mae"]


def test_format_is_readable_and_complete() -> None:
    line = format_metrics("dev", summarise([_row(72.0, 70.0), _row(90.0, 92.0)]))
    for field in ("MAE", "RMSE", "MAPE", "rho", "SNR", "MACC", "n="):
        assert field in line


# --- the loss terms, scored on the unseen side ----------------------------
#
# Heart-rate MAE on a handful of subjects is quantised by the periodogram bin
# spacing and swings several bpm between epochs on sampling noise. Measured on a
# 15-epoch UBFC run: dev MAE wandered 0.61-5.82 with no trend while dev loss rose
# from 1.80 to 14.01. Without these keys the run reports as healthy.


def test_the_loss_terms_are_averaged_over_every_window() -> None:
    """Including windows whose rate could not be read.

    A window the readout failed on still has a well-defined loss. Dropping it
    would put a moving denominator under the dev loss curve, so a run would look
    like it improved on any epoch where fewer windows happened to fail.
    """
    rows = [
        _row(60.0, 60.0, loss=1.0, time=0.5, freq=0.9),
        _row(72.0, 72.0, loss=3.0, time=0.7, freq=2.9),
        _row(float("nan"), 90.0, loss=8.0, time=0.9, freq=7.8),
    ]
    m = summarise(rows)
    assert m["dropped"] == 1
    assert m["loss"] == pytest.approx((1.0 + 3.0 + 8.0) / 3)
    assert m["time"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert m["freq"] == pytest.approx((0.9 + 2.9 + 7.8) / 3)


def test_a_split_with_no_readable_window_still_reports_its_loss() -> None:
    """The case that matters most: a model predicting nothing readable."""
    rows = [_row(float("nan"), 60.0, loss=2.0, time=1.0, freq=1.8)]
    m = summarise(rows)
    assert "mae" not in m
    assert m["loss"] == pytest.approx(2.0)


def test_a_run_without_loss_terms_omits_them_rather_than_reporting_zero() -> None:
    """Runs written before the terms were logged must not read as loss 0.0."""
    m = summarise([_row(60.0, 60.0), _row(72.0, 72.0)])
    assert "loss" not in m
    assert "time" not in m
    assert "freq" not in m


def test_the_evaluate_command_is_registered_and_needs_no_training() -> None:
    """The tables in README.md and MODEL_CARD.md go stale whenever the reported
    readout or the beat detector changes, because every number in them passes through
    `postprocess.compare`. Before this command the only way to refresh them was a
    50-epoch fit, so they stayed stale. It is registered here so the entry point
    cannot be removed without a failure."""
    from click.testing import CliRunner

    from src.cli import cli

    result = CliRunner().invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "without retraining" in result.output
    for option in ("--split", "--model", "--manifest", "--out"):
        assert option in result.output


def test_the_evaluate_command_says_what_it_needs_rather_than_crashing() -> None:
    """Mamba-3's scan kernel has no CPU path, so this cannot run without a card. The
    message has to say that rather than surfacing a Triton error."""
    import torch
    from click.testing import CliRunner

    from src.cli import cli

    if torch.cuda.is_available():
        pytest.skip("a card is present, so the guard cannot be exercised")
    result = CliRunner().invoke(cli, ["evaluate"])
    assert result.exit_code != 0
    assert "CUDA required" in result.output
