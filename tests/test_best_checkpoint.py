"""`best.pt` is the epoch the dev pass liked, not the epoch the run stopped on.

`last.pt` is overwritten every epoch, so the weights of a better earlier epoch are
gone the moment the next one finishes. Selecting a checkpoint therefore has to
happen *during* the run: each epoch asks whether it is the best dev score so far
and, if it is, writes a second checkpoint.

Selection reads dev only. Test is still scored once, on the last epoch, so the
reported number keeps the papers' protocol; best.pt is a separate artefact.
"""

from __future__ import annotations

import pytest

from src.model.train import best_index


def _history(*losses: float) -> list[dict]:
    return [{"epoch": i, "dev": {"loss": v, "mae": v * 2}}
            for i, v in enumerate(losses)]


def test_the_lowest_dev_loss_wins() -> None:
    assert best_index(_history(3.6, 2.1, 2.4, 2.0, 2.3), "dev") == 3


def test_the_last_epoch_wins_when_it_is_the_best() -> None:
    """The case where best.pt and last.pt hold the same weights."""
    assert best_index(_history(3.6, 2.4, 1.9), "dev") == 2


def test_another_metric_can_pick_another_epoch() -> None:
    history = _history(2.0, 2.1)
    history[1]["dev"]["mae"] = 1.0
    assert best_index(history, "dev") == 0
    assert best_index(history, "dev", "mae") == 1


def test_a_tie_keeps_the_earlier_epoch() -> None:
    """Only a strict improvement rewrites best.pt, so a plateau does not churn
    an 11 MB file for weights that score the same."""
    assert best_index(_history(2.0, 2.0), "dev") == 0


def test_epochs_with_no_dev_score_are_skipped() -> None:
    """A record from before the dev pass existed, or one whose metric came back
    NaN, is not a candidate -- it is an absent measurement, not a good one."""
    history = _history(2.5, 2.2)
    history.insert(0, {"epoch": -1, "train_loss": 3.4})
    history.append({"epoch": 2, "dev": {"loss": float("nan")}})
    assert best_index(history, "dev") == 2      # the 2.2 record, shifted by one


def test_an_empty_history_has_no_best() -> None:
    assert best_index([], "dev") is None
    assert best_index([{"epoch": 0, "train_loss": 3.4}], "dev") is None


def test_the_watch_split_is_named() -> None:
    """Runs with no dev split watch test instead, and the records key on that."""
    history = [{"epoch": 0, "test": {"loss": 2.0}}]
    assert best_index(history, "test") == 0
    assert best_index(history, "dev") is None


def test_a_higher_is_better_metric_is_refused() -> None:
    """Minimising corr or macc would select the worst epoch. Refuse rather than
    silently invert the comparison."""
    with pytest.raises(ValueError, match="corr"):
        best_index(_history(2.0), "dev", "corr")
