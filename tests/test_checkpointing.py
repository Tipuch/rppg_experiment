"""A terminated run must not lose every epoch it finished.

`history.json` and `final.pt` were written only after the last epoch, so a run that
was interrupted left an empty output directory however far it had got. That
happened twice on this project: a 3-epoch full-corpus run terminated mid-epoch-1 after
2.6 hours, and a 6-epoch run terminated at a session boundary after another 2.6 hours.
Both had completed epochs whose metrics existed only in a terminal log.

Writing after each epoch costs a 4 MB checkpoint and a JSON dump against epochs
that take over an hour.
"""

from __future__ import annotations

import json

from src.model.train import write_progress


def test_history_is_readable_after_a_single_epoch(tmp_path) -> None:
    result = {"config": {"epochs": 6}}
    history = [{"epoch": 0, "train_loss": 4.19}]
    write_progress(tmp_path, result, history)

    saved = json.loads((tmp_path / "history.json").read_text())
    assert saved["history"] == history
    assert saved["config"]["epochs"] == 6


def test_each_epoch_overwrites_the_last(tmp_path) -> None:
    result = {"config": {}}
    write_progress(tmp_path, result, [{"epoch": 0}])
    write_progress(tmp_path, result, [{"epoch": 0}, {"epoch": 1}])

    saved = json.loads((tmp_path / "history.json").read_text())
    assert [r["epoch"] for r in saved["history"]] == [0, 1]


def test_the_partial_flag_marks_an_unfinished_run(tmp_path) -> None:
    """A reader has to be able to tell 2 epochs of 6 from a finished 2-epoch run."""
    result = {"config": {"epochs": 6}}
    write_progress(tmp_path, result, [{"epoch": 0}])
    assert json.loads((tmp_path / "history.json").read_text())["complete"] is False

    write_progress(tmp_path, result, [{"epoch": 0}], complete=True)
    assert json.loads((tmp_path / "history.json").read_text())["complete"] is True


def test_writing_does_not_mutate_the_caller_s_result(tmp_path) -> None:
    """The final write builds on `result`; a partial write must not pollute it."""
    result = {"config": {}}
    write_progress(tmp_path, result, [{"epoch": 0}])
    assert "history" not in result
    assert "complete" not in result


def test_the_directory_is_created_if_absent(tmp_path) -> None:
    target = tmp_path / "nested" / "run"
    write_progress(target, {"config": {}}, [{"epoch": 0}])
    assert (target / "history.json").exists()
