"""The two roots -- corpora in, artefacts out -- must be repointable without editing source.

A machine that keeps the corpora on another volume, or wants the multi-gigabyte
frame cache off the system disk, sets an environment variable. The defaults stay
relative to the working directory so a clone with no environment set behaves
exactly as it did before these variables existed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reloaded():
    """The roots resolve at import, so a changed environment needs a reload."""
    from src import paths

    return importlib.reload(paths)


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Leave `src.paths` matching the ambient environment for other test modules."""
    yield
    _reloaded()


def test_defaults_are_relative_to_the_working_directory(monkeypatch) -> None:
    monkeypatch.delenv("RPPG_DATA_ROOT", raising=False)
    monkeypatch.delenv("RPPG_BUILD_ROOT", raising=False)
    paths = _reloaded()
    assert paths.DATA_ROOT == Path("datasets")
    assert paths.BUILD_ROOT == Path("build")


def test_data_root_follows_its_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("RPPG_DATA_ROOT", "/mnt/corpora")
    paths = _reloaded()
    assert paths.DATA_ROOT / "ubfc-rppg" == Path("/mnt/corpora/ubfc-rppg")


def test_build_root_follows_its_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("RPPG_BUILD_ROOT", "/scratch/artefacts")
    paths = _reloaded()
    assert paths.BUILD_ROOT / "clips.parquet" == Path("/scratch/artefacts/clips.parquet")


def test_the_two_roots_are_independent(monkeypatch) -> None:
    """Repointing the corpora must not move the artefacts, and the reverse."""
    monkeypatch.setenv("RPPG_DATA_ROOT", "/mnt/corpora")
    monkeypatch.delenv("RPPG_BUILD_ROOT", raising=False)
    paths = _reloaded()
    assert paths.DATA_ROOT == Path("/mnt/corpora")
    assert paths.BUILD_ROOT == Path("build")


def test_a_tilde_is_expanded(monkeypatch) -> None:
    """`~/corpora` in a shell profile arrives here unexpanded."""
    monkeypatch.setenv("RPPG_DATA_ROOT", "~/corpora")
    paths = _reloaded()
    assert paths.DATA_ROOT == Path.home() / "corpora"


def test_an_empty_variable_falls_back_to_the_default(monkeypatch) -> None:
    """`RPPG_BUILD_ROOT=` must not resolve every artefact path to the CWD."""
    monkeypatch.setenv("RPPG_BUILD_ROOT", "")
    paths = _reloaded()
    assert paths.BUILD_ROOT == Path("build")
