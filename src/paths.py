"""The two roots this project reads from and writes to.

Every corpus path hangs off one directory and every derived artefact off another.
Both default to a relative path under the working directory, so a fresh clone run
from the repository root behaves as it always has:

    datasets/   the corpora, obtained separately -- see DATASETS.md
    build/      manifests, frame caches, masks, run directories, figures

Both are repointable from the environment, which is the whole reason this module
exists. The corpora are tens of gigabytes of human-subject video that will not
always live beside the source, and the frame cache is large enough to want its own
volume:

    RPPG_DATA_ROOT=/mnt/corpora RPPG_BUILD_ROOT=/scratch/build uv run python -m src.cli ...

Resolved once, at import. The constants are used as default argument values
throughout the package, so a variable changed mid-process would leave half the
paths pointing at the old root; a process that must switch roots reloads this
module, and `tests/test_paths.py` is the only thing that does.

Nothing here creates a directory. The command that writes is the command that
makes its parent, exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "RPPG_DATA_ROOT"
BUILD_ROOT_ENV = "RPPG_BUILD_ROOT"

DATA_ROOT_DEFAULT = "datasets"
BUILD_ROOT_DEFAULT = "build"


def _root(variable: str, fallback: str) -> Path:
    """`or` rather than a default, so `RPPG_BUILD_ROOT=` is not an empty path.

    An empty value resolving to `Path("")` would silently rewrite every artefact
    path to the working directory root -- the failure would appear later, as
    manifests scattered across the repository.
    """
    return Path(os.environ.get(variable) or fallback).expanduser()


DATA_ROOT = _root(DATA_ROOT_ENV, DATA_ROOT_DEFAULT)
BUILD_ROOT = _root(BUILD_ROOT_ENV, BUILD_ROOT_DEFAULT)
