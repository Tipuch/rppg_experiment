"""The optimised dataloader paths must produce exactly what the readable ones did.

The hot path was restructured for speed (masking by multiply, luma gathered before
the matmul). These pin the numerics so a later speed change cannot quietly alter
what the model is fed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.dataset import BT601


@pytest.fixture
def frames_and_mask() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    frames = rng.uniform(20, 220, size=(12, 32, 32, 3)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[6:26, 8:24] = True
    return frames, mask


def test_masking_by_multiply_matches_boolean_assignment(frames_and_mask) -> None:
    """The raw path swapped two-sided fancy indexing for a multiply."""
    frames, mask = frames_and_mask
    reference = np.zeros_like(frames)
    repeated = mask[None, :, :].repeat(len(frames), axis=0)
    reference[repeated] = frames[repeated]
    assert frames * mask[None, :, :, None] == pytest.approx(reference)


def test_luma_gathered_before_the_matmul_matches_projecting_first(
    frames_and_mask,
) -> None:
    """mean_y gathers masked pixels then projects, instead of the reverse."""
    frames, mask = frames_and_mask
    reference = (frames @ BT601)[:, mask].mean(axis=1)
    assert (frames[:, mask] @ BT601).mean(axis=1) == pytest.approx(reference, abs=1e-4)
