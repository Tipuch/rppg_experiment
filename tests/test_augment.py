"""HR-balanced temporal resampling, and the crop/flip geometry around it.

The augmentation stretches the decode timebase and the PPG sampling timebase
together. If those two ever disagree, every target in the batch is silently
shifted against its frames and nothing raises -- the loss just stops falling. The
first three tests pin the relationship.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from src.model.dataset import (
    HR_HIGH_BPM,
    HR_LOW_BPM,
    K_FASTER,
    K_SLOWER,
    TARGET_FPS,
    WindowDataset,
)
from src.model.waveform import hr_from_waveform

DURATION = 60.0


def _dataset(train: bool = True, **kwargs) -> WindowDataset:
    manifest = pl.DataFrame({
        "clip_id": ["c0"], "subject_id": ["s0"], "video_path": ["/nonexistent/v.avi"],
        "mask_path": ["/nonexistent/m.npy"], "duration_s": [DURATION],
        "box_x": [0], "box_y": [0], "box_side": [0], "hr_bpm": [72.0],
        "fps": [30.0],
    })
    return WindowDataset(manifest, n_frames=160, train=train, targets=("hr_bpm",), **kwargs)


def _with_tone(dataset: WindowDataset, bpm: float) -> dict:
    """Install a synthetic contact PPG at `bpm`, bypassing the filesystem."""
    row = dataset.rows[0]
    times = np.arange(0.0, DURATION, 1.0 / 256.0)
    values = np.sin(2 * math.pi * (bpm / 60.0) * times)
    dataset._ppg[row["video_path"]] = (times, values)
    return row


@pytest.mark.parametrize("k", [0.7, 0.85, 1.0, 1.2, 1.4])
def test_resampling_scales_the_apparent_heart_rate_by_one_over_k(k: float) -> None:
    """A window read at TARGET_FPS*k and replayed at TARGET_FPS reads hr/k."""
    dataset = _dataset()
    row = _with_tone(dataset, 96.0)
    wave = dataset._waveform(row, start=1.0, k=k)
    assert hr_from_waveform(wave, fps=TARGET_FPS) == pytest.approx(96.0 / k, rel=0.02)


def test_a_fast_window_is_slowed_and_a_slow_one_is_sped_up() -> None:
    """Direction, which is the half of this that is easy to get backwards."""
    import random

    fast = _dataset()
    _with_tone(fast, HR_HIGH_BPM + 25)
    slow = _dataset()
    _with_tone(slow, HR_LOW_BPM - 20)

    for _ in range(20):
        rng = random.Random(_)
        k_fast = fast._resample_factor(fast.rows[0], 1.0, rng)
        k_slow = slow._resample_factor(slow.rows[0], 1.0, rng)
        assert 1.0 <= k_fast <= K_SLOWER
        assert K_FASTER <= k_slow <= 1.0


def test_the_middle_band_is_left_alone() -> None:
    import random

    dataset = _dataset()
    _with_tone(dataset, 0.5 * (HR_LOW_BPM + HR_HIGH_BPM))
    for seed in range(10):
        assert dataset._resample_factor(dataset.rows[0], 1.0, random.Random(seed)) == 1.0


def test_evaluation_never_resamples() -> None:
    import random

    dataset = _dataset(train=False)
    _with_tone(dataset, HR_HIGH_BPM + 30)
    assert dataset._resample_factor(dataset.rows[0], 1.0, random.Random(0)) == 1.0


def test_a_stretched_window_that_would_overrun_the_clip_falls_back() -> None:
    """k < 1 makes the window longer in real time; near the end there is no room."""
    import random

    dataset = _dataset()
    _with_tone(dataset, HR_LOW_BPM - 20)
    late = DURATION - dataset._span(1.0) - 0.01
    for seed in range(10):
        assert dataset._resample_factor(dataset.rows[0], late, random.Random(seed)) == 1.0


def test_window_start_leaves_room_for_the_whole_window() -> None:
    import random

    dataset = _dataset()
    rng = random.Random(0)
    for _ in range(50):
        for k in (0.7, 1.0, 1.4):
            start = dataset._window_start(dataset.rows[0], k, rng)
            assert start >= 0.0
            assert start + dataset._span(k) <= DURATION + 1e-9


def test_span_shrinks_as_k_grows() -> None:
    dataset = _dataset()
    assert dataset._span(1.0) == pytest.approx(160 / TARGET_FPS)
    assert dataset._span(2.0) == pytest.approx(dataset._span(1.0) / 2)


def test_flip_moves_frames_and_mask_together() -> None:
    """A prior derived from an unflipped mask would aim at the mirror face."""
    import random

    dataset = _dataset()
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:, 0] = 1.0                                  # skin hard against the left

    flipped_any = False
    for seed in range(20):
        out_frames, out_mask = dataset._maybe_flip(frames, mask, random.Random(seed))
        if out_mask[0, 0] == 1.0:
            assert np.array_equal(out_frames, frames)
            continue
        flipped_any = True
        # Mask moved to the right edge, and the frames moved with it.
        assert out_mask[0, -1] == 1.0
        assert np.array_equal(out_frames, frames[:, :, ::-1])
    assert flipped_any, "20 seeds produced no flip; the coin is not being tossed"


def test_evaluation_never_flips() -> None:
    import random

    dataset = _dataset(train=False)
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:, 0] = 1.0
    for seed in range(10):
        out_frames, out_mask = dataset._maybe_flip(frames, mask, random.Random(seed))
        assert np.array_equal(out_frames, frames)
        assert np.array_equal(out_mask, mask)


# --- segment starts are exact ------------------------------------------------
#
# Temporal jitter was removed. It moved an enumerated segment's start by up to
# +/-0.5 s in training, on the argument that a segment boundary is an artefact of
# the enumeration rather than of the recording. It was also the only thing that
# broke the strict non-overlap both source papers score under: at +/-0.5 s two
# adjacent 5.33 s windows could share up to 1 s of footage, so a window's
# neighbours leaked into it.


def _segment_dataset(train: bool) -> WindowDataset:
    manifest = pl.DataFrame({
        "clip_id": ["c0"], "subject_id": ["s0"], "video_path": ["/nonexistent/v.avi"],
        "mask_path": ["/nonexistent/m.npy"], "duration_s": [DURATION],
        "box_x": [0], "box_y": [0], "box_side": [0], "hr_bpm": [72.0],
        "fps": [30.0], "window_start_s": [20.0],
    })
    return WindowDataset(manifest, n_frames=160, train=train, targets=("hr_bpm",))


def test_an_enumerated_segment_starts_exactly_where_it_was_enumerated() -> None:
    """Training included: no jitter, so adjacent segments never share footage."""
    import random

    dataset = _segment_dataset(train=True)
    starts = {dataset._window_start(dataset.rows[0], 1.0, random.Random(s))
              for s in range(30)}
    assert starts == {20.0}


def test_evaluation_segments_are_unchanged_too() -> None:
    import random

    dataset = _segment_dataset(train=False)
    assert dataset._window_start(dataset.rows[0], 1.0, random.Random(0)) == 20.0


def test_a_clip_without_an_enumerated_start_still_moves_in_training() -> None:
    """One window per clip is a different case: it must still see the whole
    recording across epochs, or it would only ever train on one 5.33 s window."""
    import random

    dataset = _dataset(train=True)
    starts = {dataset._window_start(dataset.rows[0], 1.0, random.Random(s))
              for s in range(30)}
    assert len(starts) > 5


# --- the augmentation has to actually vary -----------------------------------
#
# It did not. The per-item RNG was seeded `self.seed * 1_000_003 + index`, and
# nothing mutated `self.seed` between epochs, so every segment saw one fixed crop,
# one fixed flip and one fixed k for an entire run -- a 50/50 partition assigned
# once, not an augmentation. Measured before the fix: segment 0 drew crop (22, 19)
# and flip False in epoch 0, 1, 2 and 3 alike.
#
# The usual remedy -- reseed from `torch.initial_seed()`, which DataLoader varies
# per epoch -- does not work here, because `persistent_workers=True` keeps workers
# alive across epochs so their base seed never changes either. Instead each worker
# draws from one continuous stream, which costs exact per-index reproducibility and
# buys augmentation that moves.


def test_repeated_draws_for_one_index_do_not_repeat() -> None:
    """The regression. Identical draws here mean the augmentation is frozen."""
    dataset = _dataset(train=True)
    crops, flips = set(), set()
    for _ in range(40):
        rng = dataset._rng(0)
        crops.add(dataset._crop_window(256, rng)[:2])
        flips.add(rng.random() < 0.5)
    assert len(crops) > 5, "the crop is frozen across draws"
    assert flips == {True, False}, "the flip is frozen across draws"


def test_evaluation_draws_stay_deterministic() -> None:
    """Eval must not move: a scored window has to be the same window every time."""
    dataset = _dataset(train=False)
    first = [dataset._rng(3).random() for _ in range(5)]
    assert len(set(first)) == 1


def test_two_indices_are_not_correlated() -> None:
    dataset = _dataset(train=True)
    assert dataset._rng(0).random() != dataset._rng(1).random()


# --- the supervision signal has to exist -------------------------------------
#
# A missing contact PPG does not raise. `load_ppg` returns None, `_waveform`
# returns zeros, every tensor keeps its shape, and training proceeds against a flat
# target -- `neg_pearson` pins at exactly 1.0 and the frequency term collapses onto
# the single constant label that argmax of a zero PSD produces. That looks like
# rapid progress in the log. It cost a 1.3-hour epoch once, after a manifest
# repoint severed MCD's labels from its clips.


def test_a_dataset_with_no_waveform_is_rejected() -> None:
    import pytest as _pytest

    from src.model.train import check_targets_are_supervised

    dataset = _dataset(train=True)          # video_path is /nonexistent, so no PPG
    with _pytest.raises(RuntimeError, match="flat"):
        check_targets_are_supervised(dataset, sample=8)


def test_a_dataset_with_a_real_waveform_passes() -> None:
    from src.model.train import check_targets_are_supervised

    dataset = _dataset(train=True)
    _with_tone(dataset, 72.0)
    check_targets_are_supervised(dataset, sample=8)


def test_the_check_names_the_counts_and_the_likely_cause() -> None:
    """The message has to be actionable: a bare 'assertion failed' would not be."""
    import pytest as _pytest

    from src.model.train import check_targets_are_supervised

    dataset = _dataset(train=True)          # one row, so one window is sampled
    with _pytest.raises(RuntimeError) as caught:
        check_targets_are_supervised(dataset, sample=8)
    message = str(caught.value)
    assert "1 of 1" in message
    assert "ppg_video_path" in message
