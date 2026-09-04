"""Data preparation: segment enumeration, the 85/10/5 split, frame normalisation.

Every one of these guards something that fails invisibly. A split that leaks gives
a better score. A ratio computed over clips instead of segments gives the wrong
number of examples with the right number of recordings. A normalisation applied
per channel flattens the chrominance the pulse resides in.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np
import polars as pl
import pytest
import torch

from src.model.dataset import (
    AUG_FRACTION,
    FRAME_NORMALISATIONS,
    MASK_RES,
    TARGET_FPS,
    WindowDataset,
    expand_to_segments,
    prepare_splits,
)


def _manifest(n_subjects: int = 40, seconds: float = 60.0) -> pl.DataFrame:
    return pl.DataFrame({
        "clip_id": [f"ubfc/subject{i}" for i in range(n_subjects)],
        "source": ["ubfc"] * n_subjects,
        "subject_id": [f"ubfc_subject{i}" for i in range(n_subjects)],
        "video_path": [f"/x/subject{i}/vid.avi" for i in range(n_subjects)],
        "mask_path": [f"/x/m{i}.npy" for i in range(n_subjects)],
        "duration_s": [seconds] * n_subjects,
        "fps": [30.0] * n_subjects,
        "box_x": [0] * n_subjects, "box_y": [0] * n_subjects,
        "box_side": [0] * n_subjects, "hr_bpm": [72.0] * n_subjects,
    })


# --- segment enumeration ------------------------------------------------------

def test_a_clip_becomes_its_non_overlapping_windows() -> None:
    """One centred window per clip would make a 12-subject test set a mean over
    twelve numbers, which is not a measurement."""
    segments = expand_to_segments(_manifest(1, seconds=60.0), 160, TARGET_FPS)
    assert segments.height == int((60.0 - 160 / 30) // (160 / 30)) + 1
    starts = segments["window_start_s"].to_list()
    assert starts[0] == 0.0
    assert all(b - a == pytest.approx(160 / 30) for a, b in pairwise(starts))


def test_a_clip_shorter_than_one_window_still_contributes() -> None:
    """WindowDataset pads the tail; dropping the clip would lose the whole subject."""
    assert expand_to_segments(_manifest(1, seconds=2.0), 160, TARGET_FPS).height == 1


def test_the_default_stride_reproduces_the_toolbox_chunking() -> None:
    """rPPG-Toolbox does `frames[i*160:(i+1)*160]` for `n_frames // 160` chunks --
    strictly non-overlapping, partial tail thrown away. That is what all three papers
    ran under ("dividing them into fixed segments of 160 frames"), so the default
    has to match it exactly, not approximately.
    """
    span = 160 / TARGET_FPS
    for seconds in (20.0, 45.0, 68.2, 179.6, 225.0):
        manifest = _manifest(1, seconds=seconds)
        ours = expand_to_segments(manifest, 160, TARGET_FPS).height
        assert ours == int(seconds // span), seconds


def test_an_exact_multiple_does_not_lose_a_segment_to_floating_point() -> None:
    """A clip of exactly 160.0 s sits on a boundary where (duration - span) / stride
    evaluates to 28.999999 rather than 29.0. One MCD clip is exactly that length.
    """
    span = 160 / TARGET_FPS
    for multiple in (2, 10, 29, 30):
        seconds = multiple * span
        got = expand_to_segments(_manifest(1, seconds=seconds), 160, TARGET_FPS).height
        assert got == multiple, (multiple, seconds, got)


def test_no_segment_runs_past_the_end_of_its_clip() -> None:
    """The tail is dropped rather than padded, as the toolbox does. Padding would
    feed the model repeated frames and score them as if they were logged."""
    span = 160 / TARGET_FPS
    segments = expand_to_segments(_manifest(1, seconds=68.2), 160, TARGET_FPS)
    assert (segments["window_start_s"] + span <= 68.2 + 1e-6).all()


def test_a_stride_subsamples_a_large_corpus() -> None:
    dense = expand_to_segments(_manifest(1, seconds=180.0), 160, TARGET_FPS)
    sparse = expand_to_segments(_manifest(1, seconds=180.0), 160, TARGET_FPS,
                                stride_frames=480)
    assert 0 < sparse.height < dense.height


# --- the split ----------------------------------------------------------------

def test_the_split_hits_85_10_5_in_segments() -> None:
    splits = prepare_splits(_manifest(60), n_frames=160, fps=TARGET_FPS)
    total = sum(part.height for part in splits.values())
    for name, expected in (("train", 0.85), ("dev", 0.10), ("test", 0.05)):
        assert splits[name].height / total == pytest.approx(expected, abs=0.05), name


def test_no_subject_appears_on_two_sides() -> None:
    """Windows from one recording are near-duplicates, so a row-level split would
    score the model on people it trained on."""
    splits = prepare_splits(_manifest(60))
    ids = {name: set(part["subject_id"]) for name, part in splits.items()}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]


def test_the_split_is_deterministic_and_seed_dependent() -> None:
    manifest = _manifest(60)
    a = prepare_splits(manifest, seed=1)["test"]["subject_id"].to_list()
    b = prepare_splits(manifest, seed=1)["test"]["subject_id"].to_list()
    c = prepare_splits(manifest, seed=2)["test"]["subject_id"].to_list()
    assert a == b
    assert set(a) != set(c)


def test_the_ratios_hold_even_when_clip_lengths_are_wildly_uneven() -> None:
    """The reason prepare_splits expands before it splits.

    The underlying splitter balances by row count. Hand it clips and "row" means
    "recording", so a corpus mixing 20 s and 400 s clips lands 85/10/5 in
    recordings and something else in examples. Hand it segments and the two
    coincide. This corpus is intentionally 20x skewed, which is more extreme than
    the real one (UBFC 43-118 s against MCD 111-225 s).
    """
    brief = _manifest(20, seconds=20.0)
    lengthy = _manifest(20, seconds=400.0).with_columns(
        pl.col("subject_id") + "_long", pl.col("clip_id") + "_long"
    )
    splits = prepare_splits(pl.concat([brief, lengthy]))
    total = sum(part.height for part in splits.values())
    for name, expected in (("train", 0.85), ("dev", 0.10), ("test", 0.05)):
        assert splits[name].height / total == pytest.approx(expected, abs=0.05), name


def test_raw_normalisation_is_the_unit_interval() -> None:
    dataset = WindowDataset(_manifest(1), frame_norm="raw")
    out = dataset._normalise(torch.full((4, 3, 8, 8), 255.0))
    assert float(out.max()) == pytest.approx(1.0)


def test_a_dead_window_normalises_to_zeros_rather_than_nan() -> None:
    dataset = WindowDataset(_manifest(1), frame_norm="standardized")
    out = dataset._normalise(torch.zeros(4, 3, 8, 8))
    assert torch.isfinite(out).all() and float(out.abs().max()) == 0.0


def test_an_unknown_normalisation_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown frame_norm"):
        WindowDataset(_manifest(1), frame_norm="imagenet")
    assert set(FRAME_NORMALISATIONS) == {"standardized", "raw"}


# --- MCD's frame-aligned waveform --------------------------------------------

def test_mcd_ppg_is_read_as_one_sample_per_frame(tmp_path) -> None:
    """MCD releases no timestamps because it needs none: ppg_sync is already
    frame-synchronised, so the time axis comes from the video's frame rate."""
    from src.model.waveform import load_ppg

    video_dir = tmp_path / "video"
    ppg_dir = tmp_path / "ppg_sync"
    video_dir.mkdir()
    ppg_dir.mkdir()
    video = video_dir / "1020_USBVideo_after.avi"
    video.touch()
    values = np.sin(np.arange(300) * 2 * np.pi * 1.2 / 30.0) * 50 + 120
    np.savetxt(ppg_dir / "1020_USBVideo_after.txt",
               np.column_stack([values, np.full(300, 0.001)]))

    got = load_ppg(video.parent, video_path=video, fps=30.0)
    assert got is not None
    times, loaded = got
    assert len(times) == 300
    assert times[-1] == pytest.approx(299 / 30.0)
    assert loaded == pytest.approx(values, abs=1e-6)


def test_a_clip_with_no_waveform_returns_none(tmp_path) -> None:
    from src.model.waveform import load_ppg

    assert load_ppg(tmp_path, video_path=tmp_path / "x.mov", fps=30.0) is None


# --- geometry: one resample, in the right direction ---------------------------

def test_inter_area_enlarging_is_nearest_neighbour() -> None:
    """The fault that motivated removing the 256 intermediate.

    94% of clips have a face box smaller than 256, so the old chain enlarged to 256
    before shrinking to 128 -- and cv2.INTER_AREA enlarging is exactly equal to
    INTER_NEAREST. Pinned here because it is surprising, undocumented in the OpenCV
    signature, and the reason `_resize` branches on direction at all.
    """
    small = np.arange(16, dtype=np.float32).reshape(4, 4)
    area = cv2.resize(small, (8, 8), interpolation=cv2.INTER_AREA)
    nearest = cv2.resize(small, (8, 8), interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(small, (8, 8), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(area, nearest)
    assert not np.array_equal(area, linear)
    assert len(np.unique(linear)) > len(np.unique(area))


def test_resize_shrinks_by_area_and_enlarges_by_interpolation() -> None:
    coarse = np.arange(16, dtype=np.float32).reshape(4, 4)
    fine = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    enlarged = WindowDataset._resize(coarse, 8)
    shrunk = WindowDataset._resize(fine, 32)
    assert np.array_equal(enlarged, cv2.resize(coarse, (8, 8), interpolation=cv2.INTER_LINEAR))
    assert np.array_equal(shrunk, cv2.resize(fine, (32, 32), interpolation=cv2.INTER_AREA))


def test_shrinking_preserves_the_mean() -> None:
    """Why INTER_AREA and not INTER_LINEAR when shrinking.

    The signal is a spatially smooth sub-LSB brightness change, so what has to
    survive downsampling is the local mean. An area filter integrates every source
    pixel that lands in an output pixel and preserves it; linear point-samples and
    aliases.
    """
    rng = np.random.default_rng(1)
    image = (rng.random((235, 235)).astype(np.float32) * 40 + 120)
    area = WindowDataset._resize(image, 128)
    linear = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    assert abs(float(area.mean()) - float(image.mean())) < 0.01
    assert abs(float(area.mean()) - float(image.mean())) <= \
        abs(float(linear.mean()) - float(image.mean())) + 1e-6


def test_a_uniform_brightness_step_survives_downsampling_exactly() -> None:
    """The pulse *is* a uniform brightness step. It must pass through unchanged."""
    base = np.full((235, 235), 120.0, dtype=np.float32)
    stepped = base + 0.3                                   # a sub-LSB pulse
    delta = WindowDataset._resize(stepped, 128) - WindowDataset._resize(base, 128)
    assert float(delta.min()) == pytest.approx(0.3, abs=1e-4)
    assert float(delta.max()) == pytest.approx(0.3, abs=1e-4)


def test_resize_is_a_no_op_at_the_target_size() -> None:
    image = np.random.default_rng(2).random((128, 128)).astype(np.float32)
    assert WindowDataset._resize(image, 128) is image


# --- the crop window ---------------------------------------------------------

def test_the_crop_keeps_the_intended_fraction_and_stays_in_bounds() -> None:
    import random

    dataset = _dataset_for_geometry(train=True)
    for side in (94, 128, 210, 355):
        for seed in range(20):
            top, left, size = dataset._crop_window(side, random.Random(seed))
            assert size == max(1, round(side * AUG_FRACTION))
            assert 0 <= top <= side - size
            assert 0 <= left <= side - size


def test_evaluation_takes_the_centre_crop() -> None:
    import random

    dataset = _dataset_for_geometry(train=False)
    top, left, size = dataset._crop_window(256, random.Random(0))
    assert (top, left) == ((256 - size) // 2, (256 - size) // 2)


def test_training_actually_jitters_the_crop() -> None:
    import random

    dataset = _dataset_for_geometry(train=True)
    seen = {dataset._crop_window(256, random.Random(s))[:2] for s in range(30)}
    assert len(seen) > 5, "the crop is not moving"


# --- the mask mapping --------------------------------------------------------

def test_the_mask_window_tracks_the_frame_window() -> None:
    """A mis-mapped mask hands PGA a prior describing a different part of the face
    than the frames show, and both tensors keep the right shape, so nothing raises.
    """
    dataset = _dataset_for_geometry(train=True)
    mask = np.zeros((MASK_RES, MASK_RES), dtype=np.float32)
    mask[: MASK_RES // 2, : MASK_RES // 2] = 1.0           # skin in the top-left
    dataset._masks["m"] = mask.astype(bool)
    row = {"mask_path": "m"}

    source_side = 200
    size = round(source_side * AUG_FRACTION)
    top_left = dataset._mask_window(row, 0, 0, size, source_side)
    bottom_right = dataset._mask_window(
        row, source_side - size, source_side - size, size, source_side
    )
    # Both windows are 224 of the mask's 256 pixels (175 source pixels scaled by
    # 256/200), so with skin filling the top-left 128x128 the coverage is exactly
    # computable -- and pinning it exactly is a stronger statement than a margin:
    #
    #   top-left     window [0:224]  covers rows 0-127 x cols 0-127 -> 128^2/224^2
    #   bottom-right window [32:256] covers rows 32-127 x cols 32-127 -> 96^2/224^2
    #
    # An 87.5% crop cannot isolate a quadrant, so the two necessarily overlap; what
    # this checks is that sliding the window slides the coverage, by the right
    # amount and in the right direction.
    assert top_left.shape == (224, 224)
    assert float(top_left.mean()) == pytest.approx(128**2 / 224**2, abs=0.01)
    assert float(bottom_right.mean()) == pytest.approx(96**2 / 224**2, abs=0.01)
    assert float(top_left.mean()) > float(bottom_right.mean())


def test_the_mask_window_stays_inside_the_mask() -> None:
    dataset = _dataset_for_geometry(train=True)
    dataset._masks["m"] = np.ones((MASK_RES, MASK_RES), dtype=bool)
    row = {"mask_path": "m"}
    for source_side in (94, 128, 210, 355, 512):
        size = round(source_side * AUG_FRACTION)
        for top in (0, (source_side - size) // 2, source_side - size):
            window = dataset._mask_window(row, top, top, size, source_side)
            assert window.shape[0] == window.shape[1]
            assert 1 <= window.shape[0] <= MASK_RES


# --- the face box ------------------------------------------------------------

def test_a_missing_detection_falls_back_to_a_centred_square() -> None:
    """Not to the whole 640x480 frame: resizing that to a square would stretch the
    face horizontally by 33%."""
    raw = np.zeros((4, 480, 640, 3), dtype=np.uint8)
    box, side = WindowDataset._face_box(raw, {"box_side": 0, "box_x": 0, "box_y": 0})
    assert side == 480
    assert box.shape == (4, 480, 480, 3)


def test_a_detection_is_used_as_given() -> None:
    raw = np.arange(2 * 100 * 120 * 3, dtype=np.uint8).reshape(2, 100, 120, 3)
    box, side = WindowDataset._face_box(raw, {"box_side": 40, "box_x": 10, "box_y": 20})
    assert side == 40
    assert np.array_equal(box, raw[:, 20:60, 10:50])


def _dataset_for_geometry(train: bool) -> WindowDataset:
    return WindowDataset(_manifest(1), train=train)
