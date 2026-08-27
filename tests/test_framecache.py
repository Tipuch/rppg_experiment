"""The frame cache must feed the model the same window the decoder did.

The cache exists because UBFC-rPPG is uncompressed rawvideo: one 640x480 frame is
921,600 bytes on disk, so a 160-frame window costs 147 MB of read to yield a 23 MB
face box. Caching the box cuts that 6.4x -- but only if the window it hands back is
the same window.

"The same" is defined here rather than assumed, because it cannot be bit-identity
against ffmpeg. Measured: ffmpeg's `fps=` filter is pure nearest-frame selection --
every frame it delivers is an exact source frame, never an interpolation -- but it
selects with a running accumulator, while index arithmetic rounds each frame
independently. The two agree on the anchor exactly and disagree by at most one
source frame (35 ms at 28.7 fps) on the duplication boundaries where 28.7 fps is
stretched to 30.

So `window_indices` becomes the definition, and both paths use it. These tests pin
the three properties that make that safe: the window starts on the same frame, it
covers the same span, and the pulse read out of it is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.dataset import TARGET_FPS
from src.model.framecache import window_indices

# --- arithmetic, no filesystem ---------------------------------------------


def test_the_anchor_is_the_nearest_source_frame_to_the_start() -> None:
    """Measured against ffmpeg on six (clip, start) pairs: it picks this frame."""
    for fps in (28.671786, 29.782, 30.0, 60.0):
        for start in (0.0, 3.7, 10.0, 21.3):
            idx = window_indices(start, 4, fps, 1.0, 10_000)
            assert idx[0] == round(start * fps)


def test_at_native_thirty_and_k_one_the_window_is_consecutive_frames() -> None:
    idx = window_indices(2.0, 160, 30.0, 1.0, 10_000)
    assert np.array_equal(idx, np.arange(60, 220))


def test_a_sixty_fps_source_is_decimated_by_two() -> None:
    idx = window_indices(0.0, 8, 60.0, 1.0, 10_000)
    assert np.array_equal(idx, np.arange(8) * 2)


@pytest.mark.parametrize("k", [0.7, 0.85, 1.0, 1.2, 1.4])
@pytest.mark.parametrize("fps", [24.0, 28.671786, 30.0, 60.0])
def test_the_window_spans_n_over_thirty_k_seconds_whatever_the_source_rate(
    k: float, fps: float
) -> None:
    """The whole point of the resample: a frame count means one span everywhere."""
    n = 160
    idx = window_indices(5.0, n, fps, k, 1_000_000)
    covered = (idx[-1] - idx[0]) / fps
    assert covered == pytest.approx((n - 1) / (TARGET_FPS * k), abs=1.0 / fps)


@pytest.mark.parametrize("k", [0.7, 1.0, 1.4])
def test_indices_never_go_backwards(k: float) -> None:
    idx = window_indices(1.0, 160, 28.671786, k, 10_000)
    assert np.all(np.diff(idx) >= 0)


def test_running_off_the_end_repeats_the_last_frame() -> None:
    """Matches what WindowDataset already does when the decoder returns short."""
    available = 100
    idx = window_indices(3.0, 160, 30.0, 1.0, available)
    assert idx.max() == available - 1
    assert idx[-1] == available - 1


def test_the_anchor_is_clamped_into_a_short_clip() -> None:
    idx = window_indices(500.0, 16, 30.0, 1.0, 50)
    assert np.all(idx == 49)


# --- equivalence against the decoder, on real video -------------------------

pytestmark_real = pytest.mark.skipif(
    not __import__("pathlib").Path("build/clips_clean_ubfc.parquet").exists(),
    reason="needs the UBFC manifest and videos on disk",
)


@pytest.fixture(scope="module")
def clip() -> dict:
    import polars as pl

    manifest = pl.read_parquet("build/clips_clean_ubfc.parquet")
    return manifest.to_dicts()[0]


@pytest.fixture(scope="module")
def cache(clip, tmp_path_factory):
    """(directory, frames) for one real clip, built once for the whole module."""
    import polars as pl

    from src.model.framecache import build, open_clip

    out = tmp_path_factory.mktemp("framecache")
    counts = build(pl.DataFrame([clip]), out)
    assert counts["built"] == 1, counts
    frames = open_clip(out, clip["clip_id"])
    assert frames is not None
    return out, frames


@pytestmark_real
@pytest.mark.parametrize("start,k", [(0.0, 1.0), (10.0, 1.0), (3.7, 1.0),
                                     (10.0, 1.35), (10.0, 0.75)])
def test_every_cached_frame_is_a_real_source_frame(clip, cache, start, k) -> None:
    """Not an interpolation, and within one frame of what ffmpeg chose."""
    from pathlib import Path

    from src.aggregation.video import read_window

    decoded = read_window(
        Path(clip["video_path"]), start, 160, target_fps=TARGET_FPS * k,
        crop=(clip["box_x"], clip["box_y"], clip["box_side"]),
    )
    _, cached = cache
    idx = window_indices(start, len(decoded), clip["fps"], k, len(cached))
    ours = cached[idx]

    assert np.array_equal(ours[0], decoded[0]), "the window starts on a different frame"

    # Each frame ffmpeg delivered must appear in the cache within +/-1 of our index.
    for j in range(0, len(decoded), 7):
        near = cached[max(0, idx[j] - 1): idx[j] + 2]
        assert any(np.array_equal(f, decoded[j]) for f in near)


@pytestmark_real
@pytest.mark.parametrize("start,k", [(0.0, 1.0), (10.0, 1.0), (10.0, 1.35)])
def test_the_pulse_read_out_of_the_window_is_unchanged(clip, cache, start, k) -> None:
    """The property that actually matters: a +/-1 frame tie-break moves no rate."""
    from pathlib import Path

    from src.aggregation.video import read_window
    from src.model.dataset import BT601
    from src.model.waveform import hr_from_waveform

    decoded = read_window(
        Path(clip["video_path"]), start, 160, target_fps=TARGET_FPS * k,
        crop=(clip["box_x"], clip["box_y"], clip["box_side"]),
    )
    _, cached = cache
    idx = window_indices(start, len(decoded), clip["fps"], k, len(cached))

    def rate(frames: np.ndarray) -> float:
        luma = (frames.astype(np.float32) @ BT601).mean(axis=(1, 2))
        return hr_from_waveform(luma - luma.mean(), fps=TARGET_FPS)

    # One periodogram bin at T=160, 30 fps, 8x zero-padded.
    assert rate(cached[idx]) == pytest.approx(rate(decoded), abs=1.5)


@pytestmark_real
def test_the_dataset_prefers_the_cache_and_agrees_with_the_decoder(clip, cache) -> None:
    """End to end: the same fixed window through WindowDataset both ways.

    Evaluation settings, so nothing is random and the only thing that can differ is
    the tie-break frame. The tolerance is on the mean absolute difference of the
    standardised tensor, which a genuine misalignment -- a shifted window, a wrong
    crop, a flipped mask -- would blow through by orders of magnitude.
    """
    import polars as pl

    from src.model.dataset import WindowDataset

    cache_dir, _ = cache
    manifest = pl.DataFrame([{**clip, "window_start_s": 6.0}])
    common = {"n_frames": 160, "train": False, "hr_balance": False,
              "flip": False, "return_waveform": False}

    plain = WindowDataset(manifest, cache_dir=None, **common)[0]["frames"]
    cached = WindowDataset(manifest, cache_dir=cache_dir, **common)[0]["frames"]

    assert cached.shape == plain.shape
    assert (cached - plain).abs().mean() < 0.05


@pytestmark_real
def test_a_clip_with_no_cache_entry_still_loads(clip, tmp_path) -> None:
    """The fallback that keeps MCD and any uncached corpus working unchanged."""
    import polars as pl

    from src.model.dataset import WindowDataset

    manifest = pl.DataFrame([{**clip, "window_start_s": 6.0}])
    item = WindowDataset(manifest, n_frames=160, train=False, hr_balance=False,
                         flip=False, return_waveform=False, cache_dir=tmp_path)[0]
    assert item["frames"].shape == (160, 3, 128, 128)


# --- skipping the probe on an already-validated crop -------------------------
#
# `probe` costs 650 ms on an MCD clip, against 130 ms to decode the window it is
# describing. MCD's AVIs carry no usable duration, so `_probe_uncached` falls back
# to `_duration_from_keyframes`, which walks the whole keyframe index in a second
# subprocess. The LRU cache hides that only once a worker has seen the clip, and
# with 3,600 clips across 12 workers it is cold for most of the first epoch.
#
# Everything it returns is already in the manifest except width and height, and
# those are used for two things: clamping a crop box that face detection derived
# from the real frames, and choosing a downscale that cannot fire when the box is
# smaller than max_side. So a caller holding a manifest row can skip it.


@pytestmark_real
@pytest.mark.parametrize("start,k", [(0.0, 1.0), (10.0, 1.0), (10.0, 1.35)])
def test_trusting_the_crop_decodes_the_same_frames(clip, start, k) -> None:
    from pathlib import Path

    from src.aggregation.video import read_window

    box = (clip["box_x"], clip["box_y"], clip["box_side"])
    common = {"start_s": start, "n_frames": 160,
              "target_fps": TARGET_FPS * k, "crop": box}
    probed = read_window(Path(clip["video_path"]), **common)
    trusted = read_window(Path(clip["video_path"]), trust_crop=True, **common)
    assert trusted is not None
    assert np.array_equal(trusted, probed)


@pytestmark_real
def test_trusting_the_crop_never_calls_probe(clip, monkeypatch) -> None:
    """The point of the flag. A probe here would cost more than the decode."""
    from pathlib import Path

    from src.aggregation import video

    def explode(path):
        raise AssertionError("probe was called on a trusted crop")

    monkeypatch.setattr(video, "probe", explode)
    frames = video.read_window(
        Path(clip["video_path"]), 4.0, 32, target_fps=TARGET_FPS,
        crop=(clip["box_x"], clip["box_y"], clip["box_side"]), trust_crop=True,
    )
    assert frames is not None and len(frames) == 32


@pytestmark_real
def test_an_oversized_box_falls_back_to_probing(clip, monkeypatch) -> None:
    """trust_crop is a fast path, not a licence to skip the clamp that matters.

    A box wider than max_side would need the downscale branch, and that branch
    needs the real frame size. The flag must defer rather than emit a filter
    graph built on an assumption.
    """
    from pathlib import Path

    from src.aggregation import video

    seen = []
    real = video.probe
    monkeypatch.setattr(video, "probe", lambda p: (seen.append(p), real(p))[1])
    video.read_window(
        Path(clip["video_path"]), 4.0, 8, target_fps=TARGET_FPS,
        crop=(0, 0, clip["box_side"]), trust_crop=True, max_side=64,
    )
    assert seen, "an oversized box must still be probed"
