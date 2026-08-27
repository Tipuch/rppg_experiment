"""Rewriting MCD's container so `-ss` stops scanning from frame 0.

MCD-rPPG's AVIs carry no index -- `ffprobe` reports `duration_ts=0`,
`nb_frames=N/A`, `duration=N/A` -- so ffmpeg cannot map a timestamp to a byte
offset and `-ss` decodes forward from the start. Measured on this corpus, that is
~15 ms per second of video: a seek to 120 s costs 1,449 ms against 44 ms once the
file is indexed, and it is **90% of the loader's per-window cost**. Keyframes are
dense (one per 0.4 s), so this was never a keyframe problem.

`-c copy` rewrites the container and leaves every video packet untouched, so the
decoded pixels must be identical. That is the property these tests pin, because a
remux that changes one pixel is not acceptable on a 0.1-0.5 LSB signal.

**The container has to stay AVI.** Measured: remuxing to MP4 or MKV silently drops
three frames on the 29.9167 fps clips -- a third of the corpus -- and the loss is
mid-stream, not at the tail, so every frame after it shifts and the contact-PPG
target desynchronises from the video. AVI-to-AVI drops nothing. An early check that
sampled only four window positions passed MP4 by luck; a denser grid caught it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.aggregation.remux import is_indexed, remux_path

MANIFEST = Path("build/clips.parquet")
needs_data = pytest.mark.skipif(
    not MANIFEST.exists(), reason="needs build/clips.parquet and the MCD videos"
)


def test_output_keeps_the_stem_and_stays_avi(tmp_path) -> None:
    """MP4 and MKV drop frames on this corpus; the extension is load-bearing."""
    out = remux_path(tmp_path, "mcd/1234_USBVideo_after")
    assert out.suffix == ".avi"
    assert out.parent == tmp_path
    assert "1234_USBVideo_after" in out.name


def test_a_clip_id_with_a_slash_becomes_one_filename(tmp_path) -> None:
    assert "/" not in remux_path(tmp_path, "mcd/a_b").name


def test_a_missing_file_is_not_indexed(tmp_path) -> None:
    assert not is_indexed(tmp_path / "nope.avi")


# --- against the real corpus -------------------------------------------------


@pytest.fixture(scope="module")
def clip() -> dict:
    import polars as pl

    return (
        pl.read_parquet(MANIFEST).filter(pl.col("source") == "mcd").to_dicts()[0]
    )


@needs_data
def test_the_source_really_is_unindexed(clip) -> None:
    """The premise. If this ever fails, the remux is no longer needed."""
    assert not is_indexed(Path(clip["video_path"]))


@needs_data
def test_remuxing_makes_the_file_indexed(clip, tmp_path) -> None:
    from src.aggregation.remux import remux_clip

    out = remux_clip(Path(clip["video_path"]), tmp_path, clip["clip_id"])
    assert out is not None
    assert is_indexed(out)


@needs_data
@pytest.mark.parametrize("start,k", [(0.0, 1.0), (15.0, 1.0), (29.6, 1.35),
                                     (45.0, 0.75), (90.0, 1.0), (150.0, 1.0)])
def test_the_decoded_window_is_bit_identical(clip, tmp_path_factory, start, k) -> None:
    """The gate. `-c copy` must not alter a single pixel.

    The grid is deliberately dense around 15-45 s: that is where the MP4 remux
    corrupted the stream, and a sparser grid missed it.
    """
    from src.aggregation.remux import remux_clip
    from src.aggregation.video import read_window
    from src.model.dataset import TARGET_FPS

    out_dir = tmp_path_factory.mktemp("remux")
    src = Path(clip["video_path"])
    out = remux_clip(src, out_dir, clip["clip_id"])
    assert out is not None

    box = (int(clip["box_x"]), int(clip["box_y"]), int(clip["box_side"]))
    if start + 160 / (TARGET_FPS * k) > clip["duration_s"] - 1:
        pytest.skip("window runs past the end of this clip")
    common = {"n_frames": 160, "target_fps": TARGET_FPS * k,
              "crop": box, "trust_crop": True}
    before = read_window(src, start, **common)
    after = read_window(out, start, **common)
    assert before is not None and after is not None
    assert after.shape == before.shape
    assert np.array_equal(after, before)


@needs_data
def test_build_skips_what_is_already_there(clip, tmp_path) -> None:
    """Resume: 3,600 clips is long enough that an interrupted run must not restart."""
    import polars as pl

    from src.aggregation.remux import build

    one = pl.DataFrame([clip])
    first = build(one, tmp_path)
    assert first["remuxed"] == 1 and first["skipped"] == 0
    second = build(one, tmp_path)
    assert second["remuxed"] == 0 and second["skipped"] == 1


@needs_data
def test_the_rewritten_manifest_uses_absolute_paths(clip, tmp_path) -> None:
    """Workers do not necessarily share the cwd the manifest was written from."""
    import polars as pl

    from src.aggregation.remux import build, rewrite_manifest

    one = pl.DataFrame([clip])
    build(one, tmp_path)
    updated = rewrite_manifest(one, tmp_path, source="mcd")
    path = updated["video_path"][0]
    assert Path(path).is_absolute(), path
    assert Path(path).exists()


@needs_data
def test_rewriting_changes_nothing_but_the_path(clip, tmp_path) -> None:
    """Geometry and labels must survive: `-c copy` does not move a pixel."""
    import polars as pl

    from src.aggregation.remux import build, rewrite_manifest

    one = pl.DataFrame([clip])
    build(one, tmp_path)
    updated = rewrite_manifest(one, tmp_path, source="mcd")
    for column in one.columns:
        if column == "video_path":
            continue
        assert one[column].equals(updated[column]), column


# --- the PPG must survive the repoint -----------------------------------------
#
# It did not, and it cost an epoch. MCD's contact PPG is located *from the video
# path*: `load_ppg` requires the video to sit in a `video/` directory and reads its
# `ppg_sync/<stem>.txt` sibling. Repointing `video_path` at build/mcd_remux broke
# that lookup, `load_ppg` returned None, and `_waveform` fell back to returning
# zeros -- silently. Nothing raised, every tensor kept its shape, and the run
# trained a full epoch against all-zero targets.
#
# The give-away in the logs was `train_time` pinned at exactly 1.000 (neg_pearson
# against a zero-variance target) while the frequency term "improved" to 0.604 --
# the model had learned the single constant label that argmax of a zero PSD
# produces. 1,492 of 1,500 dev windows were dropped as unscoreable.


@needs_data
def test_the_repointed_manifest_still_finds_the_contact_ppg(clip, tmp_path) -> None:
    import polars as pl

    from src.aggregation.remux import build, rewrite_manifest
    from src.model.waveform import load_ppg

    one = pl.DataFrame([clip])
    build(one, tmp_path)
    updated = rewrite_manifest(one, tmp_path, source="mcd").to_dicts()[0]

    ppg_source = Path(updated["ppg_video_path"])
    loaded = load_ppg(ppg_source.parent, video_path=ppg_source, fps=updated["fps"])
    assert loaded is not None, "the PPG is unreachable from the repointed manifest"
    assert len(loaded[1]) > 100


@needs_data
def test_a_window_from_the_repointed_manifest_has_a_real_target(clip, tmp_path) -> None:
    """The property that actually matters: a non-degenerate supervision signal."""
    import numpy as np
    import polars as pl

    from src.aggregation.remux import build, rewrite_manifest
    from src.model.dataset import WindowDataset

    one = pl.DataFrame([clip])
    build(one, tmp_path)
    updated = rewrite_manifest(one, tmp_path, source="mcd")
    updated = updated.with_columns(pl.lit(20.0).alias("window_start_s"))

    dataset = WindowDataset(updated, n_frames=300, train=False, cache_dir=None)
    wave = dataset._waveform(dataset.rows[0], 20.0, 1.0)
    assert np.isfinite(wave).all()
    assert float(np.std(wave)) > 1e-3, "target is flat -- the PPG did not load"
