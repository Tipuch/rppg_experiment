"""Regressions: faults that reached the tree, and what each looked like.

They are collected in one file because what they have in common is a failure
mode rather than a module: each produced a plausible artifact rather than an
exception, so none was caught by the code that had to be correct for it to
matter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.datasets import mrnirp

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# A column of nulls, from a row cached by an older version
# --------------------------------------------------------------------------- #
def test_a_row_missing_a_newer_column_does_not_poison_the_manifest():
    """Polars infers a frame's schema from its first 100 rows.

    29 sessions were cached before `fps_source` existed, so their rows lacked the
    key. Every one of the first 100 segment rows came from those, the column was
    inferred as Null, and the first real value later in the frame failed to
    append -- inside `expand_to_segments`, three steps from the cause.
    """
    fresh = {k: None for k in mrnirp.ROW_SCHEMA} | {
        "clip_id": "mrnirp/b", "source": "mrnirp", "corpus": "indoor",
        "fps_source": "camera_log",
    }
    stale = {k: v for k, v in fresh.items() if k != "fps_source"}
    stale["clip_id"] = "mrnirp/a"

    frame = pl.DataFrame(
        [{k: r.get(k) for k in mrnirp.ROW_SCHEMA} for r in (stale, fresh)],
        schema=mrnirp.ROW_SCHEMA,
    )
    assert frame["fps_source"].dtype == pl.String
    assert frame["fps_source"].to_list() == [None, "camera_log"]
    # The round trip through dicts is what expand_to_segments does.
    assert pl.DataFrame(frame.to_dicts())["fps_source"].dtype == pl.String


def test_every_row_key_is_declared_in_the_schema():
    """A key written by `prepare` but absent from ROW_SCHEMA is silently dropped."""
    source = (REPO / "src/datasets/mrnirp.py").read_text()
    body = source[source.index('        row = {'):source.index('        (session_dir / "meta.json")')]
    written = {line.split('"')[1] for line in body.splitlines()
               if line.strip().startswith('"') and '":' in line}
    assert written <= set(mrnirp.ROW_SCHEMA), written - set(mrnirp.ROW_SCHEMA)


# --------------------------------------------------------------------------- #
# .gitignore shadowing the source tree
# --------------------------------------------------------------------------- #
def test_the_data_ignore_rules_cannot_swallow_a_source_package():
    """`datasets/` unanchored matches `src/datasets/` too.

    The entire corpus-reader package was invisible to git, which no test, lint or
    import would ever notice -- the code runs fine locally and simply does not
    exist for anyone who clones the repository.
    """
    rules = [
        line.strip() for line in (REPO / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for directory in ("datasets", "build", "models", "research"):
        assert f"{directory}/" not in rules, (
            f"'{directory}/' is unanchored and matches any directory of that name "
            f"at any depth; use '/{directory}/'"
        )
    assert (REPO / "src/datasets/__init__.py").exists()


# --------------------------------------------------------------------------- #
# SegFace exhausting a shared GPU
# --------------------------------------------------------------------------- #
def test_a_gpu_out_of_memory_falls_back_to_cpu_instead_of_dropping_the_clip(monkeypatch):
    """Several ingest workers each hold a ~1.2 GB SegFace copy on one card.

    When that ran out, `prepare` caught the error per session and skipped it, so
    15 sessions silently vanished from the manifest. A slow clip beats a missing
    one.
    """
    from src.aggregation import skin

    calls = []

    def fake_segment_on(loaded, frames_rgb, batch):
        calls.append(loaded[1])
        if loaded[1] != "cpu":
            raise RuntimeError("CUDA error: out of memory")
        return np.zeros(frames_rgb.shape[:3], dtype=np.uint8)

    monkeypatch.setattr(skin, "_segment_on", fake_segment_on)
    monkeypatch.setattr(skin, "load_model", lambda device=None: (object(), device or "cuda"))

    out = skin.segment(np.zeros((2, 256, 256, 3), dtype=np.uint8))
    assert out.shape == (2, 256, 256)
    assert calls == ["cuda", "cpu"]


def test_the_device_reported_is_where_the_model_actually_is(monkeypatch):
    """After the CPU fallback, a later no-argument call must not claim "cuda".

    `load_model` used to report the device it was *asked* for. Once `segment` had
    fallen back, the next call would be handed (cpu_model, "cuda") and move its
    input to a card the weights had left -- a device-mismatch error thrown from
    inside the model, nowhere near the cause.
    """
    from src.aggregation import skin

    class FakeModel:
        def __init__(self):
            self.device = "cuda"

        def to(self, dev):
            self.device = dev
            return self

    monkeypatch.setattr(skin, "_MODEL", FakeModel())
    monkeypatch.setattr(skin, "_DEVICE", "cuda")

    model, dev = skin.load_model("cpu")
    assert (dev, model.device) == ("cpu", "cpu")
    model, dev = skin.load_model()          # no argument: must not revert to cuda
    assert (dev, model.device) == ("cpu", "cpu")


def test_an_error_that_is_not_out_of_memory_is_re_raised(monkeypatch):
    """Blanket-catching here would turn a real bug into a silently slow clip."""
    from src.aggregation import skin

    def fake_segment_on(loaded, frames_rgb, batch):
        raise RuntimeError("weights did not fit")

    monkeypatch.setattr(skin, "_segment_on", fake_segment_on)
    monkeypatch.setattr(skin, "load_model", lambda device=None: (object(), device or "cuda"))

    with pytest.raises(RuntimeError, match="weights did not fit"):
        skin.segment(np.zeros((1, 256, 256, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Picking frames out of an archive that holds two streams
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # A self-contained Indoor bundle: NIR and RGB side by side, session-prefixed.
        ([f"S1_still_940/{d}/Frame{i:05d}.pgm" for d in ("NIR", "RGB") for i in range(3)],
         [f"S1_still_940/RGB/Frame{i:05d}.pgm" for i in range(3)]),
        # An orphan RGB-###.zip, and a Car RGB.zip member: both hold only colour
        # frames at RGB/Frame*.pgm with no session prefix.
        ([f"RGB/Frame{i:05d}.pgm" for i in range(3)],
         [f"RGB/Frame{i:05d}.pgm" for i in range(3)]),
    ],
)
def test_only_colour_frames_are_selected_whatever_the_archive_layout(names, expected):
    """Selecting on the corpus rather than the directory took NIR from a bundle.

    NIR frames are the same size and dtype as RGB ones, so the cache, the
    manifest and the loader would all have accepted a monochrome clip demosaiced
    as if it were Bayer.
    """
    frames = [n for n in names if n.endswith(".pgm")]
    selected = sorted(n for n in frames if "RGB/" in n) or sorted(frames)
    assert selected == expected


# --------------------------------------------------------------------------- #
# Aligning the pulse trace to frame 0
# --------------------------------------------------------------------------- #
def test_load_ppg_rebases_onto_the_camera_origin_recorded_at_build_time(tmp_path):
    """The oximeter's first sample is not always the video's first frame.

    Rebasing on the oximeter instead shifts the whole target by the difference,
    which for a waveform loss is a phase error and not a visible one.
    """
    from scipy.io import savemat

    start = 1_518_987_800.0
    times = start + np.arange(600) / 60.0
    savemat(tmp_path / "pulseOx.mat", {
        "pulseOxTime": times.reshape(1, -1),
        "pulseOxRecord": np.array([np.array([[100.0 + i % 7]]) for i in range(600)],
                                  dtype=object).reshape(1, -1),
        "numPulseSample": np.array([[600]]),
    })

    camera_origin = start - 5.0          # camera rolled 5 s before the oximeter
    (tmp_path / "meta.json").write_text(json.dumps({"ppg_origin_epoch": camera_origin}))
    stamps, _ = mrnirp.load_ppg(tmp_path)
    assert stamps[0] == pytest.approx(5.0)

    # With no meta.json the trace falls back to the oximeter's own start.
    (tmp_path / "meta.json").unlink()
    stamps, _ = mrnirp.load_ppg(tmp_path)
    assert stamps[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The split's segment arithmetic against the loader's own expansion
# --------------------------------------------------------------------------- #
def test_segment_counts_matches_the_loader_at_the_project_default_length():
    """The two are separate implementations of one definition.

    `segment_counts` exists so the split can weight by examples without
    materialising them, so it has to track `expand_to_segments` exactly -- at
    whatever the default window is, not only at the length someone once tested.
    """
    from src.aggregation.splits import segment_counts
    from src.model.dataset import DEFAULT_CLIP_FRAMES, TARGET_FPS, expand_to_segments

    manifest = pl.DataFrame({
        "clip_id": [f"c{i}" for i in range(7)],
        "source": ["mrnirp"] * 7,
        "subject_id": [f"s{i}" for i in range(7)],
        "duration_s": [2.0, 10.0, 10.1, 60.0, 120.0, 121.0, 240.0],
        "fps": [30.0] * 7, "box_x": [0] * 7, "box_y": [0] * 7, "box_side": [0] * 7,
        "video_path": [""] * 7, "mask_path": [""] * 7, "hr_bpm": [70.0] * 7,
    })
    for stride in (None, 100, 480, DEFAULT_CLIP_FRAMES):
        expected = (
            expand_to_segments(manifest, stride_frames=stride)
            .group_by("clip_id").len().sort("clip_id")["len"].to_list()
        )
        got = (
            manifest.with_columns(segment_counts(
                DEFAULT_CLIP_FRAMES, TARGET_FPS, stride).alias("n"))
            .sort("clip_id")["n"].to_list()
        )
        assert got == expected, f"stride={stride}: {got} != {expected}"


# --------------------------------------------------------------------------- #
# Resuming onto a cache an older version wrote
# --------------------------------------------------------------------------- #
def _cached_session(tmp_path: Path, clip_id: str, row_extra: dict, side: int):
    """A session directory and frame cache as `prepare` would leave them."""
    from src.model import framecache

    session_dir = tmp_path / "session"
    cache_dir = tmp_path / "cache"
    session_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    row = {k: None for k in mrnirp.ROW_SCHEMA} | {"clip_id": clip_id, "box_side": side}
    row.update(row_extra)
    (session_dir / "meta.json").write_text(json.dumps({"row": row}))
    framecache.sidecar_path(cache_dir, clip_id).write_text(
        json.dumps({"clip_id": clip_id, "side": side, "n_frames": 10, "fps": 30.0})
    )
    return session_dir, cache_dir


def test_a_cached_session_from_before_the_cap_is_rebuilt_not_reused(tmp_path):
    """Orphaned workers from an interrupted run wrote three uncapped sessions.

    They arrived *after* the cap existed, looked complete, and the only symptom
    was a 369 MB window read. Resume has to be able to reject its own cache.
    """
    session_dir, cache_dir = _cached_session(tmp_path, "mrnirp/x", {}, side=640)
    assert mrnirp._reusable_row(session_dir, cache_dir, "mrnirp/x") is None

    session_dir, cache_dir = _cached_session(tmp_path, "mrnirp/x", {}, side=256)
    assert mrnirp._reusable_row(session_dir, cache_dir, "mrnirp/x") is not None


def test_a_cached_session_missing_a_newer_column_is_rebuilt(tmp_path):
    session_dir, cache_dir = _cached_session(tmp_path, "mrnirp/y", {}, side=256)
    meta = json.loads((session_dir / "meta.json").read_text())
    del meta["row"]["fps_source"]
    (session_dir / "meta.json").write_text(json.dumps(meta))
    assert mrnirp._reusable_row(session_dir, cache_dir, "mrnirp/y") is None


def test_a_manifest_row_disagreeing_with_its_cache_is_rebuilt(tmp_path):
    """The row says one box size and the memmap another: one of them is a lie."""
    from src.model import framecache

    session_dir, cache_dir = _cached_session(tmp_path, "mrnirp/z", {}, side=256)
    framecache.sidecar_path(cache_dir, "mrnirp/z").write_text(
        json.dumps({"clip_id": "mrnirp/z", "side": 200, "n_frames": 10, "fps": 30.0})
    )
    assert mrnirp._reusable_row(session_dir, cache_dir, "mrnirp/z") is None


# --------------------------------------------------------------------------- #
# Ingesting the monochrome stream as if it were colour
# --------------------------------------------------------------------------- #
def _mosaic(rng, height: int, width: int, levels=(60, 110, 170)) -> np.ndarray:
    """A BayerBG frame: red at (0,0), green on the anti-diagonal, blue at (1,1)."""
    planes = [rng.normal(v, 3, size=(height, width)).clip(0, 4000) for v in levels]
    out = np.zeros((height, width), dtype=np.uint16)
    for (dy, dx), channel in {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 0}.items():
        out[dy::2, dx::2] = planes[channel][dy::2, dx::2]
    return out


def _mono(rng, height: int, width: int) -> np.ndarray:
    """One detector behind no filter: every position samples the same spectrum."""
    return rng.normal(2000, 3, size=(height, width)).clip(0, 4000).astype(np.uint16)


def test_a_monochrome_stream_is_not_mistaken_for_a_mosaiced_one():
    """MR-NIRP Indoor pairs a mono camera with a colour one. Only RGB is used.

    `Subject2_still_940` ships `cam_flea3_1/` and `RGB/` with no NIR directory,
    and the folder named RGB holds the *mono* frames. Trusting the name ingested
    6310 monochrome frames and demosaiced them as Bayer -- a plausible
    false-colour clip whose channels mean something different from every other
    clip, and nothing raised, because a mono PGM has the shape a mosaiced one has.
    """
    from src.datasets.base import BAYER_MODULATION_MIN, bayer_modulation

    rng = np.random.default_rng(11)
    assert bayer_modulation(_mosaic(rng, 64, 64)) > BAYER_MODULATION_MIN
    assert bayer_modulation(_mono(rng, 64, 64)) < BAYER_MODULATION_MIN
    # The measured populations sit two orders of magnitude apart (29.67% against
    # 0.05%), so the threshold's exact value cannot decide anything.
    assert bayer_modulation(_mosaic(rng, 64, 64)) > 20 * bayer_modulation(_mono(rng, 64, 64))


def _archive(tmp_path: Path, folders: dict[str, np.ndarray], n: int = 4):
    import zipfile

    from src.datasets.base import read_pgm16  # noqa: F401  (shape contract)

    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for folder, frame in folders.items():
            blob = b"P5\n%d %d\n65535\n" % frame.shape[::-1] + frame.astype(">u2").tobytes()
            for i in range(n):
                archive.writestr(f"{folder}/Frame{i:05d}.pgm", blob)
    return zipfile.ZipFile(path)


def test_the_colour_stream_is_chosen_by_measurement_not_by_folder_name(tmp_path):
    """The mosaiced stream wins even when the mono one is the one called RGB."""
    rng = np.random.default_rng(12)
    with _archive(tmp_path, {"S/RGB": _mono(rng, 64, 64),
                             "S/cam_flea3_1": _mosaic(rng, 64, 64)}) as archive:
        names = mrnirp.select_colour_frames(archive)
    assert names and names[0].startswith("S/cam_flea3_1/")


def test_the_standard_layout_still_selects_rgb(tmp_path):
    rng = np.random.default_rng(13)
    with _archive(tmp_path, {"S/NIR": _mono(rng, 64, 64),
                             "S/RGB": _mosaic(rng, 64, 64)}) as archive:
        names = mrnirp.select_colour_frames(archive)
    assert names and names[0].startswith("S/RGB/")


def test_an_archive_with_no_colour_stream_is_refused(tmp_path):
    """Better a skipped session than a monochrome one demosaiced into fake colour."""
    rng = np.random.default_rng(14)
    with _archive(tmp_path, {"S/NIR": _mono(rng, 64, 64)}) as archive:
        assert mrnirp.select_colour_frames(archive) is None


# --------------------------------------------------------------------------- #
# Pooling every corpus into one manifest
# --------------------------------------------------------------------------- #
def _corpus(source: str, n: int, seconds: float, extra: dict | None = None) -> pl.DataFrame:
    base = {
        "clip_id": [f"{source}/{i}" for i in range(n)],
        "source": [source] * n,
        "subject_id": [f"{source}_s{i}" for i in range(n)],
        "video_path": [f"/x/{source}/{i}.avi" for i in range(n)],
        "duration_s": [seconds] * n,
        "fps": [30.0] * n,
        "hr_bpm": [70.0] * n,
    }
    frame = pl.DataFrame(base).with_columns(pl.col("fps").cast(pl.Float32))
    for name, value in (extra or {}).items():
        frame = frame.with_columns(pl.lit(value).alias(name))
    return frame


def test_pooling_reconciles_both_missing_columns_and_mismatched_dtypes(tmp_path):
    """`fps` is Float32 in the manifests `clips` writes and Float64 in MR-NIRP's.

    A plain vertical concat refuses that outright; a hand-rolled column alignment
    accepts it and then picks whichever dtype it saw first.
    """
    from src.aggregation.combine import load_parts

    ubfc = _corpus("ubfc", 4, 60.0)
    mrnirp = _corpus("mrnirp", 3, 120.0, {"corpus": "car"}).with_columns(
        pl.col("fps").cast(pl.Float64), pl.lit("camera_log").alias("fps_source")
    )
    parts = {}
    for name, frame in (("ubfc", ubfc), ("mrnirp", mrnirp)):
        path = tmp_path / f"{name}.parquet"
        frame.write_parquet(path)
        parts[name] = (path,)

    pooled = load_parts(parts)
    assert pooled.height == 7
    assert set(pooled["source"]) == {"ubfc", "mrnirp"}
    # The columns only one corpus has survive, as typed nulls on the other.
    assert pooled.filter(pl.col("source") == "ubfc")["corpus"].null_count() == 4
    assert pooled["corpus"].dtype == pl.String


def test_a_corpus_is_taken_from_exactly_one_file(tmp_path):
    """The MCD manifest also holds UBFC rows, and CLBP rows whose videos are gone.

    Concatenating files and deduplicating would have to pick a winner; selecting
    per source by name does not.
    """
    from src.aggregation.combine import load_parts

    ubfc_only = tmp_path / "ubfc.parquet"
    _corpus("ubfc", 4, 60.0).write_parquet(ubfc_only)
    mixed = tmp_path / "remux.parquet"
    pl.concat([_corpus("mcd", 5, 180.0), _corpus("ubfc", 4, 60.0),
               _corpus("clbp300", 2, 30.0)]).write_parquet(mixed)

    pooled = load_parts({"ubfc": (ubfc_only,), "mcd": (mixed,)})
    assert pooled.group_by("source").len().sort("source").to_dicts() == [
        {"source": "mcd", "len": 5}, {"source": "ubfc", "len": 4},
    ]
    assert "clbp300" not in pooled["source"].to_list()


def test_a_missing_corpus_file_is_skipped_not_fatal(tmp_path):
    """The corpora are built by separate commands; having only some is normal."""
    from src.aggregation.combine import load_parts

    path = tmp_path / "ubfc.parquet"
    _corpus("ubfc", 3, 60.0).write_parquet(path)
    pooled = load_parts({"ubfc": (path,), "mcd": (tmp_path / "absent.parquet",)})
    assert pooled.height == 3


def test_pooling_puts_every_corpus_on_every_side_of_the_split(tmp_path):
    """Unstratified, a greedy fill can hand a whole corpus to one side.

    MR-NIRP's own split did exactly that -- dev was all Car, test all Indoor --
    and pooling three corpora of wildly different size makes it likelier, not
    less: MCD is 98% of the segments.
    """
    from src.aggregation.combine import build

    parts = {}
    for name, n, seconds in (("ubfc", 40, 60.0), ("mrnirp", 18, 120.0),
                             ("mcd", 300, 180.0)):
        path = tmp_path / f"{name}.parquet"
        _corpus(name, n, seconds).write_parquet(path)
        parts[name] = (path,)

    tagged = build(parts)
    present = tagged.group_by("split").agg(pl.col("source").n_unique())
    assert present["source"].min() == 3, "a corpus is missing from a split"
    assert tagged.group_by("subject_id").agg(
        pl.col("split").n_unique())["split"].max() == 1


def test_a_corpus_falls_back_to_the_next_manifest_that_has_it(tmp_path):
    """MCD prefers the remuxed manifest and settles for the plain one.

    The manifests come from different commands, so which exist depends on what
    the user has run. Requiring an exact file makes `combine` fail on a valid
    half-built tree.
    """
    from src.aggregation.combine import load_parts

    plain = tmp_path / "clips.parquet"
    pl.concat([_corpus("mcd", 3, 180.0), _corpus("ubfc", 2, 60.0)]).write_parquet(plain)

    # Remuxed absent: falls through to the plain manifest.
    pooled = load_parts({"mcd": (tmp_path / "remux.parquet", plain)})
    assert pooled.height == 3

    # Remuxed present but holding no MCD rows: also falls through, rather than
    # returning empty because the preferred file happened to exist.
    empty = tmp_path / "remux.parquet"
    _corpus("ubfc", 2, 60.0).write_parquet(empty)
    assert load_parts({"mcd": (empty, plain)}).height == 3


# --------------------------------------------------------------------------- #
# A CLI option read through a default argument
# --------------------------------------------------------------------------- #
def test_samples_seconds_reaches_prepare():
    """`--seconds` was set on a module global that nothing read at call time.

    `make_samples.prepare` took `seconds: float = SECONDS`, and Python binds that
    default at import. The CLI assigned `make_samples.SECONDS = seconds` after
    import, so `main()` called `prepare(video)` and got 5.0 whatever was asked
    for. No error, and the contact sheets looked correct.
    """
    import inspect

    from src.aggregation import make_samples

    # `main` accepts the value rather than reading a module global.
    assert "seconds" in inspect.signature(make_samples.main).parameters
    assert "per_source" in inspect.signature(make_samples.main).parameters
    assert not hasattr(make_samples, "SECONDS")

    # And the value reaches `prepare` on a real call.
    seen: list[float] = []
    original_prepare, original_targets = make_samples.prepare, make_samples._targets
    try:
        make_samples.prepare = lambda video, seconds: seen.append(seconds) or None
        make_samples._targets = lambda per_source=4: [("ubfc", Path("/nonexistent.avi"))]
        make_samples.main(per_source=1, seconds=2.0)
    finally:
        make_samples.prepare = original_prepare
        make_samples._targets = original_targets
    assert seen == [2.0]


# --------------------------------------------------------------------------- #
# `info` reporting a partition no training run uses
# --------------------------------------------------------------------------- #
def test_info_reports_the_split_the_manifest_carries(tmp_path):
    """`info` re-derived an 85/10/5 split and printed that as the split.

    `train` reads the manifest's own `split` column, which `combine` writes at
    90/3/7. The two commands therefore described different partitions of the same
    file, and the one `info` printed was not the one being trained on.
    """
    from click.testing import CliRunner

    from src.cli import cli

    manifest = tmp_path / "m.parquet"
    pl.DataFrame({
        "clip_id": [f"ubfc/{i}" for i in range(10)],
        "subject_id": [f"s{i}" for i in range(10)],
        "source": ["ubfc"] * 10,
        "duration_s": [60.0] * 10,
        "fps": [30.0] * 10,
        "skin_frac": [0.4] * 10,
        "hr_bpm": [72.0] * 10,
        # Deliberately not 85/10/5: 8/1/1 is what must come back.
        "split": ["train"] * 8 + ["dev", "test"],
    }).write_parquet(manifest)

    result = CliRunner().invoke(cli, ["info", "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert f"as {manifest} carries it" in result.output
    # 80/10/10, the column's own ratios. The derived split would give 85/10/5.
    reported = {
        line.split()[0]: line.split()[-1]
        for line in result.output.splitlines()
        if line.startswith(("  train", "  dev", "  test"))
    }
    assert reported == {"train": "80.0%", "dev": "10.0%", "test": "10.0%"}


def test_info_says_so_when_it_derives_the_split(tmp_path):
    """A manifest with no split column gets one derived, and the line says so."""
    from click.testing import CliRunner

    from src.cli import cli

    manifest = tmp_path / "m.parquet"
    pl.DataFrame({
        "clip_id": [f"ubfc/{i}" for i in range(10)],
        "subject_id": [f"s{i}" for i in range(10)],
        "source": ["ubfc"] * 10,
        "duration_s": [60.0] * 10,
        "fps": [30.0] * 10,
        "skin_frac": [0.4] * 10,
        "hr_bpm": [72.0] * 10,
    }).write_parquet(manifest)

    result = CliRunner().invoke(cli, ["info", "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert "carries no split column" in result.output
