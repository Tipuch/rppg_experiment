"""MR-NIRP ingest: the parts that fail silently if they are wrong.

Every test here targets a failure that produces a *trainable* artifact rather
than an exception -- a black frame, a bridged dropout, a red/blue swap, a subject
on both sides of the split. Those are the ones worth pinning.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.aggregation.splits import RATIOS_POOLED, assign, segment_counts
from src.datasets import base, mrnirp


def _pgm(array: np.ndarray) -> bytes:
    height, width = array.shape
    return b"P5\n%d %d\n65535\n" % (width, height) + array.astype(">u2").tobytes()


# --------------------------------------------------------------------------- #
# Reading the frames
# --------------------------------------------------------------------------- #
def test_pgm_round_trips_at_full_16_bit_depth():
    original = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000).astype(np.uint16)
    assert np.array_equal(base.read_pgm16(_pgm(original)), original)


def test_pgm_header_tolerates_comments_and_extra_whitespace():
    body = np.full((4, 4), 4096, dtype=np.uint16)
    blob = b"P5\n# written by FlyCapture\n4  4\n65535\n" + body.astype(">u2").tobytes()
    assert np.array_equal(base.read_pgm16(blob), body)


def test_an_8_bit_pgm_is_refused_rather_than_misread():
    # Reading 8-bit data as >u2 would pair adjacent pixels into garbage that still
    # has the right shape, so this has to raise rather than return an array.
    with pytest.raises(ValueError):
        base.read_pgm16(b"P5\n4 4\n255\n" + bytes(16))


def test_12_bit_left_shifted_data_is_scaled_not_truncated():
    """The observed MR-NIRP layout: 12-bit values shifted up into 16 bits.

    Reducing this by 4 instead of 8 would leave everything above 255 and wrap to
    noise; the detector has to see the empty low nibble and choose 8.
    """
    twelve_bit = np.random.default_rng(0).integers(0, 4096, size=(64, 64))
    frame = (twelve_bit << 4).astype(np.uint16)
    assert base.detect_shift([frame]) == 8
    reduced = frame >> base.detect_shift([frame])
    assert reduced.max() > 200          # uses the range, not squashed to black
    assert reduced.max() <= 255


def test_12_bit_right_aligned_data_is_not_shifted_into_darkness():
    """The other alignment. Shifting by 8 would return an almost black frame.

    A black clip trains quietly to a flat prediction, so this is exactly the kind
    of wrong that never raises.
    """
    frame = np.random.default_rng(1).integers(0, 4096, size=(64, 64)).astype(np.uint16)
    assert base.detect_shift([frame]) == 4
    assert (frame >> 4).max() > 200


def test_genuinely_16_bit_data_is_reduced_by_8():
    frame = np.random.default_rng(2).integers(0, 65536, size=(64, 64)).astype(np.uint16)
    assert base.detect_shift([frame]) == 8


# --------------------------------------------------------------------------- #
# The colour filter array
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern", ["BayerBG", "BayerGB", "BayerRG", "BayerGR"])
def test_the_cfa_search_recovers_the_pattern_a_frame_was_mosaiced_with(pattern):
    """Mosaic a known image, then ask the search to name the tile it used.

    Guessing wrong swaps red with blue, which for rPPG is not cosmetic: the pulse
    is largest in green and the two chroma channels carry different noise.
    """
    rng = np.random.default_rng(3)
    # Skin-like: red above green above blue, plus texture so the parity test has
    # something to measure.
    height = width = 64
    truth = np.stack([
        rng.normal(base_level, 4, size=(height, width)).clip(0, 255)
        for base_level in (60, 110, 170)          # B, G, R
    ], axis=-1).astype(np.uint8)

    # Which sensor position holds which colour, for each OpenCV tile name.
    offsets = {
        "BayerBG": {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        "BayerRG": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2},
        "BayerGB": {(0, 0): 1, (0, 1): 2, (1, 0): 0, (1, 1): 1},
        "BayerGR": {(0, 0): 1, (0, 1): 0, (1, 0): 2, (1, 1): 1},
    }[pattern]
    mosaic = np.zeros((height, width), dtype=np.uint16)
    for (dy, dx), channel in offsets.items():
        mosaic[dy::2, dx::2] = truth[dy::2, dx::2, channel]

    chosen, margins = base.choose_bayer_code(mosaic, 0, None)
    assert chosen == pattern, f"expected {pattern}, got {chosen} (margins {margins})"


def test_the_cfa_test_is_red_above_blue_not_a_full_channel_ordering():
    """Green may outrun red. MR-NIRP Car is IR-lit and it does on every session.

    Requiring R > G > B rejects the correct pattern on 18 of the 24 sessions, so
    the discriminator is R > B alone. This pins that, because "fix" the test to
    the stricter ordering and the whole Car corpus silently demosaics wrong.
    """
    rng = np.random.default_rng(4)
    height = width = 64
    # Green brightest, then red, then blue -- the measured Car situation.
    truth = np.stack([
        rng.normal(level, 3, size=(height, width)).clip(0, 255)
        for level in (40, 150, 90)                # B, G, R
    ], axis=-1).astype(np.uint8)
    mosaic = np.zeros((height, width), dtype=np.uint16)
    for (dy, dx), channel in {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 0}.items():
        mosaic[dy::2, dx::2] = truth[dy::2, dx::2, channel]

    chosen, _ = base.choose_bayer_code(mosaic, 0, None)
    assert chosen == "BayerBG"


# --------------------------------------------------------------------------- #
# The pulse trace
# --------------------------------------------------------------------------- #
def _mat(times: np.ndarray, values: np.ndarray) -> bytes:
    from scipy.io import savemat

    buffer = io.BytesIO()
    savemat(buffer, {
        "pulseOxTime": times.reshape(1, -1),
        # MR-NIRP stores the samples as a cell array of 1x1 numerics.
        "pulseOxRecord": np.array(
            [np.array([[v]]) for v in values], dtype=object
        ).reshape(1, -1),
        "numPulseSample": np.array([[values.size]]),
    })
    return buffer.getvalue()


def test_zero_samples_are_dropped_as_dropouts_not_kept_as_signal():
    """A zero is the oximeter reporting nothing, not a pulse amplitude of zero.

    Every Indoor session has them -- 11 to 233 samples. Left in, they are a
    train of downward spikes at the dropout rate, which is a periodic signal the
    model can learn instead of the pulse.
    """
    times = np.arange(600, dtype=np.float64) / 60.0 + 1_500_000_000.0
    values = 100 + 20 * np.sin(2 * np.pi * 1.2 * (times - times[0]))
    values[100:120] = 0.0

    parsed = mrnirp.parse_pulseox(_mat(times, values))
    assert parsed is not None
    assert (parsed["values"] != 0).all()
    assert parsed["n_pulse"] == 580
    assert parsed["ppg_zero_frac"] == pytest.approx(20 / 600)


def test_the_gap_left_by_a_dropout_is_reported_not_hidden():
    """np.interp will bridge any gap silently. The width has to survive to the row.

    A 20-sample hole at 60 Hz is a third of a second of invented target inside a
    5.3 s window. `ppg_max_gap_s` is what lets a caller judge that, and
    REJECT_GAP_S is where it stops being judgeable.
    """
    times = np.arange(600, dtype=np.float64) / 60.0
    values = np.full(600, 100.0)
    values[300:330] = 0.0

    parsed = mrnirp.parse_pulseox(_mat(times, values))
    assert parsed["ppg_max_gap_s"] == pytest.approx(31 / 60.0, abs=1e-6)


def test_a_trace_that_is_mostly_dropout_is_rejected_by_threshold():
    times = np.arange(600, dtype=np.float64) / 60.0
    values = np.full(600, 100.0)
    values[:400] = 0.0
    parsed = mrnirp.parse_pulseox(_mat(times, values))
    assert parsed["ppg_zero_frac"] > mrnirp.REJECT_ZERO_FRAC


def test_the_time_axis_is_the_oximeters_own_clock_not_an_assumed_rate():
    """The sample rate wanders 32-60 Hz across the corpus, so it cannot be assumed.

    Timestamps stay absolute here; the caller rebases them onto the camera's
    frame 0, which is not always the oximeter's first sample. A reader that
    replaced them with arange/60 would mistime every label on every Car session.
    """
    times = np.cumsum(np.r_[0.0, np.random.default_rng(5).uniform(0.01, 0.04, 599)])
    times += 1_500_000_000.0
    values = 100 + np.sin(np.arange(600))

    parsed = mrnirp.parse_pulseox(_mat(times, values))
    assert np.allclose(np.diff(parsed["times_epoch"]), np.diff(times))
    assert parsed["t0_epoch"] == pytest.approx(times[0])
    assert parsed["span_s"] == pytest.approx(times[-1] - times[0])


def test_the_span_is_measured_before_dropouts_are_removed():
    """A trace ending in dropout still covers the time those samples occupied.

    Measuring the span after filtering shortens it, and every rate derived from
    it comes out too high -- which is exactly the failure `Subject6_still_940`
    shows from the other direction.
    """
    times = np.arange(600, dtype=np.float64) / 60.0 + 1_500_000_000.0
    values = np.full(600, 100.0)
    values[-60:] = 0.0                      # last second is dropout
    parsed = mrnirp.parse_pulseox(_mat(times, values))
    assert parsed["span_s"] == pytest.approx(599 / 60.0)


def test_heart_rate_is_derived_from_the_trace_because_the_corpus_ships_none():
    rate_hz = 1.3
    times = np.arange(3600, dtype=np.float64) / 60.0
    values = 100 + 20 * np.sin(2 * np.pi * rate_hz * times)
    parsed = mrnirp.parse_pulseox(_mat(times + 1_500_000_000.0, values))
    assert mrnirp.clip_hr(parsed) == pytest.approx(rate_hz * 60, abs=1.0)


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #
def _manifest(clip_lengths: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for subject, durations in clip_lengths.items():
        for n, duration in enumerate(durations):
            rows.append({
                "clip_id": f"mrnirp/{subject}_{n}", "source": "mrnirp",
                "subject_id": subject, "duration_s": duration,
            })
    return pl.DataFrame(rows)


def test_no_subject_lands_on_two_sides_of_the_split():
    """The requirement the ratios are allowed to bend for, but this one is not.

    A subject in both train and test scores the model on a face it trained on,
    and MR-NIRP has up to 3 sessions per subject, so this is a live risk here
    rather than a theoretical one.
    """
    manifest = _manifest({f"mrnirp_car_subject{i}": [120.0, 120.0, 240.0]
                          for i in range(15)})
    tagged = assign(
        manifest.with_columns(segment_counts().alias("n_segments")),
        ratios=RATIOS_POOLED, weight="n_segments",
    )
    per_subject = tagged.group_by("subject_id").agg(pl.col("split").n_unique())
    assert per_subject["split"].max() == 1


def test_the_split_fills_by_segment_count_not_by_recording_count():
    """`weight` makes a long recording count for what it is worth in examples.

    Clip length varies 4x across MR-NIRP, so a subject holding one 600 s session
    is worth ten holding 60 s ones. Unweighted, `assign` counts them equally and
    train swallows nearly every subject before its target is met; weighted, three
    long subjects fill most of it, so measurably fewer subjects land in train.

    Averaged over seeds because `assign` fills greedily in a seeded subject order
    -- a single seed says more about that order than about the weighting.
    """
    lengths = {f"mrnirp_car_long{i}": [600.0] for i in range(3)}
    lengths |= {f"mrnirp_car_short{i}": [60.0] for i in range(12)}
    manifest = _manifest(lengths)
    weighted = manifest.with_columns(segment_counts().alias("n_segments"))

    def train_subjects(**kwargs) -> float:
        return float(np.mean([
            assign(weighted, ratios=RATIOS_POOLED, seed=seed, **kwargs)
            .filter(pl.col("split") == "train")["subject_id"].n_unique()
            for seed in range(40)
        ]))

    assert train_subjects(weight="n_segments") < train_subjects() - 1.0


def test_a_three_percent_dev_share_is_unreachable_at_this_corpus_size():
    """Documents the limit rather than pretending the target was met.

    A split that keeps subjects whole cannot cut one in half, so the smallest dev
    share available is one subject's worth of segments. Across MR-NIRP's 15
    subjects that floor sits above the 3% asked for, and the achieved figure is
    the one to quote.
    """
    manifest = _manifest({f"mrnirp_car_subject{i}": [120.0] for i in range(15)})
    weighted = manifest.with_columns(segment_counts().alias("n_segments"))
    total = weighted["n_segments"].sum()
    smallest_subject = weighted.group_by("subject_id").agg(
        pl.col("n_segments").sum()
    )["n_segments"].min()

    assert smallest_subject / total > RATIOS_POOLED["dev"]

    tagged = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments")
    dev = tagged.filter(pl.col("split") == "dev")["n_segments"].sum()
    assert dev / total >= smallest_subject / total


def test_segment_counts_match_the_expansion_the_loader_performs():
    """`segment_counts` is arithmetic standing in for `expand_to_segments`.

    If the two drift apart the split is balanced against a segment count that
    never materialises.
    """
    from src.model.dataset import expand_to_segments

    manifest = _manifest({"mrnirp_car_subject1": [10.0, 60.0, 121.0, 160.0, 240.0]})
    manifest = manifest.with_columns(
        pl.lit(30.0).alias("fps"), pl.lit(0).alias("box_x"), pl.lit(0).alias("box_y"),
        pl.lit(0).alias("box_side"), pl.lit("").alias("video_path"),
        pl.lit("").alias("mask_path"), pl.lit(70.0).alias("hr_bpm"),
    )
    expected = (
        expand_to_segments(manifest, n_frames=160, fps=30.0)
        .group_by("clip_id").len().sort("clip_id")
    )
    got = (
        manifest.with_columns(segment_counts(160, 30.0).alias("n"))
        .select("clip_id", "n").sort("clip_id")
    )
    assert expected["len"].to_list() == got["n"].to_list()


def test_the_split_is_deterministic_across_runs():
    manifest = _manifest({f"mrnirp_car_subject{i}": [120.0] for i in range(15)})
    weighted = manifest.with_columns(segment_counts().alias("n_segments"))
    first = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments", seed=7)
    again = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments", seed=7)
    other = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments", seed=8)
    assert first["split"].to_list() == again["split"].to_list()
    assert first["split"].to_list() != other["split"].to_list()


def test_packing_largest_first_hits_a_target_smaller_than_one_subject():
    """The default fill cannot reach 3% here; largest-first into absolute room can.

    The default picks the split furthest below target as a *fraction* of its own
    target. That is scale-free, so an empty dev bin looks as starved wanting 3%
    as train does wanting 90%, and the first subjects placed land in the smallest
    bins -- the only ones a single subject can overshoot. Measured on the real
    manifest, shuffled order returns 64/14/21 against a 90/3/7 request.
    """
    lengths = {f"mrnirp_car_big{i}": [240.0, 240.0] for i in range(6)}
    lengths |= {f"mrnirp_car_small{i}": [60.0] for i in range(9)}
    manifest = _manifest(lengths)
    weighted = manifest.with_columns(segment_counts().alias("n_segments"))
    total = weighted["n_segments"].sum()

    def shares(**kwargs) -> dict[str, float]:
        tagged = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments", **kwargs)
        got = tagged.group_by("split").agg(pl.col("n_segments").sum()).to_dict(
            as_series=False
        )
        return {k: v / total for k, v in
                zip(got["split"], got["n_segments"], strict=True)}

    packed, shuffled = shares(order="size"), shares()
    error = lambda s: sum(abs(s.get(k, 0) - v) for k, v in RATIOS_POOLED.items())

    assert error(packed) < error(shuffled)
    assert packed["train"] > 0.85


def test_packing_still_keeps_every_subject_whole():
    """Whatever the ordering buys in ratio accuracy, it may not buy it with leakage."""
    lengths = {f"mrnirp_car_big{i}": [240.0, 240.0, 120.0] for i in range(5)}
    lengths |= {f"mrnirp_car_small{i}": [60.0] for i in range(10)}
    manifest = _manifest(lengths)
    weighted = manifest.with_columns(segment_counts().alias("n_segments"))
    tagged = assign(weighted, ratios=RATIOS_POOLED, weight="n_segments", order="size")
    assert tagged.group_by("subject_id").agg(pl.col("split").n_unique())["split"].max() == 1


def test_an_unknown_ordering_is_refused_rather_than_ignored():
    manifest = _manifest({"mrnirp_car_subject1": [120.0]})
    with pytest.raises(ValueError, match="unknown order"):
        assign(manifest, order="smallest")


# --------------------------------------------------------------------------- #
# Attributing the orphaned stream archives
# --------------------------------------------------------------------------- #
def test_the_clock_offset_is_fitted_rather_than_assumed():
    """Zip mtimes record no timezone, and the two corpora were recorded months apart.

    Car is October and Indoor the preceding February, so a constant baked in for
    one is wrong for the other by an hour. The real shift recurs once per genuine
    pair; coincidences do not, so the mode finds it.
    """
    offset = -6 * 3600.0
    pulses = [1_540_000_000.0 + 600 * i for i in range(12)]
    orphans = [p + offset + 0.7 for p in pulses]          # real pairs, with write lag
    orphans += [1_540_050_000.0, 1_540_060_000.0]          # unmatched archives
    assert mrnirp.fit_offset(pulses, orphans) == pytest.approx(offset)


def test_orphans_are_matched_by_start_time_within_tolerance():
    offset = -6 * 3600.0
    sessions = {f"s{i}": 1_540_000_000.0 + 600 * i for i in range(5)}
    orphans = {f"RGB-{i}.zip": v + offset + 0.4 for i, v in enumerate(sessions.values())}
    matched, ambiguous = mrnirp.match_orphans(sessions, orphans)
    assert not ambiguous
    assert matched == {f"s{i}": f"RGB-{i}.zip" for i in range(5)}


def test_an_orphan_two_sessions_could_claim_is_dropped_not_guessed():
    """Guessing here pairs one subject's face with another subject's pulse.

    That trains and scores without raising anything, so an ambiguous match has to
    be refused rather than resolved by nearest-wins.
    """
    offset = -6 * 3600.0
    # Two sessions starting a second apart -- inside the tolerance of one archive.
    sessions = {"a": 1_540_000_000.0, "b": 1_540_000_001.0}
    orphans = {"RGB-1.zip": 1_540_000_000.0 + offset}
    matched, ambiguous = mrnirp.match_orphans(sessions, orphans)
    assert matched == {}
    assert ambiguous


def test_a_session_with_no_orphan_within_tolerance_matches_nothing():
    """12 of the 23 real orphans belong to sessions whose PulseOX never arrived.

    They carry no label, so they must fall out rather than attach to the nearest
    session in the file listing.
    """
    offset = -6 * 3600.0
    sessions = {"a": 1_540_000_000.0}
    orphans = {"RGB-1.zip": 1_540_000_000.0 + offset,
               "RGB-2.zip": 1_540_003_600.0 + offset}
    matched, ambiguous = mrnirp.match_orphans(sessions, orphans)
    assert matched == {"a": "RGB-1.zip"}
    assert "RGB-2.zip" not in matched.values()
    assert not ambiguous


def test_matching_needs_both_sides_and_returns_empty_otherwise():
    assert mrnirp.match_orphans({}, {"RGB-1.zip": 1.0}) == ({}, [])
    assert mrnirp.match_orphans({"a": 1.0}, {}) == ({}, [])


def test_the_loose_indoor_bundle_names_parse_to_subject_and_session():
    """Nine Indoor sessions arrived as top-level files with Drive's batch suffix.

    Missing this pattern is what limited Indoor to 6 of its 15 sessions, and it
    fails silently -- discovery simply returns fewer rows.
    """
    found = mrnirp.LOOSE_RE.match("Subject3_still_940-015.zip")
    assert found and found["session"] == "Subject3_still_940" and found["num"] == "3"

    unsuffixed = mrnirp.LOOSE_RE.match("Subject6_motion_940.zip")
    assert unsuffixed and unsuffixed["session"] == "Subject6_motion_940"

    # Must not swallow the orphan stream archives or the outer batch zips.
    assert mrnirp.LOOSE_RE.match("RGB-027.zip") is None
    assert mrnirp.LOOSE_RE.match("NIR-034.zip") is None
    assert mrnirp.LOOSE_RE.match("MR-NIRP Indoor-20260830T170652Z-1-003.zip") is None


# --------------------------------------------------------------------------- #
# Frame timing
# --------------------------------------------------------------------------- #
def _bundle(tmp_path: Path, n_frames: int,
            log_stamps: dict[str, np.ndarray]) -> zipfile.ZipFile:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n_frames):
            archive.writestr(f"S1/RGB/Frame{i:05d}.pgm", b"")
        for name, stamps in log_stamps.items():
            archive.writestr(f"S1/{name}", "\n".join(f"{v:.6f}" for v in stamps))
    return zipfile.ZipFile(path)


def test_frame_times_come_from_the_log_matching_the_frame_count(tmp_path):
    """Two logs ship per session, for two cameras that differ by a frame or two.

    Picking the wrong one times the colour frames by the mono camera's clock.
    """
    rgb = 1_518_987_793.0 + np.arange(500) / 29.98
    nir = 1_518_987_793.0 + np.arange(640) / 29.98
    with _bundle(tmp_path, 500, {"CameraTimeLog0.txt": nir,
                                 "CameraTimeLog1.txt": rgb}) as archive:
        got = mrnirp.camera_frame_times(archive, 500)
    assert got is not None and got.size == 500
    assert got[0] == pytest.approx(rgb[0])


def test_a_log_that_does_not_describe_this_recording_is_refused(tmp_path):
    stamps = 1_518_987_793.0 + np.arange(50) / 29.98
    with _bundle(tmp_path, 500, {"CameraTimeLog0.txt": stamps}) as archive:
        assert mrnirp.camera_frame_times(archive, 500) is None


def test_no_camera_log_at_all_returns_none_rather_than_guessing(tmp_path):
    """Every Car session is in this position -- its RGB archive holds only frames."""
    with _bundle(tmp_path, 100, {}) as archive:
        assert mrnirp.camera_frame_times(archive, 100) is None


def test_a_truncated_pulse_trace_cannot_inflate_the_frame_rate():
    """`Subject6_still_940`: the oximeter stopped 13.25 s before the camera did.

    Inferring 5811 frames over the pulse's 180.55 s gives 32.19 fps against a
    true 29.98, which slides the label 13 s out by the end of the clip. Without a
    camera log the only defence is that nominal is documented and every
    measurable session sits within 0.05 of it.
    """
    inflated = 5811 / 180.55
    assert abs(inflated - mrnirp.NOMINAL_FPS) > mrnirp.FPS_TOLERANCE

    honest = 5811 / 193.80
    assert abs(honest - mrnirp.NOMINAL_FPS) <= mrnirp.FPS_TOLERANCE


def test_the_cached_box_is_capped_so_a_window_read_stays_proportionate():
    """Indoor boxes reach 707 px, and the cache is read one window at a time.

    Uncapped, a 160-frame Indoor window is 189 MB average and 240 MB at worst,
    against the 7.9 MB the model consumes at 128x128. The cap is what keeps the
    whole training corpus inside page cache instead of eight times over it.
    """
    def window_mb(side: int) -> float:
        return 160 * side * side * 3 / 1e6

    assert window_mb(707) > 200
    assert window_mb(mrnirp.MAX_CACHE_SIDE) < 35
    # UBFC (160-275) and MR-NIRP Car (200-256) sit at or under the cap already,
    # so the native-resolution argument in framecache's docstring is untouched
    # for them -- this only bites where the box is several times the model input.
    assert mrnirp.MAX_CACHE_SIDE >= 256


def test_the_cap_never_enlarges_a_smaller_box():
    """Enlarging with INTER_AREA is identical to INTER_NEAREST.

    It would duplicate pixels and then average them back down, which costs
    precision on a signal that lives at 0.1-0.5 LSB.
    """
    for side in (160, 200, 256, 475, 707):
        assert min(side, mrnirp.MAX_CACHE_SIDE) <= side
