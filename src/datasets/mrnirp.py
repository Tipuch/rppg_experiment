"""MR-NIRP: Bayer PGM sequences in nested zips -> the standard frame cache.

    uv run python -m src.cli mrnirp --downloads ~/Downloads

MR-NIRP ships no video. Each recording is a directory of 640x640 16-bit PGM
frames inside a zip, inside another zip, beside a MATLAB file holding the pulse
oximeter trace. Nothing in this repo's pixel path can read that -- `video.py` and
`framecache.py` both shell out to ffmpeg against a single container -- so this
reader does what `clips.build_clip` and `framecache.build` do for the other
corpora, in one pass, and writes the *same* artifacts: a `.raw` box dump, its
JSON sidecar, a `_ppg.npz` trace and a packed skin mask. `WindowDataset` then
reads MR-NIRP through the path it already uses for UBFC, with no new branch and
nothing decoded at training time.

## What the source actually is, measured rather than assumed

**The frames are 12-bit, left-shifted into a 16-bit container.** The header says
`maxval 65535`, but the low nibble is zero in every frame sampled and the largest
value seen is 65520 = 4095 << 4. `base.detect_shift` re-checks per session.

**The RGB stream is raw Bayer with no stated pattern**, which is why DATASETS.md
first recorded it as unusable. It is recoverable, and `base.choose_bayer_code`
recovers it: both corpora come out `BayerBG`, by green-parity and by red-above-
blue on the face.

**The pulse oximeter's rate is neither constant nor nominal** -- 32.0 to 59.9 Hz
across these 24 sessions. `pulseOxTime` carries epoch seconds per sample and is
the only trustworthy time axis, as `gtdump.xmp` column 0 is for UBFC DATASET_1.

**The frame rate is measured, not read.** `Setup0.log` claims 30 fps for every
recording; frame count against the pulse span gives 30.05 on Car. The log happens
to be right, but the manifest carries the measured value.

**Zero samples in the pulse trace are dropouts, not measurements.** No Car session
has any. *Every* Indoor session does -- 11 to 233 samples, worst
`Subject2_motion_940` with a 69-sample flat run. They are dropped before anything
is derived, and a session whose trace is too broken to trust is rejected outright.

## Why 44 sessions and not all of them

Google Drive split both downloads by size rather than by structure, and it did so
in three different ways. Discovery has to undo all three or it silently ingests a
fraction of what is on disk:

  1. **Paired inner archives.** A Car session's `RGB.zip` and `PulseOX.zip` are
     members of unrelated outer zips. 56 sessions appear this way; 40 have a
     pulse trace and 34 have RGB, but only **18 have both**.
  2. **Orphaned streams.** Anything too large to batch came down alone as
     `RGB-###.zip` or `NIR-###.zip`, carrying no subject, no session and no
     attribution -- an archive holds `RGB/Frame00000.pgm` and nothing else.
     `match_orphans` recovers the 23 orphan RGB archives from timestamps, which
     adds **11 more Car sessions**, for 29 over 10 subjects.
  3. **Loose bundles.** Nine Indoor sessions arrived as top-level
     `Subject3_still_940-015.zip` rather than inside an `MR-NIRP Indoor-*.zip`.
     With the 6 that did, Indoor is complete: **15 sessions over 8 subjects**.

The 12 orphan RGB archives that match nothing are sessions whose PulseOX was
never downloaded. They carry no label and are dropped.

## How the orphans are attributed

`pulseOx.mat` carries `pulseOxTime`, the oximeter's own epoch clock, and every
PGM kept the zip mtime it was written with. Both come from one recording, so a
session's RGB archive is the one whose frames start when its pulse trace starts.

The two clocks differ by a whole-hour timezone offset, which is **fitted, not
assumed**: zip mtimes are naive local time with no zone recorded, and Car was
recorded in October while Indoor was recorded the preceding February. Every
candidate pair votes on a rounded hour and the mode wins -- measured, -6 h with
92 votes against 37 for the runner-up. Matches are then required to be **unique
in both directions**: an archive claimed by two sessions, or a session claiming
two archives, is reported and dropped rather than guessed. Getting this wrong
pairs one subject's face with another subject's pulse, which trains and evaluates
without ever raising.

The result is checked by two things the matcher never looks at: all 11 matches
land within 1.0 s of the fitted offset, and each implies 30.04-30.05 fps against
its own pulse span, which is the rate every `Setup0.log` claims.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from ..aggregation.face import BOX_PAD, median_face_box
from ..aggregation.skin import median_skin_mask
from ..model import framecache
from ..model.clips import crop_and_resize
from ..model.waveform import hr_from_waveform
from ..paths import BUILD_ROOT, DATA_ROOT
from .base import (
    BAYER_CANDIDATES,
    BAYER_MODULATION_MIN,
    HR_VALID,
    bayer_modulation,
    choose_bayer_code,
    detect_shift,
    read_pgm16,
    sessions_from,
    to_bgr,
)

NAME = "mrnirp"

# The manifest's columns, declared rather than inferred. `build` assembles rows
# from sessions that may have been cached by an older version of this module, and
# polars infers a DataFrame's schema from the first 100 rows: one missing key
# across the first 100 gives that column dtype Null, and the first real value
# later in the frame then fails to append. Declaring the schema makes a stale row
# come back as a null of the right type instead of poisoning the column.
ROW_SCHEMA: dict[str, pl.DataType] = {
    "clip_id": pl.String,
    "source": pl.String,
    "subject_id": pl.String,
    "video_path": pl.String,
    "fps": pl.Float64,
    "n_frames": pl.Int64,
    "duration_s": pl.Float64,
    "box_x": pl.Int64,
    "box_y": pl.Int64,
    "box_side": pl.Int64,
    "mask_path": pl.String,
    "skin_frac": pl.Float64,
    "hr_bpm": pl.Float64,
    "sbp_mmhg": pl.Float64,
    "dbp_mmhg": pl.Float64,
    "corpus": pl.String,
    "ppg_zero_frac": pl.Float64,
    "ppg_max_gap_s": pl.Float64,
    "fps_source": pl.String,
}
DATASET_DIR = DATA_ROOT / "mr-nirp"
MASK_DIR = BUILD_ROOT / "masks"
DETECT_FRAMES = 24

# The corpus-wide answer from choose_bayer_code. Sessions are still checked
# individually and a disagreement is reported rather than silently demosaiced.
BAYER_DEFAULT = "BayerBG"

# A trace is rejected rather than trained on when it fails either of these. The
# thresholds sit clear of what these 24 sessions measure -- the worst is 6.4%
# dropout and a 1.15 s gap -- so they reject a *new* broken session rather than
# trimming the corpus that exists.
REJECT_ZERO_FRAC = 0.20
REJECT_GAP_S = 2.0

# Both cameras log 30 fps and every session that can be measured exactly comes
# out 29.955-29.979. Car ships no camera log, so its rate is inferred from the
# pulse span -- which is only valid while the two recordings cover the same
# window. `Subject6_still_940` proves they need not: its oximeter stopped 13.25 s
# early, and the inferred rate was 32.19 against a true 29.98, which would slide
# the label 13 s out by the end of the clip. A session whose inferred rate misses
# nominal by more than the tolerance falls back to nominal and says so.
NOMINAL_FPS = 30.0
FPS_TOLERANCE = 1.0

# Largest face box written to the cache. Indoor recordings are already framed
# tight on the head, so YuNet returns boxes of 475-707 px -- and the cache is read
# at training time, one window at a time. Uncapped, one 160-frame Indoor window is
# **189 MB average and 240 MB at worst**, against the 7.9 MB the model actually
# consumes at 128x128. That is 24x the pixels, off an encrypted volume, every item.
#
# framecache's docstring argues for native resolution, on the grounds that caching
# smaller puts a resample *before* WindowDataset's augmentation crop. That holds
# where the box is near the model's own size -- UBFC's are 160-275 px and Car's
# 200-256, so neither is touched here. It stops holding at 700 px: the pipeline
# already area-resamples that down to 128, so capping replaces a per-epoch
# downscale with one at build time, and removes 6x the disk and 6x the read.
MAX_CACHE_SIDE = 256

# Zip mtimes carry 2-second resolution, so a match cannot be tighter than that.
# The 11 real matches land within 1.0 s of the fitted offset; the nearest
# non-matching take is minutes away, so 5 s separates them with room to spare.
MATCH_TOLERANCE_S = 5.0

CAR_RE = re.compile(
    r"MR-NIRP Car/(?P<subject>Subject\d+)/(?P<session>[^/]+)/(?P<stream>RGB|PulseOX)\.zip$"
)
INDOOR_RE = re.compile(r"MR-NIRP Indoor/(?P<session>Subject(?P<num>\d+)_[^/]+)\.zip$")
# Nine Indoor sessions arrived as top-level files instead, with Drive's batch
# number appended: `Subject3_still_940-015.zip`.
LOOSE_RE = re.compile(r"^(?P<session>Subject(?P<num>\d+)_[a-z]+_\d+)(?:-\d+)?\.zip$")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _car_members(downloads: Path) -> pl.DataFrame:
    """Every Car inner archive found, one row per (session, stream)."""
    rows = []
    for outer in sorted(downloads.glob("MR-NIRP Car-*.zip")):
        with zipfile.ZipFile(outer) as archive:
            for info in archive.infolist():
                found = CAR_RE.match(info.filename)
                if found:
                    rows.append({
                        "session": found["session"],
                        "subject": found["subject"].lower(),
                        "stream": found["stream"].lower(),
                        "outer": str(outer),
                        "inner": info.filename,
                    })
    return pl.DataFrame(rows, schema={
        "session": pl.String, "subject": pl.String, "stream": pl.String,
        "outer": pl.String, "inner": pl.String,
    })


def _zip_start(path: Path) -> tuple[int, float] | None:
    """Frame count and first-frame mtime of a standalone stream archive.

    Central directory only -- these are 2.5-10 GB files and nothing is inflated.
    The mtime is naive local time read as if it were UTC; `fit_offset` supplies
    the shift.
    """
    with zipfile.ZipFile(path) as archive:
        frames = sorted(
            (i for i in archive.infolist() if i.filename.endswith(".pgm")),
            key=lambda i: i.filename,
        )
    if not frames:
        return None
    # Read as UTC on purpose: the stamp is naive local time and the real shift
    # is what fit_offset recovers. Anchoring it here keeps that one variable.
    stamp = datetime(*frames[0].date_time, tzinfo=UTC)
    return len(frames), stamp.timestamp()


def fit_offset(pulse_starts: list[float], orphan_starts: list[float]) -> float:
    """The whole-hour shift between naive zip mtimes and the oximeter's UTC clock.

    Fitted rather than hardcoded. Zip mtimes record no timezone, and MR-NIRP Car
    was recorded in October while Indoor was recorded the preceding February, so
    a constant baked in for one is wrong for the other.

    Every session-orphan pair votes for a rounded hour. The true shift recurs once
    per real recording and coincidences do not, so the mode wins. Rounding before
    the vote is what makes it work: real pairs differ by a second or two of write
    lag and would otherwise never land on the same value.
    """
    votes: Counter[int] = Counter()
    for start in pulse_starts:
        for orphan in orphan_starts:
            delta = orphan - start
            if abs(delta) <= 24 * 3600:
                votes[round(delta / 3600.0)] += 1
    if not votes:
        raise ValueError("no candidate offsets within a day of each other")
    return votes.most_common(1)[0][0] * 3600.0


def match_orphans(
    sessions: dict[str, float], orphans: dict[str, float]
) -> tuple[dict[str, str], list[str]]:
    """Pair sessions with orphan stream archives by start time.

    `sessions` maps name -> pulse start (UTC epoch); `orphans` maps path -> first
    frame mtime. Returns the unambiguous matches and a list of the ambiguities it
    refused to guess at.

    A match must be unique in **both** directions. Two sessions claiming one
    archive, or one session claiming two, means the timestamps cannot tell them
    apart -- and picking either would pair one subject's face with another
    subject's pulse, which trains and scores without raising.
    """
    if not sessions or not orphans:
        return {}, []
    offset = fit_offset(list(sessions.values()), list(orphans.values()))

    pairs = [
        (abs(start - offset - pulse), name, path)
        for name, pulse in sessions.items()
        for path, start in orphans.items()
        if abs(start - offset - pulse) <= MATCH_TOLERANCE_S
    ]
    by_session: dict[str, set[str]] = {}
    by_orphan: dict[str, set[str]] = {}
    for _, name, path in pairs:
        by_session.setdefault(name, set()).add(path)
        by_orphan.setdefault(path, set()).add(name)

    ambiguous = [
        f"{name} <- {sorted(paths)}" for name, paths in by_session.items() if len(paths) > 1
    ] + [
        f"{Path(path).name} -> {sorted(names)}"
        for path, names in by_orphan.items() if len(names) > 1
    ]
    matched = {
        name: path for _, name, path in sorted(pairs)
        if len(by_session[name]) == 1 and len(by_orphan[path]) == 1
    }
    return matched, ambiguous


def _pulse_start(outer: str, inner: str) -> float | None:
    """The oximeter start time for one session, read without extracting frames."""
    try:
        with zipfile.ZipFile(outer) as archive, archive.open(inner) as handle, \
                zipfile.ZipFile(handle) as inner_zip:
            name = next(
                (n for n in inner_zip.namelist() if n.endswith("pulseOx.mat")), None
            )
            if name is None:
                return None
            parsed = parse_pulseox(inner_zip.read(name))
    except (zipfile.BadZipFile, OSError):
        return None
    return None if parsed is None else parsed["t0_epoch"]


def _recover_orphan_rgb(wide: pl.DataFrame, downloads: Path) -> pl.DataFrame:
    """Fill in `outer_rgb` for sessions whose colour stream arrived unattributed.

    Their pulse trace is here but their frames came down as a bare `RGB-###.zip`.
    `inner_rgb` stays null for these: the archive *is* the file, so there is no
    member to open inside it.
    """
    needing = wide.filter(pl.col("outer_rgb").is_null())
    if not needing.height:
        return wide

    orphans = {}
    for path in sorted(downloads.glob("RGB-*.zip")):
        found = _zip_start(path)
        if found:
            orphans[str(path)] = found[1]

    starts = {
        row["session"]: start
        for row in needing.to_dicts()
        if (start := _pulse_start(row["outer_pulseox"], row["inner_pulseox"])) is not None
    }
    matched, ambiguous = match_orphans(starts, orphans)
    for entry in ambiguous:
        print(f"  AMBIGUOUS, dropped: {entry}")
    if not matched:
        return wide

    print(f"  recovered {len(matched)} orphan RGB archives by timestamp")
    recovered = pl.DataFrame({"session": list(matched), "outer_rgb": list(matched.values())})
    return (
        wide.join(recovered, on="session", how="left")
        .with_columns(pl.coalesce("outer_rgb", "outer_rgb_right").alias("outer_rgb"))
        .drop("outer_rgb_right")
    )


def _car_sessions(downloads: Path) -> pl.DataFrame:
    """Car sessions holding both streams, after recovering the orphaned ones."""
    members = _car_members(downloads)
    if not members.height:
        return pl.DataFrame(schema={"clip_id": pl.String})

    # The pairing that decides the corpus size, as a join rather than a pile of
    # dict lookups: a session is usable exactly where both streams landed.
    wide = members.pivot(
        on="stream", index=["session", "subject"],
        values=["outer", "inner"], aggregate_function="first",
    )
    for column in ("outer_rgb", "inner_rgb", "outer_pulseox", "inner_pulseox"):
        if column not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.String).alias(column))
    wide = _recover_orphan_rgb(
        wide.drop_nulls(["outer_pulseox", "inner_pulseox"]), downloads
    )

    return wide.drop_nulls(["outer_rgb"]).select(
        clip_id=pl.lit(f"{NAME}/car_") + pl.col("session"),
        session=pl.col("session"),
        corpus=pl.lit("car"),
        source=pl.lit(NAME),
        subject_id=pl.lit(f"{NAME}_car_") + pl.col("subject"),
        rgb_outer=pl.col("outer_rgb"),
        rgb_inner=pl.col("inner_rgb").fill_null(""),
        pulse_outer=pl.col("outer_pulseox"), pulse_inner=pl.col("inner_pulseox"),
    )


def _indoor_sessions(downloads: Path, schema) -> pl.DataFrame:
    """Indoor sessions, from outer zips and from top-level bundles alike.

    Both layouts are self-contained -- one archive holds RGB, NIR and PulseOX --
    so the two stream references point at the same place. Nine of the fifteen
    arrived loose, and missing that pattern is what limited Indoor to six.
    """
    rows = []
    for outer in sorted(downloads.glob("MR-NIRP Indoor-*.zip")):
        with zipfile.ZipFile(outer) as archive:
            for info in archive.infolist():
                found = INDOOR_RE.match(info.filename)
                if found:
                    rows.append((found["session"], found["num"], str(outer), info.filename))
    for path in sorted(downloads.glob("Subject*.zip")):
        found = LOOSE_RE.match(path.name)
        if found:
            rows.append((found["session"], found["num"], str(path), ""))

    return pl.DataFrame([
        {
            "clip_id": f"{NAME}/indoor_{session}",
            "session": session,
            "corpus": "indoor",
            "source": NAME,
            "subject_id": f"{NAME}_indoor_subject{num}",
            "rgb_outer": outer, "rgb_inner": inner,
            "pulse_outer": outer, "pulse_inner": inner,
        }
        for session, num, outer, inner in rows
    ], schema=schema)


def discover(downloads: Path | None = None) -> pl.DataFrame:
    """Sessions with both a colour stream and a pulse trace, from zip indexes only.

    Central directories are all that is read here -- nothing is extracted and no
    pixels are decoded, so this runs in seconds over ~245 GiB of archives.
    """
    downloads = Path(downloads or Path.home() / "Downloads")
    if not downloads.is_dir():
        return sessions_from([])

    car = _car_sessions(downloads)
    indoor = _indoor_sessions(downloads, car.schema if car.height else None)
    parts = [f for f in (car, indoor) if f.height]
    if not parts:
        return sessions_from([])

    return (
        pl.concat(parts, how="vertical")
        .with_columns(
            video_path=pl.lit(str(DATASET_DIR) + "/") + pl.col("corpus") + "/" + pl.col("session")
        )
        .sort("clip_id")
    )


def select_colour_frames(archive: zipfile.ZipFile) -> list[str] | None:
    """The mosaiced stream's frames, chosen by measurement rather than by name.

    A bundle holds two streams and the obvious rule is to take the one in `RGB/`.
    That is wrong on at least one session: `Subject2_still_940` ships
    `cam_flea3_1/` and `RGB/`, with **no NIR directory at all**, and its `RGB/`
    frames modulate 0.05% of their mean while `cam_flea3_1/` modulates 29.67%.
    The folder labelled RGB is the monochrome camera; the colour frames are under
    the other name.

    Taking it on trust ingested 6310 monochrome frames and demosaiced them as if
    they were Bayer, producing a plausible false-colour clip whose channels mean
    something different from every other clip in the corpus -- and nothing raised,
    because a mono PGM has exactly the shape a mosaiced one does.

    So each directory is sampled and the most strongly mosaiced one wins. Returns
    None when no stream looks like colour at all.
    """
    frames: dict[str, list[str]] = {}
    for name in archive.namelist():
        if name.endswith(".pgm"):
            frames.setdefault(name.rsplit("/", 1)[0], []).append(name)
    if not frames:
        return None

    scored = []
    for folder, names in frames.items():
        names.sort()
        try:
            sample = read_pgm16(archive.read(names[len(names) // 2]))
        except (ValueError, KeyError):
            continue
        scored.append((bayer_modulation(sample), folder, names))
    if not scored:
        return None

    best, folder, names = max(scored)
    if best < BAYER_MODULATION_MIN:
        return None
    if len(scored) > 1:
        runner = sorted(scored, reverse=True)[1]
        if runner[1].endswith("RGB") and not folder.endswith("RGB"):
            print(f"  note: colour frames are in {folder!r} "
                  f"({100 * best:.1f}% modulation), not {runner[1]!r} "
                  f"({100 * runner[0]:.2f}%)")
    return names


def parse_pulseox(blob: bytes) -> dict | None:
    """The pulse trace plus the statistics that decide whether it is usable.

    `pulseOxRecord` is a genuine PPG waveform, not a heart-rate readout -- it
    oscillates at the cardiac rate (76.5 bpm on the first session checked, against
    a 55.5 Hz sample rate), so it can supervise per frame.

    Zero samples are removed as dropouts, which leaves gaps in the time axis that
    `np.interp` later bridges. `ppg_max_gap_s` is what says whether that bridge is
    short enough to be honest.
    """
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(blob))
    if "pulseOxTime" not in mat or "pulseOxRecord" not in mat:
        return None
    times = np.asarray(mat["pulseOxTime"], dtype=np.float64).ravel()
    # A MATLAB cell array: one 1x1 numeric array per sample.
    values = np.array(
        [float(np.asarray(v).ravel()[0]) for v in mat["pulseOxRecord"].ravel()],
        dtype=np.float64,
    )
    if times.size < 2 or values.size != times.size:
        return None

    # Span is taken across the whole recording, before dropouts are removed. A
    # trace that begins or ends in dropout would otherwise report a shorter span
    # than it covers, and every rate derived from it would be too high.
    first, last = float(times[0]), float(times[-1])

    keep = values != 0.0
    zero_frac = float((~keep).mean())
    times, values = times[keep], values[keep]
    if times.size < 2:
        return None
    return {
        # Absolute epoch seconds. The caller rebases these onto the camera's
        # frame 0, which is not always the oximeter's first sample.
        "times_epoch": times,
        "values": values,
        "t0_epoch": first,
        "span_s": last - first,
        "n_pulse": int(times.size),
        "ppg_zero_frac": zero_frac,
        "ppg_max_gap_s": float(np.diff(times).max()),
    }


def camera_frame_times(archive: zipfile.ZipFile, n_frames: int) -> np.ndarray | None:
    """Per-frame capture times for the colour camera, from the session's own log.

    Indoor bundles ship `CameraTimeLog{0,1}.txt`, one epoch-second stamp per
    frame: index 0 is the NIR camera and index 1 the colour one, matching
    `Setup{0,1}.log`. Rather than trust that numbering, the log whose length is
    closest to the frame count wins -- the two differ by a frame or two and the
    ambiguity is not worth a silent mismatch.

    This is the only exact time axis available. Car sessions ship no logs at all.
    """
    logs = [n for n in archive.namelist() if "CameraTimeLog" in n]
    best: np.ndarray | None = None
    for name in logs:
        try:
            stamps = np.array(
                [float(x) for x in
                 re.findall(r"\d+\.\d+", archive.read(name).decode("utf8", "replace"))],
                dtype=np.float64,
            )
        except (KeyError, ValueError):
            continue
        if stamps.size < 2:
            continue
        if best is None or abs(stamps.size - n_frames) < abs(best.size - n_frames):
            best = stamps
    # A log that disagrees with the frame count by more than a handful of frames
    # is describing a different recording, not this one.
    if best is None or abs(best.size - n_frames) > 8:
        return None
    return best


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contact PPG for a prepared session directory, or None.

    `prepare` copies each session's `pulseOx.mat` into the project tree precisely
    so this works: the frame cache holds the parsed trace, and this is what
    re-derives it if that cache is ever rebuilt.
    """
    mat = clip_dir / "pulseOx.mat"
    if not mat.exists():
        return None
    parsed = parse_pulseox(mat.read_bytes())
    if parsed is None:
        return None
    # Rebased onto the camera's frame 0, which `prepare` recorded, because the
    # oximeter's first sample is not always the video's first frame. Falling back
    # to the oximeter's own start reproduces the pre-camera-log behaviour.
    meta = clip_dir / "meta.json"
    origin = parsed["t0_epoch"]
    if meta.exists():
        origin = float(json.loads(meta.read_text()).get("ppg_origin_epoch", origin))
    return parsed["times_epoch"] - origin, parsed["values"]


def clip_hr(pulse: dict) -> float:
    """Heart rate from the trace itself. MR-NIRP ships no HR column."""
    times = pulse["times_epoch"] - pulse["times_epoch"][0]
    even = np.arange(0.0, float(times[-1]), 1.0 / framecache.TARGET_FPS)
    return hr_from_waveform(
        np.interp(even, times, pulse["values"]), fps=framecache.TARGET_FPS
    )


# --------------------------------------------------------------------------- #
# Preparing one session
# --------------------------------------------------------------------------- #
def _extract_member(outer: str, inner: str, destination: Path) -> Path:
    """Copy one inner archive out so its members can be random-accessed.

    Necessary, not incidental. The inner archives are deflated members of the
    outer zip, so seeking to frame 1800 through a nested ZipFile re-inflates
    everything before it -- quadratic over 3633 frames. Copied out first, the same
    access is a seek. Outer members store at ~1.0 ratio, so this is a 2 second
    copy of already-compressed bytes, not a decompression.
    """
    if not inner:
        # An orphan or loose bundle: the archive already *is* a standalone file on
        # disk, so it can be opened where it lies. Nothing to copy, nothing to
        # clean up, and no second copy of a 2.5-10 GB archive.
        return Path(outer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer) as archive, archive.open(inner) as source, \
            destination.open("wb") as sink:
        shutil.copyfileobj(source, sink, 1 << 20)
    return destination


def _reusable_row(session_dir: Path, cache_dir: Path, clip_id: str) -> dict | None:
    """The manifest row for an already-cached session, or None to rebuild it.

    Resuming has to be able to reject its own cache. A session cached by an older
    version of this module is not merely out of date -- it is a row that looks
    finished and is wrong, and the manifest gives no sign of it. Three sessions
    reached the manifest at 475-640 px after the cap was introduced, written by
    workers orphaned from an interrupted run, and the only symptom was a window
    read of 369 MB.

    So the cached artifacts are checked against what this version would produce:
    every column present, and a stored box within the cap.
    """
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        row = json.loads(meta_path.read_text())["row"]
    except (json.JSONDecodeError, KeyError):
        return None
    if set(ROW_SCHEMA) - set(row):
        return None                                   # written before a column existed
    if int(row.get("box_side") or 0) > MAX_CACHE_SIDE:
        return None                                   # written before the cap
    if int(row.get("box_side") or 0) != int(
        json.loads(framecache.sidecar_path(cache_dir, clip_id).read_text())["side"]
    ):
        return None                                   # manifest and cache disagree
    return row


@dataclass(frozen=True)
class _Geometry:
    """Everything about a session's pixels that is decided once, from a sample."""

    box: tuple[int, int, int]        # x, y, side, in source pixels
    store_side: int                  # side actually written, after MAX_CACHE_SIDE
    shift: int                       # bits dropped to reach 8-bit
    bayer: str                       # CFA tile name
    margins: dict[str, float]        # red-minus-blue per surviving candidate
    mask: np.ndarray                 # boolean skin mask at MASK_RES


def _read_labels(
    session: dict, scratch: Path, clip_id: str
) -> tuple[dict, bytes, Path] | None:
    """The pulse trace, its bytes, and the archive it came from -- or None.

    Runs before any frame is touched, because every rejection here is cheap and
    the alternative is inflating several GB to then discard it.
    """
    pulse_zip = _extract_member(
        session["pulse_outer"], session["pulse_inner"], scratch / "pulse.zip"
    )
    with zipfile.ZipFile(pulse_zip) as archive:
        name = next((n for n in archive.namelist() if n.endswith("pulseOx.mat")), None)
        if name is None:
            print(f"  skip {clip_id}: no pulseOx.mat")
            return None
        blob = archive.read(name)

    pulse = parse_pulseox(blob)
    if pulse is None:
        print(f"  skip {clip_id}: pulse trace unreadable")
        return None
    if pulse["ppg_zero_frac"] > REJECT_ZERO_FRAC:
        print(f"  skip {clip_id}: {100 * pulse['ppg_zero_frac']:.1f}% of the pulse "
              "trace is dropout")
        return None
    if pulse["ppg_max_gap_s"] > REJECT_GAP_S:
        print(f"  skip {clip_id}: {pulse['ppg_max_gap_s']:.2f}s gap in the pulse trace")
        return None
    return pulse, blob, pulse_zip


def _frame_rate(
    archive: zipfile.ZipFile, n_frames: int, pulse: dict, clip_id: str
) -> tuple[float, float, str]:
    """Frame rate, the epoch of frame 0, and which source supplied them.

    Best source first. The camera log is exact -- one stamp per frame on the same
    clock as the oximeter. Failing that, the pulse span, which is only valid while
    both recordings cover the same window, and is therefore checked against
    nominal: `Subject6_still_940`'s oximeter stopped 13.25 s early and the span
    implied 32.19 fps against a true 29.98.
    """
    stamps = camera_frame_times(archive, n_frames)
    if stamps is not None:
        return (stamps.size - 1) / (stamps[-1] - stamps[0]), float(stamps[0]), "camera_log"

    fps = n_frames / pulse["span_s"]
    if abs(fps - NOMINAL_FPS) > FPS_TOLERANCE:
        print(f"  WARNING {clip_id}: pulse span implies {fps:.2f} fps, outside "
              f"{NOMINAL_FPS} +/- {FPS_TOLERANCE}. The oximeter and the camera did "
              "not cover the same window; falling back to nominal.")
        return NOMINAL_FPS, pulse["t0_epoch"], "nominal_fallback"
    return fps, pulse["t0_epoch"], "pulse_span"


def _geometry(
    archive: zipfile.ZipFile, names: list[str], clip_id: str
) -> _Geometry | None:
    """Decide the CFA tile, bit shift, face box and skin mask from sampled frames.

    All four are per-clip constants, and all four are measured rather than
    configured -- see the module docstring for why each one cannot be assumed.
    """
    picks = [names[i] for i in np.linspace(0, len(names) - 1, DETECT_FRAMES).astype(int)]
    raws = [read_pgm16(archive.read(n)) for n in picks]
    shift = detect_shift(raws)
    middle = raws[len(raws) // 2]

    # First pass over the whole frame, which is all there is until a box exists.
    chosen, _ = choose_bayer_code(middle, shift, None)
    sample = np.stack([to_bgr(r, shift, BAYER_CANDIDATES[chosen]) for r in raws])

    box = median_face_box(list(sample), pad=BOX_PAD)
    if box is None:
        print(f"  skip {clip_id}: no face detected in {DETECT_FRAMES} frames")
        return None

    # Second pass on the face box. The first included the garage on Car sessions,
    # where the background is what the red-above-blue test would have measured.
    chosen, margins = choose_bayer_code(middle, shift, box)
    if chosen != BAYER_DEFAULT:
        print(f"  WARNING {clip_id}: pixels prefer {chosen}, not the corpus-wide "
              f"{BAYER_DEFAULT}. margins={margins}")
    sample = np.stack([to_bgr(r, shift, BAYER_CANDIDATES[chosen]) for r in raws])

    # Square-crop then resize to CROP, exactly as clips.build_clip does: SegFace
    # segments at MASK_RES and WindowDataset maps the mask back through that same
    # 256-pixel frame, so a mask built at any other size lands on the wrong pixels
    # without raising.
    mask = median_skin_mask(np.ascontiguousarray(crop_and_resize(sample, box)[:, :, :, ::-1]))
    if not mask.any():
        print(f"  skip {clip_id}: skin segmentation returned nothing")
        return None

    return _Geometry(
        box=(box[0], box[1], box[2]), store_side=min(box[2], MAX_CACHE_SIDE),
        shift=shift, bayer=chosen, margins=margins, mask=mask,
    )


def _write_frames(
    archive: zipfile.ZipFile, names: list[str], geom: _Geometry, destination: Path
) -> int:
    """Stream every frame's crop to `destination`. Returns the count written.

    Frame by frame: one session is 0.6-2.3 GB, and building 44 back to back would
    keep handing the allocator a fresh gigabyte.
    """
    x, y, side = geom.box
    code = BAYER_CANDIDATES[geom.bayer]
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as sink:
        for name in names:
            frame = to_bgr(read_pgm16(archive.read(name)), geom.shift, code)
            cropped = frame[y : y + side, x : x + side]
            if geom.store_side != side:
                # INTER_AREA, and only ever a downscale. Enlarging with it is
                # identical to INTER_NEAREST, which would duplicate pixels and
                # then average them back down.
                cropped = cv2.resize(cropped, (geom.store_side, geom.store_side),
                                     interpolation=cv2.INTER_AREA)
            sink.write(np.ascontiguousarray(cropped).tobytes())
            written += 1
    return written


def prepare(
    session: dict,
    scratch: Path,
    cache_dir: Path = framecache.CACHE_DIR,
    dataset_dir: Path = DATASET_DIR,
    force: bool = False,
) -> dict | None:
    """Extract, demosaic, crop and cache one session. Returns its manifest row.

    None means unusable, having said why. Every rejection is a measurement: a dead
    pulse trace, no colour stream, no detectable face, or a heart rate outside
    what a person has.
    """
    clip_id = session["clip_id"]
    session_dir = Path(dataset_dir) / session["corpus"] / session["session"]
    cache_dir = Path(cache_dir)

    if not force:
        cached = _reusable_row(session_dir, cache_dir, clip_id) \
            if framecache.open_clip(cache_dir, clip_id) is not None else None
        if cached is not None:
            return cached

    scratch = Path(scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        labels = _read_labels(session, scratch, clip_id)
        if labels is None:
            return None
        pulse, mat_blob, pulse_zip = labels

        hr = clip_hr(pulse)
        if not (HR_VALID[0] <= hr <= HR_VALID[1]):
            print(f"  skip {clip_id}: derived HR {hr:.1f} bpm outside "
                  f"{HR_VALID[0]:.0f}-{HR_VALID[1]:.0f}")
            return None

        same_archive = (
            session["rgb_outer"] == session["pulse_outer"]
            and session["rgb_inner"] == session["pulse_inner"]
        )
        rgb_zip = pulse_zip if same_archive else _extract_member(
            session["rgb_outer"], session["rgb_inner"], scratch / "rgb.zip"
        )
        with zipfile.ZipFile(rgb_zip) as archive:
            names = select_colour_frames(archive)
            if names is None:
                print(f"  skip {clip_id}: no mosaiced colour stream in the archive")
                return None
            if len(names) < DETECT_FRAMES:
                print(f"  skip {clip_id}: only {len(names)} frames")
                return None

            fps, origin, fps_source = _frame_rate(archive, len(names), pulse, clip_id)
            geom = _geometry(archive, names, clip_id)
            if geom is None:
                return None
            written = _write_frames(
                archive, names, geom, framecache.frames_path(cache_dir, clip_id)
            )

        x, y, side = geom.box
        mask_path = MASK_DIR / f"{framecache.slug(clip_id)}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, np.packbits(geom.mask))

        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pulseOx.mat").write_bytes(mat_blob)
        np.savez(framecache.ppg_path(cache_dir, clip_id),
                 times=pulse["times_epoch"] - origin, values=pulse["values"])

        row = {
            "clip_id": clip_id,
            "source": NAME,
            "subject_id": session["subject_id"],
            "video_path": str(session_dir),
            "fps": float(fps),
            "n_frames": int(written),
            "duration_s": float(written / fps),
            # box_side is the *stored* side, because that is the shape the loader
            # memmaps and the size its mask mapping scales against. The crop taken
            # from the source frame is in meta.json as `source_box`.
            "box_x": int(x), "box_y": int(y), "box_side": int(geom.store_side),
            "mask_path": str(mask_path),
            "skin_frac": float(geom.mask.mean()),
            "hr_bpm": float(hr),
            "sbp_mmhg": None,
            "dbp_mmhg": None,
            "corpus": session["corpus"],
            "ppg_zero_frac": float(pulse["ppg_zero_frac"]),
            "ppg_max_gap_s": float(pulse["ppg_max_gap_s"]),
            "fps_source": fps_source,
        }
        (session_dir / "meta.json").write_text(json.dumps({
            "row": row,
            "bayer": geom.bayer,
            "bayer_margins": geom.margins,
            "shift": geom.shift,
            "source_box": [int(x), int(y), int(side)],
            "ppg_origin_epoch": origin,
            "fps_source": fps_source,
            "pulse_hz": round(pulse["n_pulse"] / pulse["span_s"], 2),
            "source_zip": [session["rgb_outer"], session["rgb_inner"]],
        }, indent=1))

        # Sidecar last, as framecache.build does: `open_clip` needs both files, so
        # an interrupted run leaves a .raw that reads as absent and is rebuilt
        # rather than read short without notice.
        framecache.sidecar_path(cache_dir, clip_id).write_text(json.dumps({
            "clip_id": clip_id,
            "side": int(geom.store_side),
            "box": [int(x), int(y), int(geom.store_side)],
            "fps": float(fps),
            "n_frames": written,
            "video_path": str(session_dir),
        }, indent=1))
        return row
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _prepare_one(job: tuple) -> dict | None:
    session, scratch, cache_dir, dataset_dir, force = job
    try:
        return prepare(session, scratch, cache_dir, dataset_dir, force)
    except Exception as exc:  # noqa: BLE001 - one bad session must not end the pass
        print(f"  skip {session['clip_id']}: {type(exc).__name__}: {exc}")
        return None


def build(
    downloads: Path | None = None,
    scratch_root: Path = BUILD_ROOT / "mrnirp_scratch",
    cache_dir: Path = framecache.CACHE_DIR,
    dataset_dir: Path = DATASET_DIR,
    limit: int | None = None,
    force: bool = False,
    workers: int = 4,
) -> pl.DataFrame:
    """Prepare every discovered session. Resumable; skips what is already cached.

    Sessions run in parallel processes because the work is per-session serial and
    embarrassingly parallel across them: inflate, demosaic and YuNet are all CPU,
    and one session alone leaves most of the machine idle. Each worker gets its
    own scratch directory, so a crash cannot leave two of them sharing a path.
    """
    sessions = discover(downloads)
    if not sessions.height:
        print("no MR-NIRP sessions found")
        return pl.DataFrame()

    picked = sessions.head(limit) if limit else sessions
    print(f"{sessions.height} sessions with both an RGB stream and a pulse trace"
          f"{f', preparing {picked.height}' if limit else ''}")

    jobs = [
        (row, Path(scratch_root) / row["session"], Path(cache_dir),
         Path(dataset_dir), force)
        for row in picked.to_dicts()
    ]
    rows: list[dict] = []
    workers = max(1, min(workers, len(jobs), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_prepare_one, job): job[0]["clip_id"] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is None:
                continue
            rows.append(row)
            print(f"  [{done}/{len(jobs)}] {row['clip_id']}  {row['n_frames']} frames "
                  f"@ {row['fps']:.2f} fps, box {row['box_side']}px, "
                  f"HR {row['hr_bpm']:.1f} bpm, skin {100 * row['skin_frac']:.1f}%",
                  flush=True)

    shutil.rmtree(scratch_root, ignore_errors=True)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [{k: r.get(k) for k in ROW_SCHEMA} for r in rows], schema=ROW_SCHEMA
    ).sort("clip_id")
