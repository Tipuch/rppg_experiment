"""On-the-fly windowed dataset: decode, crop and augment inside the dataloader.

Nothing is materialised to disk. Storing preprocessed frames for all of MCD at
256x256 would need roughly 3 TB at native frame rate; decoding a window costs
0.3-0.6 s on these files and parallelises across workers, so frames are produced
when the batch needs them and released afterwards.

The deterministic parts -- face box and skin mask -- are read from the manifest,
precomputed once by src/model/clips.py. Only decoding and arithmetic happen here.

Each item is one window: frames, skin mask and the contact-PPG waveform.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from ..aggregation.splits import DEFAULT_FRAMES as SPLIT_FRAMES
from ..aggregation.video import read_window
from ..paths import BUILD_ROOT
from . import framecache
from .waveform import hr_from_waveform, load_ppg, sample_ppg

# One OpenCV thread per process, for the same reason ffmpeg gets one: this module
# runs inside dataloader workers, and cv2 defaults to a thread per core. Twelve
# workers each spawning sixteen resize threads on a sixteen-thread machine turns
# parallel decoding into contention. Set at import so every worker inherits it.
cv2.setNumThreads(1)

# Resolution the cached SegFace skin mask was computed at, by src/model/clips.py.
# It is the mask's coordinate frame; frames are never routed through it. See
# _geometry.
MASK_RES = 256
# Fraction of the face box the augmentation crop keeps, leaving the remaining
# 12.5% as translation jitter.
AUG_FRACTION = 224 / 256
# What the model is fed. CFMamba-Phys, RhythmMamba and RhythmFormer all
# specify 128x128.
MODEL_RES = 128

DEFAULT_CLIP_FRAMES = SPLIT_FRAMES

# BT.601 luma weights. They sum to 1, which is what lets a constant added to R, G
# and B shift Y by exactly that constant without touching chroma.
BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Every source is resampled onto this timebase so a frame count means the same
# span everywhere: 160 frames is 5.33 s whether the recording was 60, 29.92 or
# 24 fps. Sources otherwise disagree by up to 2.5x on what a window covers.
TARGET_FPS = 30.0

# How frames are normalised before the network sees them, matching the
# rPPG-Toolbox `DATA_TYPE` setting that all three source papers ran under.
#
#   "standardized"  (x - mean) / std over the whole window, ONE scalar pair for
#                   every pixel, frame and channel. This is what
#                   `BaseLoader.standardized_data` does, and what
#                   RHYTHMFORMER_BASIC.yaml and PHYSMAMBA_BASIC.yaml select.
#   "raw"           [0, 1]. What this loader did before, kept for comparison.
#
# The distinction is not cosmetic for a two-branch stem: `alpha*X_raw +
# beta*X_diff` mixes the two branches, and although each has its own BatchNorm
# the relative scale entering those norms departs by three orders of magnitude
# between the two settings.
FRAME_NORMALISATIONS = ("standardized", "raw")

# HR-balanced resampling augmentation (RhythmFormer 4.3). Windows whose own heart
# rate sits above HR_HIGH are stretched in time so the apparent rate falls; windows
# below HR_LOW are compressed so it rises. The middle band is left alone.
#
# The arithmetic, which is what fixes the direction: decoding n_frames at
# target_fps = TARGET_FPS * k covers n_frames / (TARGET_FPS * k) seconds, so the
# window keeps hr_true / 60 * n_frames / (TARGET_FPS * k) beats. Post-processing
# assumes TARGET_FPS, so the apparent rate is hr_true / k. k > 1 lowers it.
HR_HIGH_BPM = 90.0
HR_LOW_BPM = 75.0
K_SLOWER = 1.4                       # upper k for an already-fast window
K_FASTER = 0.7                       # lower k for an already-slow window


class WindowDataset(Dataset):
    """Windows sampled from clips listed in a manifest.

    train=True samples a random window start, a random AUG_FRACTION sub-window of
    the face box, a random horizontal flip and an HR-balanced temporal resampling
    factor. train=False takes a centred window, a centre crop, no flip and k = 1,
    so evaluation is deterministic and comparable across runs.
    """

    def __init__(
        self,
        manifest: pl.DataFrame,
        n_frames: int = DEFAULT_CLIP_FRAMES,
        train: bool = True,
        seed: int = 0,
        resolution: int = MODEL_RES,
        apply_skin_mask: bool = False,
        frame_norm: str = "standardized",
        hr_balance: bool = True,
        flip: bool = True,
        return_waveform: bool = True,
        cache_dir: Path | None = framecache.CACHE_DIR,
    ) -> None:
        self.rows = manifest.to_dicts()
        self.n_frames = n_frames
        self.train = train
        self.seed = seed
        self.resolution = resolution
        # Off by default: none of the three papers masks, and PGA's Gaussian prior
        # is what focuses the model on skin. The mask is still returned either way,
        # because PGA obtains that prior's centroid and spread from it.
        self.apply_skin_mask = apply_skin_mask
        if frame_norm not in FRAME_NORMALISATIONS:
            raise ValueError(
                f"unknown frame_norm {frame_norm!r}, expected one of {FRAME_NORMALISATIONS}"
            )
        self.frame_norm = frame_norm
        self.hr_balance = hr_balance
        self.flip = flip
        self.return_waveform = return_waveform
        # Where src/model/framecache.py put the decoded face boxes. A clip with no
        # entry there falls back to ffmpeg, so MCD and any corpus that was never
        # cached still load unchanged; None disables the cache directly, which is
        # what the equivalence tests use to force the decoder path.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # One RNG per worker process, created lazily in _rng so each worker gets
        # its own stream. See _rng for why this is not seeded per item.
        self._stream: random.Random | None = None
        self._masks: dict[str, np.ndarray] = {}
        self._ppg: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._frames: dict[str, np.memmap | None] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, row: dict) -> np.ndarray:
        """Unpack and cache the clip's skin mask; one bit per pixel on disk."""
        key = row["mask_path"]
        if key not in self._masks:
            packed = np.load(key)
            self._masks[key] = (
                np.unpackbits(packed)[: MASK_RES * MASK_RES]
                .reshape(MASK_RES, MASK_RES)
                .astype(bool)
            )
        return self._masks[key]

    def _rng(self, index: int) -> random.Random:
        """Randomness for one item's augmentation.

        **Training draws from one continuous per-worker stream, not a per-index
        seed.** The seed used to be `self.seed * 1_000_003 + index`, and nothing
        altered `self.seed` between epochs, so a segment saw one fixed crop, one
        fixed flip and one fixed resampling factor for the whole run -- a 50/50
        partition assigned once rather than an augmentation. Measured: segment 0
        took crop (22, 19) and flip False in every epoch of a run.

        The usual fix is to reseed from `torch.initial_seed()`, which DataLoader
        varies per epoch. That does not work here: `persistent_workers=True` keeps
        workers alive across epochs, so their base seed never changes either. A
        continuous stream does vary, at the cost of exact per-index reproducibility
        -- which is the property that caused the bug. The run as a whole is still
        seeded through `torch.manual_seed` in `train`.

        Evaluation keeps the per-index seed, because a scored window has to be the
        same window every time it is scored.
        """
        if not self.train:
            return random.Random(index)
        if self._stream is None:
            # Mixed with the pid so peer workers, which are forked from one
            # parent and would otherwise share a seed, do not draw the same sequence.
            self._stream = random.Random(self.seed * 1_000_003 + os.getpid())
        return self._stream

    def _cached(self, row: dict) -> np.memmap | None:
        """The clip's cached face box, or None if it was never built.

        Kept open per worker. A memmap is a file mapping, not a buffer, so twelve
        workers share one set of pages and the dict costs a handle apiece.
        """
        key = row["clip_id"]
        if key not in self._frames:
            self._frames[key] = (
                None if self.cache_dir is None
                else framecache.open_clip(self.cache_dir, key)
            )
        return self._frames[key]

    def _span(self, k: float) -> float:
        """Seconds of real recording one window covers at resampling factor k."""
        return self.n_frames / (TARGET_FPS * k)

    def _window_start(self, row: dict, k: float, rng: random.Random) -> float:
        """Where in the recording this window begins, in seconds.

        Two cases. A row bringing `window_start_s` came from `expand_to_segments`,
        so its position is fixed by the enumeration and is used exactly, in training
        as well as evaluation. A row without it is one window per clip: centred for
        evaluation so the number is reproducible, random for training so successive
        epochs see different parts of the recording.

        Enumerated starts used to be jittered by up to +/-0.5 s in training, on the
        argument that a segment boundary is an artifact of the enumeration rather
        than of the recording. That was removed: it was the only thing breaking the
        strict non-overlap both source papers score under, and at +/-0.5 s two
        adjacent 5.33 s windows could share a second of footage, so each window
        escaped into its neighbours.
        """
        latest = max(0.0, row["duration_s"] - self._span(k))
        fixed = row.get("window_start_s")
        if fixed is not None:
            return min(max(float(fixed), 0.0), latest)
        if not self.train:
            return latest / 2.0
        return rng.uniform(0.0, latest)

    def _resample_factor(
        self, row: dict, start: float, rng: random.Random,
        wave: np.ndarray | None = None,
    ) -> float:
        """HR-balanced temporal resampling factor for this window.

        Chosen from the contact PPG alone -- no video decode -- so the cost is a
        Welch-style FFT over a few hundred samples. Returns 1.0 whenever there is
        no usable waveform, the rate already sits in the middle band, or the clip
        is too short to hold the stretched window.

        `wave` is the k=1 sampling of this window, which the caller already needs
        as the target whenever k comes back 1.0. Passing it in is what stops
        `__getitem__` interpolating the same trace twice per item.
        """
        if not (self.train and self.hr_balance):
            return 1.0
        if wave is None:
            wave = self._waveform(row, start, 1.0)
        hr = hr_from_waveform(wave, fps=TARGET_FPS)
        if not math.isfinite(hr):
            return 1.0
        if hr > HR_HIGH_BPM:
            k = rng.uniform(1.0, K_SLOWER)
        elif hr < HR_LOW_BPM:
            k = rng.uniform(K_FASTER, 1.0)
        else:
            return 1.0
        # k < 1 stretches the window in real time. If that no longer fits inside
        # the recording, fall back rather than invisibly reading a short window.
        if start + self._span(k) > row["duration_s"]:
            return 1.0
        return k

    def _crop_window(self, side: int, rng: random.Random) -> tuple[int, int, int]:
        """(top, left, size) of the augmentation sub-window, in source pixels.

        Chosen in the source frame rather than in a fixed intermediate resolution.
        The jitter is 12.5% of the box either way, so the augmentation is unchanged;
        what changes is that no resampling happens before the crop.
        """
        size = max(1, round(side * AUG_FRACTION))
        margin = side - size
        if margin <= 0:
            return 0, 0, side
        if self.train:
            return rng.randint(0, margin), rng.randint(0, margin), size
        return margin // 2, margin // 2, size

    @staticmethod
    def _face_box(raw: np.ndarray, row: dict) -> tuple[np.ndarray, int]:
        """The square face crop at source resolution, and its side.

        A clip with no detection falls back to the largest centred square, which is
        what src/aggregation/face.apply_box does. Squaring matters: the frame is
        640x480, and resizing that straight to a square would stretch the face
        horizontally by 33%.
        """
        side = int(row["box_side"])
        if side:
            x, y = int(row["box_x"]), int(row["box_y"])
            return raw[:, y : y + side, x : x + side], side
        height, width = raw.shape[1:3]
        square = min(height, width)
        top, left = (height - square) // 2, (width - square) // 2
        return raw[:, top : top + square, left : left + square], square

    def _mask_window(
        self, row: dict, top: int, left: int, size: int, source_side: int
    ) -> np.ndarray:
        """The crop window mapped from source pixels into the mask's own frame.

        The mask was segmented once per clip at MASK_RES over the whole face box, so
        a sub-window of the box is the same sub-window of the mask scaled by
        MASK_RES/source_side. Getting this mapping wrong would hand PGA a prior
        describing a different part of the face than the frames show, and nothing
        would raise -- both tensors would still be the right shape.
        """
        mask = self._mask(row).astype(np.float32)
        scale = MASK_RES / max(source_side, 1)
        m_size = max(1, min(MASK_RES, round(size * scale)))
        m_top = min(max(0, round(top * scale)), MASK_RES - m_size)
        m_left = min(max(0, round(left * scale)), MASK_RES - m_size)
        return mask[m_top : m_top + m_size, m_left : m_left + m_size]

    @staticmethod
    def _resize(image: np.ndarray, side: int) -> np.ndarray:
        """Resample to `side`, choosing the filter by direction.

        INTER_AREA integrates every source pixel that lands in an output pixel, so
        it preserves the local mean exactly, and the signal here is a
        spatially smooth sub-LSB brightness change. But **cv2.INTER_AREA degenerates
        to nearest-neighbour when requested to enlarge**: measured, a 4x4 ramp taken to
        8x8 comes back exactly equal to INTER_NEAREST (16 distinct values against
        INTER_LINEAR's 44). So enlarging uses INTER_LINEAR, which interpolates and
        does not overshoot.

        On this corpus 12.7% of MCD clips have a box small enough to need enlarging
        (the smallest 87.5% crop is 94 px against a 128 target); no UBFC clip does.
        """
        if image.shape[0] == side and image.shape[1] == side:
            return image
        flag = cv2.INTER_AREA if image.shape[0] >= side else cv2.INTER_LINEAR
        return cv2.resize(image, (side, side), interpolation=flag)

    def _maybe_flip(
        self, frames: np.ndarray, mask: np.ndarray, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random horizontal flip, applied to frames and skin mask together.

        Both together. PGA obtains its Gaussian prior's centroid from
        the mask, so switching one without the other aims the physiological prior at
        the mirror image of the face the model is given -- and
        nothing raises, because both tensors keep their shape.

        A face is near-symmetric but not symmetric, and rPPG has no left/right
        preference, so this is close to free: it doubles the effective subject
        count for a corpus of 42 people.
        """
        if not (self.train and self.flip and rng.random() < 0.5):
            return frames, mask
        return frames[:, :, ::-1], mask[:, ::-1]

    def _normalise(self, frames: torch.Tensor) -> torch.Tensor:
        """Scale the window as the toolbox does, so the network sees its input.

        `standardized` uses one scalar mean and std for the whole window --
        intentionally, not per channel and not per frame. A per-channel z-score
        would flatten the chrominance differences between R, G and B, and the
        pulse resides in exactly those: haemoglobin absorbs green far more than red.
        A per-frame z-score would be worse still, removing the frame-to-frame
        brightness change that *is* the signal.
        """
        if self.frame_norm == "raw":
            return frames / 255.0
        spread = frames.std()
        if not torch.isfinite(spread) or spread < 1e-8:
            return torch.zeros_like(frames)
        return (frames - frames.mean()) / spread

    def _waveform(self, row: dict, start: float, k: float) -> np.ndarray:
        """Contact PPG over this window, on the same timebase as the frames.

        Sampled at the frame times the decoder was requested for -- start + i/(fps*k)
        -- so target sample i corresponds to frame i whatever k is. Getting this
        wrong under augmentation would desynchronise every target.

        The target is the PPG itself whatever the input representation. Differencing
        it to match a differenced input is tempting and wrong: differencing scales
        the spectrum by |2*pi*f|, which lifts PPG's dicrotic-notch harmonic above
        the fundamental, and the readout then locks onto the harmonic. Measured on
        this manifest, a differenced target reads 161 bpm where the label is 77 and
        194 where it is 94. Input representation and target representation are
        independent; only the target has to stay readable.
        """
        # Where the *labels* live, which is not always where the pixels live: a
        # remuxed manifest points `video_path` at the rewritten container while
        # MCD's PPG is still found relative to the original. See
        # src/aggregation/remux.rewrite_manifest.
        key = row.get("ppg_video_path") or row["video_path"]
        if key not in self._ppg:
            video = Path(key)
            # The cache keeps the same trace already parsed. load_ppg reads text
            # with np.loadtxt, and the evaluation loaders are not persistent, so
            # their workers re-parsed every dev clip on every epoch.
            loaded = (
                None if self.cache_dir is None
                else framecache.open_ppg(self.cache_dir, row["clip_id"])
            )
            # MCD-rPPG's waveform resides beside the video rather than inside a
            # per-clip directory, and is timed from the frame rate rather than
            # bringing timestamps, so both are passed through.
            if loaded is None:
                loaded = load_ppg(video.parent, video_path=video, fps=row["fps"])
            self._ppg[key] = loaded
        loaded = self._ppg[key]
        if loaded is None:
            return np.zeros(self.n_frames, dtype=np.float32)

        times, values = loaded
        frame_times = start + np.arange(self.n_frames, dtype=np.float64) / (TARGET_FPS * k)
        return sample_ppg(times, values, frame_times)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        rng = self._rng(index)

        # Start is drawn against the unstretched span, then k is chosen from the
        # PPG over that nominal window and rejected if it would overrun the clip.
        start = self._window_start(row, 1.0, rng)
        # The k=1 waveform picks k, and *is* the target whenever k comes back 1.0 --
        # which is every window in the middle HR band and every evaluation window.
        # Sampling it once removes an np.interp over the whole trace per item.
        needs_wave = self.return_waveform or (self.train and self.hr_balance)
        nominal = self._waveform(row, start, 1.0) if needs_wave else None
        k = self._resample_factor(row, start, rng, wave=nominal)

        # Cached face box first. src/model/framecache.py decoded it once at native
        # resolution and native rate, so a window is a 23 MB fancy-index into a
        # memmap instead of 147 MB of uncompressed AVI off disk and an ffmpeg
        # process per item. `window_indices` is the timebase resample; see that
        # module for why it, and not ffmpeg's `fps=` filter, is the definition.
        side = int(row["box_side"])
        cached = self._cached(row)
        if cached is not None:
            index = framecache.window_indices(
                start, self.n_frames, float(row["fps"]), k, len(cached)
            )
            raw = np.asarray(cached[index])
        else:
            # No cache for this clip. The face box goes into the filter graph, not
            # into a Python slice: it is about 9% of a 640x480 frame, so cropping
            # here rather than after the pipe cuts the per-segment decode roughly in
            # half (MCD 0.237 -> 0.133 s, UBFC 0.150 -> 0.069 s) and the piped bytes
            # by 11x. Exact pixel selection, so nothing is resampled and nothing is
            # lost.
            box = (int(row["box_x"]), int(row["box_y"]), side) if side else None
            # trust_crop: the box is a manifest column, derived by
            # src/model/clips.py from the real frames, so it needs no clamp -- and
            # the probe that would supply one costs 5x the decode on MCD.
            raw = read_window(
                Path(row["video_path"]), start, self.n_frames,
                target_fps=TARGET_FPS * k, crop=box, trust_crop=box is not None,
            )
            if raw is None:
                raise RuntimeError(f"decode failed: {row['clip_id']} at {start:.2f}s")

        # --- geometry: crop in source pixels, resample exactly once -------------
        #
        # This used to go box -> 256 -> crop 224 -> 128, three resampling steps for
        # one crop. Worse, 94% of clips have a face box smaller than 256, so the
        # first step was an *enlargement* -- and cv2.INTER_AREA enlarging is
        # exactly equal to INTER_NEAREST, so those clips were pixel-duplicated up
        # and then averaged back down. Measured on eight UBFC clips, the recovered
        # rate was unchanged and peak prominence moved by a few percent either way,
        # so it was costing precision rather than signal. It is still wrong, and
        # now there is one resample: crop the box, resize once.
        # Already cropped by the decoder when a box exists; _face_box only has to
        # square a frame that had no detection.
        box, source_side = (raw, raw.shape[1]) if side else self._face_box(raw, row)
        top, left, size = self._crop_window(source_side, rng)

        # Float before any resampling. The decode is 8-bit because the source is,
        # and that quantisation is the camera's. Every step after it is ours, and a
        # pulse is 0.1-0.5 LSB per pixel: resizing in uint8 rounds interpolated
        # values to integers and injects +/-0.5 LSB of noise, the same size as the
        # thing being measured. Rounding here is not a rounding error, it is the
        # signal.
        cropped = box[:, top : top + size, left : left + size].astype(np.float32)
        side = self.resolution
        frames = np.stack([self._resize(f, side) for f in cropped])[:, :, :, ::-1]

        # The cached mask resides in its own 256-pixel frame, so the same sub-window
        # has to be mapped into it rather than taken at face value. Skin coverage
        # stays a float, not a bool: PGA reads a centroid and a spatial spread off
        # it, and an area-resampled float mask gives the fraction of each output
        # pixel that was skin instead of a hard nearest-neighbour pick.
        mask = self._resize(self._mask_window(row, top, left, size, source_side), side)

        frames, mask = self._maybe_flip(frames, mask, rng)

        if self.apply_skin_mask:
            frames = frames * mask[None, :, :, None]

        wave = None
        if self.return_waveform:
            wave = nominal if k == 1.0 else self._waveform(row, start, k)

        # Short windows are padded by repeating the last frame rather than dropped:
        # a clip near the end of a recording should still contribute.
        if len(frames) < self.n_frames:
            pad = self.n_frames - len(frames)
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])

        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        item = {
            "frames": self._normalise(tensor),
            "skin": torch.from_numpy(np.ascontiguousarray(mask)).float(),
            # What the timebase was stretched by, so a caller can undo the
            # augmentation when reporting a heart rate in real units.
            "fps_scale": torch.tensor(k, dtype=torch.float32),
            "clip_id": row["clip_id"],
            "subject_id": row["subject_id"],
        }
        if wave is not None:
            item["wave"] = torch.from_numpy(wave)
        return item


def load_manifest(path: Path = BUILD_ROOT / "clips.parquet") -> pl.DataFrame:
    return pl.read_parquet(path)


def split_manifest(
    manifest: pl.DataFrame, seed: int = 20260822
) -> dict[str, pl.DataFrame]:
    """Subject-grouped 85/10/5, reusing the aggregation splitter.

    Grouping by subject is required: windows from one recording are
    near-duplicates of each other, so a row-level split would score a model on
    people it trained on.

    **Pass segments, not clips.** `assign` balances the split by row count, so
    giving it clips gives 85/10/5 in *clips* -- and clip length varies by 4x
    across this corpus (UBFC's 43-118 s against MCD's 111-225 s), so the ratios
    the model trains and is scored on would drift from the target. Call
    `expand_to_segments` first and the row count is the segment count.
    """
    from ..aggregation.splits import assign

    tagged = assign(manifest, seed=seed)
    return {name: tagged.filter(pl.col("split") == name) for name in ("train", "dev", "test")}


def prepare_splits(
    manifest: pl.DataFrame,
    n_frames: int = DEFAULT_CLIP_FRAMES,
    fps: float = TARGET_FPS,
    seed: int = 20260822,
    stride_frames: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Segments, then a subject-grouped 85/10/5 over them. The training entrypoint.

    Order matters, and is why this function exists: enumerate every
    fixed-length window first, then split. Splitting clips and expanding afterwards
    gives the right number of *recordings* per side and the wrong number of
    *examples*.
    """
    return split_manifest(
        expand_to_segments(manifest, n_frames=n_frames, fps=fps,
                           stride_frames=stride_frames),
        seed=seed,
    )


def expand_to_segments(
    manifest: pl.DataFrame,
    n_frames: int = DEFAULT_CLIP_FRAMES,
    fps: float = TARGET_FPS,
    stride_frames: int | None = None,
) -> pl.DataFrame:
    """One row per fixed-length segment, rather than one row per recording.

    Both source papers divide each video into fixed-length segments and score
    every one of them. Without this, an evaluation pass sees a single centred
    window per clip -- twelve windows for the paper's twelve test subjects -- and
    the reported MAE is then an average over twelve numbers, which is not a
    measurement.

    A clip shorter than one segment still contributes one row; `WindowDataset`
    pads the tail by repeating the last frame rather than dropping it.

    Done in polars rather than by building dicts, for two reasons beyond speed.
    Rebuilding a frame from `to_dicts()` **re-infers the schema from the first 100
    rows**, so a column that is null across those rows becomes dtype Null and the
    first real value later in the frame fails to append -- which is what a
    pooled manifest looks like, since MR-NIRP's columns are null on every MCD row
    and MCD sorts first. And the count comes from `segment_counts`, the same
    expression the splitter weights by, so the two cannot drift apart.
    """
    from ..aggregation.splits import segment_counts

    stride = (stride_frames if stride_frames is not None else n_frames) / fps
    return (
        manifest.with_columns(
            segment_counts(n_frames, fps, stride_frames).alias("_segments")
        )
        .with_columns(
            window_start_s=pl.int_ranges(0, pl.col("_segments")) * stride
        )
        .explode("window_start_s")
        .with_columns(pl.col("window_start_s").cast(pl.Float64))
        .drop("_segments")
    )


