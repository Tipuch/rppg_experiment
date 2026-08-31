# Aggregation plan: unified rPPG table

Target row schema, most to least important:
facial images (ordered, one resolution) -> pulse waveform -> HR.

> **Superseded in part, 2026-08-31.** Blood pressure was the original target and
> is no longer in scope, so the SBP/DBP/BR/HRV columns this document was built
> around are gone from the live schema. What survives is the per-corpus inventory
> below and the granularity discussion in section 3, both still accurate. The
> frame-store design in sections 4-6 was replaced by on-the-fly decoding plus a
> face-box frame cache -- see `ARCHITECTURE.md`. Readers now live one module per
> corpus in `src/datasets/`.

## 1. Verdict

Build the aggregation pipeline around the pulse waveform. It is the one target
that more than one corpus carries per frame rather than per recording, and it is
the only one whose label can be checked against the pixels.

## 2. What we actually hold

| Source | Face frames | Pulse waveform | HR |
|---|---|---|---|
| CLBP-300 (5 videos) | 2160x3840 + 1920x1080, 60 fps, ~32 s, H.264 | -- | scalar/video |
| SCAMPS (10 videos, 2800 label sets) | synthetic 320x240, 600 f @30 fps | yes | derive |
| UBFC-rPPG (50 videos) | 640x480 @~29 fps | trace @30 Hz | trace |
| MR-NIRP (24 sessions) | Bayer PGM 640x640 @30 fps | trace @32-60 Hz | derive |
| MCD-rPPG (3600 recordings) | multi-camera 640x480 | yes | yes |

"derive" = computable from a waveform in the source. "--" = absent.

Label formats confirmed by inspection:
- CLBP-300: labels encoded in filename. `Subject001_M44_138_99_66_448.mov`.
- SCAMPS `.mat` (HDF5): `RawFrames (3,320,240,600)`, `Xsub (3,240,240,600)`,
  `d_ppg`, `d_ekg`, `d_br` each `(600,1)`, plus pose and 13 AUs.
- UBFC DATASET_2 `ground_truth.txt`: 3 rows x N -- PPG wave, HR bpm, timestamps (~30 Hz).
- UBFC DATASET_1 `gtdump.xmp`: 4 cols -- time ms, HR bpm, SpO2 %, PPG (~62 Hz).
- MR-NIRP: `RGB/` PGM sequence, `PulseOX/pulseOx.mat` holding `pulseOxRecord`
  (waveform) and `pulseOxTime` (epoch seconds, irregular 32-60 Hz).
- MCD-rPPG: `ppg_sync/<name>.txt`, one sample per video frame.

## 3. The three problems to decide before writing code

### 3.1 Label granularity is not uniform — this is the main risk

CLBP-300 gives one HR scalar per 32 s video. SCAMPS/UBFC/MR-NIRP give
per-frame waveforms. These cannot share a row unit without an explicit decision.

Use a **fixed 10 s window as the row unit**. Broadcast per-video scalars across that
video's windows, and record how each label was obtained:

    hr_granularity  = window | clip | absent
    bp_granularity  = window | clip | absent

Without those flags, overlapping windows bringing an identical broadcast label leak
across a train/test split and report accuracy that is not real.

### 3.2 "HRV, same units" is under-specified

HRV is a family, not a quantity. Pin it: **SDNN in ms** primary, **RMSSD in ms** secondary.
Both need beat detection on PPG or ECG.

SDNN over 10 s is unstable. Compute HRV **per clip**, not per window, and broadcast with
`hrv_granularity = clip`. Note that SCAMPS clips are 20 s, so even clip-level HRV there is
slight — keep `is_synthetic` visible so you can drop it from HRV evaluation.

### 3.3 Polars should not hold pixels

At 128x128x300x3 uint8 a single window is ~14.7 MB. A Polars column of those is unusable.

Split it: **Polars keeps the manifest and labels; frames live in `.npy` shards.** Each row
carries `frames_path` + `frames_offset` + `n_frames`. This keeps the table a few MB and
scans fast, and the loader memory-maps shards.

## 4. Normalization rules

**Images.** Face-detect once per clip, take the median box (not per-frame — jitter adds
motion artifact that competes with the pulse signal), square-crop, resize to **128x128**.
Standard for rPPG and survives the 4K downscale; the pulse signal is spatially
low-frequency so the loss is small.

**Frame rate.** Resample all to **30 fps**. CLBP-300 60→30 by dropping alternate frames.
Nyquist becomes 15 Hz, still far above any credible HR.

**Modality.** MR-NIRP NIR is single-channel. Keep `modality = rgb | nir` and do not stack
NIR into an RGB tensor invisibly.

**Units.** HR bpm. The waveform itself is standardised per window, since absolute PPG amplitude is a property of the sensor.

**Derivations.**
- HR from PPG: bandpass 0.7–3.5 Hz, Welch PSD, dominant peak x 60.
- BR from `d_br`: bandpass 0.1–0.5 Hz, dominant peak x 60.
- HRV: beat-detect → IBI series in ms → `SDNN = std(IBI)`, `RMSSD = sqrt(mean(diff(IBI)^2))`.

**Domain gap to record, not hide.** CLBP-300 is compressed H.264 4K; UBFC is near-raw
640x480; SCAMPS is synthetic. Compression measurably attenuates rPPG. Carry
`compression = raw | h264` and `is_synthetic` so you can stratify results.

## 5. Schema

```
clip_id           str     # "clbp300/Subject001"
source            enum    # clbp300 | scamps | ubfc | mrnirp | mcd
subject_id        str
window_idx        u16
t_start_s         f32
t_end_s           f32
fps               f32     # 30.0 after resample
n_frames          u16     # 300
frames_path       str
frames_offset     u32
modality          enum    # rgb | nir
frame_h, frame_w  u16     # 128, 128
hr_bpm            f32?
sbp_mmhg          f32?   retained, nullable, never read
dbp_mmhg          f32?   retained, nullable, never read
br_bpm            f32?
hrv_sdnn_ms       f32?
hrv_rmssd_ms      f32?
hr_granularity    enum
bp_granularity    enum
hrv_granularity   enum
is_synthetic      bool
compression       enum
```

All six targets nullable. Sparsity is the expected state, not a fault.

## 6. Build order

1. **Per-source extractors** → `(frames .npy shard, signals .parquet)` per clip.
   Independent, parallelisable. Start with SCAMPS: complete, self-describing, no face
   detection needed (`Xsub` is already a 240x240 crop).
2. **Signal derivation** → HR/BR per window, HRV per clip.
3. **Manifest assembly** in Polars → one `dataset.parquet`.
4. **Validation gate** — assert unit ranges (HR 30–220,
   null counts per source, and confirm no `subject_id` appears in
   two splits.
5. **Splits** grouped by `subject_id`. Never split within a subject — windows from one
   video are near-duplicates.

## 7. Actual coverage after the build

Rows with all six targets: **zero**, until MCD-rPPG or VitalVideos lands.
Rows with images+HR: SCAMPS, UBFC, MR-NIRP, CLBP-300.
Rows with images + a per-frame pulse waveform: UBFC (50) and MR-NIRP (24).
Rows with images+BR: SCAMPS only, synthetic.

The pipeline is worth building now. The training set is not there yet.

---

# Implementation status

Replaced in scope. This document planned a **frame-store** pipeline writing
preprocessed windows to `.npy`. That approach was replaced by on-the-fly decoding
once the arithmetic came out at ~3 TB for MCD at 256x256, so the extractors and
`build_dataset.py` described below are no longer wired into the CLI.

The current pipeline is:

    src/model/clips.py     clip-level manifest: face box + skin mask, once per clip
    src/model/dataset.py   decodes windows on the fly, nothing materialised
    src/model/train.py     training, subject-grouped splits, baselines
    src/cli.py             mrnirp / clips / samples / info / check / train / predict

See **ARCHITECTURE.md** for the model and **DATASETS.md** for what each source can
be used for.

## What this plan got right, and kept

- **Subject-grouped splits.** Windows of one recording are near-duplicates, so a
  row-level split scores the model on people it trained on. Now 90/3/7 over the
  pooled manifest, stratified by source.
- **Predict-the-training-mean baselines** printed alongside every score. Three runs
  produced dev MAEs that looked reasonable until compared against a constant
  predictor -- which they equal to.
- **Target statistics sized on train only**, so dev/test label scale never leaks.
- **Explicit label granularity.** BP is one reading broadcast across a recording's
  windows; recording that as `clip` rather than `window` is what stops the gap being
  misinterpret.
- **Polars for the manifest, never for pixels.** One 128x128x300x3 window is
  ~14.7 MB.

## What it got wrong

- **Frame stores.** Materialising windows does not scale; decoding costs 0.3-0.6 s
  and parallelises across dataloader workers, so frames are produced on demand.
- **Brightness normalisation as a default.** Subtracting each frame's mean skin luma
  removes the spatially uniform component of the pulse by construction. Now opt-in
  (`normalise_brightness=False`), with the `mean_Y` branch off to match.
- **Resampling to a common fps, then not.** Native fps preserves waveform detail;
  a common timebase makes a frame count mean the same span everywhere. Currently
  **30 fps for every source**, since a fixed frame count otherwise spanned 2.7-6.7 s
  depending on the camera.
- **Looking at the data last instead of first.** Three training runs and ~10 GPU hours
  preceded the 20-minute check that explained all of them.

## Findings that outlived the plan

**UBFC reference HR contains dropouts** to 1 bpm. Comparing against the raw mean
gave 12.2 bpm MAE; filtering to 30-220 and taking the median gave 3.07.

**opencv-python 5.0.0.93 damages the heap** decoding UBFC's rawvideo AVIs --
`cv2.VideoCapture` opens the file, reports correct metadata, then fails inside the
first `read()`. All decoding goes through ffmpeg (`src/aggregation/video.py`).

**MCD AVIs carry no container duration.** `duration_ts` is 0 and `nb_frames` absent,
so both stream and format fields read zero. Recovered from the last keyframe
timestamp in ~0.26 s, within 0.2 s of truth.

**PSD bin width quantised the targets.** A 10 s window at 30 Hz resolves to 0.125 Hz
bins = 7.5 bpm, so every HR ended up on a multiple of 7.5. Fixed with zero-padding
plus parabolic interpolation: MAE 3.07 -> 2.44 bpm.

**Keyframe-only decoding is 10x faster** for the detection pass -- 0.30 s versus
3.15 s per MCD clip, which is 20 minutes rather than 3.2 hours across the corpus.
