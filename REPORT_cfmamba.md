# CFMamba-Phys: why training was slow, and what the loss curves say

Measured 2026-08-26 on an RTX 3070 Laptop (8 GB), 16 threads, 62 GB RAM.
Corpus: UBFC-rPPG, 48 clips. Model: 0.96M parameters.


---

## 1. The loader read 71 GB an epoch to produce 3.7 GB

UBFC-rPPG is uncompressed rawvideo AVI. One 640x480 frame is 921,600 bytes on
disk, so a 160-frame window costs **147 MB of read** to yield a 23 MB face box and
a 31 MB tensor.

`read_window` already crops the box inside the ffmpeg filter graph. That removes
the pipe and the numpy allocation but not the read: ffmpeg pulls every byte off
disk before the crop filter sees it. The box is ~9% of the frame, so 92% of the
read was thrown away.

| stage | per item |
|---|---|
| `probe` (ffprobe, LRU-cached) | 52 ms |
| `read_window` (decode of 147 MB) | 150 ms |
| `.astype(np.float32)` | 13 ms |
| 160 x `cv2.resize` to 128 | 48 ms |

Twelve workers x ~210 ms of CPU is 57 items/s of capacity; the run achieved 5.4.
The CPU was never the limit. 71 GB / 90 s = **790 MB/s sustained random read** on
a LUKS-encrypted btrfs volume, and that was.

### The fix: `src/model/framecache.py`

Decode each clip's face box once, at native resolution and native frame rate,
into one `uint8` array. **13.5 GB for all 48 clips** -- it stays in page cache on
this machine, so after the first epoch the read is free. A window becomes a 23 MB
fancy-index into a memmap and the ffmpeg process per item disappears.

Native resolution because `WindowDataset` crops 87.5% and then resamples **once**,
INTER_AREA when shrinking; a pre-resized cache would put a resample before that
crop, and the pulse is a 0.1-0.5 LSB change. Native frame rate because the
HR-balance augmentation decodes at `TARGET_FPS * k` with k in [0.7, 1.4], which a
30 fps cache would destroy.

Build it with `uv run python -m src.cli cache`.

### Equivalence, measured rather than presumed

It is not exactly equal to ffmpeg, and the difference was quantified.

ffmpeg's `fps=` filter is **pure nearest-frame selection** -- checked frame by
frame against a full decode, every frame it delivers is an exact source frame,
never an interpolation. It selects with a running running sum; index arithmetic
rounds each output frame independently. Across clips at 28.67 and 29.78 fps and
k in {0.75, 1.0, 1.35}:

- the **anchor is identical** in every case -- `round(start * fps)` is the frame
  ffmpeg delivers first;
- every other frame departs by **at most one source frame** (35 ms at 28.67 fps);
- the window covers the same span.

`window_indices` is now the definition, used by both paths.
`tests/test_framecache.py` fixes the anchor, the +/-1 bound, and that the pulse
read out of the window is unchanged.

### Result

| loader path | 12 workers | 8 | 4 |
|---|---|---|---|
| ffmpeg decode | 395 ms/batch | 496 | 616 |
| frame cache | **207 ms/batch** | 246 | 255 |
| GPU step (upper limit) | 230 ms | 230 | 230 |

The loader now delivers a batch faster than the GPU consumes one. Real epochs:
**90.6 s -> 26.1 s**, of which 3.3 s is blocked on data. Training is GPU-bound.

Other changes in the same pass: loss terms return detached device tensors instead
of floats (three device-to-host syncs per step removed), `skin` and `target` copy
with `non_blocking=True`, the contact PPG is cached as `.npy` instead of reparsed
with `np.loadtxt`, the k=1 waveform is sampled once instead of twice per item, and
`cudnn.benchmark` / `float32_matmul_precision("high")` are set.

---

## 2. Unseen loss was never logged, and it is the whole story

`evaluate()` returned heart-rate metrics only, so no loss was scored on the
unseen side. It now is, per window, on a forward pass that already happened.

15-epoch run, `build/runs/cfmamba15`:

| epoch | train loss | dev loss | dev MAE |
|---|---|---|---|
| 3 | 2.147 | **1.795** (best) | 0.61 |
| 7 | 1.578 | 2.498 | 1.33 |
| 11 | 0.661 | 7.218 | 0.61 |
| 14 | 0.337 | **14.013** | 1.94 |

Train falls 13x; unseen rises 7.8x. The frequency term does all of it (train
4.22 -> 0.20, dev 1.71 -> 13.92). With 41 training subjects the model memorises
training spectra from epoch 3 on.

**Dev MAE hides this completely.** It wanders 0.61-5.82 with no trend and ends at
1.94, which reads as healthy. On 58 windows from 4 subjects, quantised to ~0.12
bpm by the periodogram bin spacing, that curve is sampling noise.

Plot both with `uv run python tools/plot_loss.py build/runs/cfmamba15`.

---

## 3. Findings

1. **POS is not beaten.** Test MAE 3.32 bpm against POS 1.95 on the same 36
   windows; the earlier 30-epoch run reached 2.34, also short. CHROM (6.64) is
   beaten.
2. **The temporal term never learns, on either side.** Flat from epoch 2 at 0.674
   train / 0.448 dev -- Pearson ~0.33-0.55 between predicted and true waveform.
   With `alpha=0.2` it is ~1.5% of the final loss, so the optimiser has almost no
   reason to fix it. The most actionable number here. (Dev reads better than train
   because training windows carry resampling, crop and flip; dev windows do not.)
3. **The splits are too small.** Dev 58 windows / 4 subjects, test 36 / 3.
   Checkpoint selection on the dev curve would be selecting noise.
4. **The protocol is not the published one.** Both runs used a random
   subject-grouped split; all three source papers use first-30 / last-12, which
   already exists as `paper_split` in `src/model/dataset.py`.
5. **15 epochs is too many for 41 subjects.** Unseen loss bottoms at epoch 3.

---

## 4. The full corpus costs 59 hours, and the reason is seek depth

Adding MCD-rPPG takes the training split from 484 segments to 102,054 -- 12,756
steps an epoch. A first attempt at 15 epochs cleared fewer than 400 steps in nine
minutes with the GPU at 0% utilisation, so it was stopped and measured.

| stage | MCD, per window |
|---|---|
| cold `probe` | 322-650 ms |
| decode, 160 frames | **703 ms** |
| loader, 12 workers | 1115 ms/batch |

MCD **cannot** use the frame cache. It is compressed MPEG-4, so its cost is decode
rather than read, and a native-box cache would need 2.1 TB against 457 GB free.
That is the right outcome: the cache solves an amplification problem MCD does not
have.

### `trust_crop`

`read_window` called `probe` only to clamp a crop box that `src/model/clips.py`
had already derived from the real frames, and to choose a downscale that cannot
fire while the box is under 640 px. MCD's AVIs carry no usable duration, so that
probe falls back to walking the whole keyframe index in a second subprocess --
650 ms, against 130 ms to decode the window it describes.

It now skips the probe when the caller passes a manifest box, and defers to
probing when the box is really oversized. That removes 650 ms from the first
touch of each clip in each worker, which across 3,600 clips is most of epoch 0.

It does not touch the 703 ms decode, and nothing in the loader can. Seeking 100 s
into a compressed 180 s clip costs what it costs; `read_window`'s own docstring
already quotes 0.27-0.64 s "depending on seek depth". 180 hours of video decoded
fifteen times is 59 hours of work.

| option | clips | segments | 15 epochs |
|---|---|---|---|
| full MCD + UBFC | 3,648 | 120,065 | 59.3 h |
| audit-usable MCD + all UBFC | 207 | 5,846 | 0.5 h (+90 GB cache) |

**Running 3 epochs over the full corpus instead** (~12 h),
`build/runs/cfmamba_full3`. Unseen loss bottomed at epoch 3 on UBFC, so this may
sit near the useful budget anyway at 211x the training data.

### How to read that result

This project's audit puts MCD's pulse-recovery rate at 4.4% against a ~12% chance
rate. Under 85/10/5, MCD is 99.2% of the test segments, so the aggregate will
largely be a measurement of MCD. Read the per-source breakdown.
