# Dataset inventory

Eight datasets on disk, ~225 GiB actual. Categorised by what they can actually be used for,
which is not the same as what they nominally contain.

The decisive question for any rPPG source is **does the video contain a recoverable
pulse**, and it is measured, not assumed:

    uv run python -m src.cli audit --manifest build/clips.parquet

A clip passes when its mean-skin-luma spectrum has a cardiac peak that (a) is not
pinned to the bottom of the search band, (b) stands clear of the noise floor, and
(c) agrees with the labelled HR within 10 bpm. Chance agreement is ~12%, so a pass
rate below that means no signal at all.

---

## A. Usable for rPPG from video

### UBFC-rPPG -- the only source with a confirmed pulse in the pixels

| | |
|---|---|
| on disk | 86 GB: **all 50 labels, all 50 videos** (complete as of 2026-08-23) |
| subjects | 50 (8 in DATASET_1, 42 in DATASET_2) |
| labels | HR and contact PPG. **No blood pressure.** |
| video | rawvideo AVI, 640x480, **uncompressed**, 1.0-2.9 GB per clip |
| duration | 46-118 s per clip, 55 min total |

| audit | **25/48 pass (52.1%)**, 20.8% peakless, median prominence 28.4 |

Audited 2026-08-24, all 48 clips that have both a video and a label. 52.1% against
a ~12% chance rate, and against MCD-rPPG's 4.4%. This is the only corpus here that
clears the bar at scale.

| subset | clips | pass | peakless | med prominence |
|---|---|---|---|---|
| DATASET_1 | 6 | 4 (66.7%) | 16.7% | 96.5 |
| DATASET_2 | 42 | 21 (50.0%) | 21.4% | 22.8 |

Filtering on pixel-only criteria (peak exists, prominence >= 3) keeps 38 clips, and
those agree with the label 65.8% of the time -- scored after selection, never used
for it. `build/clips_clean_ubfc.parquet`.

The single clip checked before the rest arrived reproduces exactly: `10-gt` returns
**70.3 bpm against a labelled 72**, prominence 82.

**Frame rate is not 30 and not constant.** DATASET_1 runs at 28.67 fps; DATASET_2
ranges 28.77-29.98 fps, except subjects **25, 26 and 27, which run at 23.2-23.4
fps**. Any fixed-30-fps assumption mistimes every clip and mistimes those three
badly.

**DATASET_2 labels align 1:1 with frames.** All 42 subjects have exactly as many
PPG/HR samples as video frames -- no resampling needed. DATASET_1 does not: its
oximeter runs at 62 Hz against 28.67 fps (ratio 2.168-2.179), so `gtdump.xmp`
column 1 (ms) must be used to resample onto frame times.

**The HR column in DATASET_2 drops out on five subjects.** It clips at 127 and
falls to 1:

| subject | invalid HR samples |
|---|---|
| subject20 | 61.3% |
| subject11 | 28.1% |
| subject18 | 23.2% |
| subject24 | 19.2% |
| subject36 | 4.3% |

The contact PPG on those same subjects is intact -- subject20's waveform spans
-2.06 to 3.41 with a longest flat run of 6 samples. The sensor was fine; its HR
readout was not.

**This matters less than it looks.** `clips.py` already takes a median over
physiological values rather than a mean, and that median sits **2.05 bpm** from a
Welch estimate on the contact PPG across all 42 subjects. Relabelling from the PPG
moves the audit from 21/42 to 23/42 -- worth doing, not decisive. The one genuine
casualty is **subject24: labelled 96 bpm, PPG says 127.2**, and it passes the audit
against the PPG figure and fails against the label.

DATASET_1's 62 Hz oximeter has no dropouts at all and reaches 148 bpm on
`after-exercise`.

**Two DATASET_1 videos are unlabelled.** Google Drive stripped the folder names
from five DATASET_1 clips; three were matched to their subject by duration, and
`6-gt` vs `12-gt` is an exact tie. Those two sit in
`DATASET_1/_UNRESOLVED_6gt_or_12gt/` and must stay out of any split until a
spectral HR estimate separates them -- their labelled HRs differ by 12 bpm. See
`DATASET_1/_UNRESOLVED_NOTE.md`.

The 40 source zips were deleted on 2026-08-24 after verifying all 45 zipped
videos on disk byte-for-byte against the archive manifest. UBFC-rPPG is now
local-only: the Drive folder was quota-blocked for over 24 h during the original
download, so re-acquiring it is not quick.

### CLBP-300 (sample) -- signal present, far too small

| | |
|---|---|
| on disk | 1.6 GB, 5 clips |
| labels | **SBP, DBP, HR** -- the only BP source with a pulse |
| video | H.264 4K/1080p, 60 fps, ~49 Mbps |
| audit | **3/5 pass (60%)**, 20% peakless |

Cardiac peaks within 3-8 bpm on three clips. The full set is 300 subjects behind a
data use agreement; only the free 5-subject sample is here.

---

## B. Has blood pressure, but the video is unusable

### MCD-rPPG -- 249 GB, and 95.5% of it has no pulse

| | |
|---|---|
| on disk | **128 GiB actual** (`du` says 249 GiB; see below) |
| scale | 3600 recordings, 600 subjects, 180 h |
| labels | SBP, DBP, HR, respiration, SpO2 -- complete, no nulls |
| video | **MPEG-4 Part 2 Simple Profile, 640x480, 0.12-0.39 bits/pixel** |
| audit | **159/3600 pass (4.4%)**, 76.1% peakless |

4.4% is **below the ~12% chance rate**. Three cameras recording one subject
simultaneously return three different heart rates (77.3, 42.2, 77.3 for a true 100),
which no measurement of one heart can do.

Cause is not fully isolated. Contributing factors, in the order the evidence
supports them:

- **Auto-exposure/auto-gain.** The worst clip swings 80 LSB (std 32) across 10 s
  with the subject nearly still -- a cliff then a 4 s exponential recovery. A pulse
  is 0.1-0.5 LSB. Note AGC appears in CLBP-300 too, so it does not separate the two
  corpora on its own.
- **Compression.** MPEG-4 SP at 0.12-0.39 bpp with 4:2:0 chroma is hostile to a
  sub-percent, spatially smooth signal. A controlled bitrate sweep was inconclusive.

**Still useful for:** 3.6 GB of real 12-lead ECG and 700 MB of contact PPG, both
with genuine pulses -- a legitimate PPG-to-BP pretraining corpus. And appearance
-based correlates (age, adiposity) which is what the trained model actually latched
onto.

**There is no 120 GB of duplication to reclaim, despite what `du` says.** The
working tree and `.git/lfs/objects` already share extents through btrfs reflinks,
so `du -sh` double-counts every video. Measured:

    $ btrfs filesystem du -s datasets/mcd_rppg
         Total   Exclusive  Set shared
     248.65GiB     6.80GiB   120.93GiB

Actual consumption is 6.80 + 120.93 = ~128 GiB. `git lfs prune` retains all 3600
objects because every one is referenced by the checkout, and `git lfs dedup`
re-shares data that is already shared. Both are no-ops here.

---

## C. Has blood pressure, no camera

Neither can serve the primary target, but both carry real waveforms.

| dataset | on disk | contents |
|---|---|---|
| **BIDMC** | 209 MB | 53 ICU records, 8 min each. Arterial BP **waveform** on 10 records, plus PPG, ECG, RR, SpO2 and two-annotator breath labels. |
| **BUT PPG** | 282 MB | 3888 records, 50 subjects. One cuff reading per subject (`137/94`), ECG at 1 kHz with R-peak annotations, smartphone PPG. |

BIDMC's continuous arterial line is the better of the two -- real beat-to-beat
pressure rather than a single number.

---

## D. Has video, no blood pressure

| dataset | on disk | why it is here |
|---|---|---|
| **SCAMPS** | 3.6 GB | 2800 synthetic clips (labels, `.mat` and `.csv` -- the same 20 signals twice) + 10 videos. PPG, ECG and breathing waveforms, plus pose and 13 action units. Official 2000/400/400 split. Synthetic, so useful for pretraining only. |
| **MR-NIRP** | 2.8 GB | 1 of 15 indoor sessions: `Subject3_motion_940`, 1817 NIR + 1815 RGB 16-bit PGM frames, `pulseOx.mat` ground truth. RGB is raw Bayer with no stated CFA pattern, so only the NIR stream is usable. `indoor/Subject1/` is an **empty stub** -- 0 files. Remaining sessions quota-blocked. |

---

## E. Neither

**Music / working-memory** (1.6 GB). No facial video -- only FaceReader *expression
scores* derived from videos that were never distributed. Carries Empatica HR and
IBI, Biopac ECG/EDA/EMG/RESP and 44-channel fNIRS, but nothing this project can use.

---

## Summary

| category | datasets | clips with a usable pulse |
|---|---|---|
| A. video + pulse | UBFC-rPPG, CLBP-300 | **25 of 48 UBFC (52.1%)**; 3 of 5 CLBP-300 |
| B. video + BP, no pulse | MCD-rPPG | 159 of 3600, likely mostly chance |
| C. BP, no video | BIDMC, BUT PPG | n/a |
| D. video, no BP | SCAMPS, MR-NIRP | n/a |
| E. unusable | music/working-memory | n/a |

**Nothing on disk supports the original goal** -- facial video with blood pressure
and a recoverable pulse -- at a scale that would train a model. CLBP-300 has all
three for 3 clips.

What changed on 2026-08-23: UBFC-rPPG went from 1 video to all 50, and the audit
on 2026-08-24 put **25 of 48 clips (52.1%) above the pulse threshold** -- twelve
times MCD-rPPG's rate and four times chance. That does not give the project blood
pressure. It does give it a corpus where heart rate is genuinely learnable, and a
pulse-extraction trunk that a BP head could later sit on.

The audit's own resolution is now the binding constraint, not the data. `nperseg`
is capped at 256 samples, so the spectrum has 7.03 bpm bins against a 10 bpm
tolerance, and lengthening the window does not help: 300 frames scores 52.1% and
900 frames scores 50.0%, with peakless rising from 20.8% to 29.2% as the longer
window admits more drift. A finer estimator would likely move the number; a longer
one will not.

## The process lesson

Three training runs, ~10 GPU hours, all converged to predicting the training mean
before the data was checked. The audit that settles it takes 20 minutes and exists
only because those runs failed.

**Audit before acquiring, and certainly before training.** For a candidate dataset,
download a handful of clips, build a manifest, run the audit. A pass rate near or
below 12% means the corpus cannot support rPPG whatever its size or labels.
