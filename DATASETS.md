# Dataset inventory

Nine corpora on disk. Categorised by what they can actually be used for, which is
not the same as what they nominally contain.

**The target is the pulse waveform, and heart rate read off it.** Blood pressure
was the original goal and is no longer in scope. Two corpora here ship cuff
readings and the manifest still carries nullable `sbp_mmhg` / `dbp_mmhg` columns,
but nothing trains on them; they are recorded below as a property of the data,
not as a target.

**On the pass rates below.** They come from a spectral audit that has since been
removed. Its resolution was its own binding constraint (7.03 bpm bins against a
10 bpm tolerance), so read them as one weak signal, not as a gate.

**On the paths below.** They are written relative to `datasets/`, the default
corpus root. Set `RPPG_DATA_ROOT` to read them from anywhere else; no reader
hardcodes the root. See README.md § Requirements.

---

## A. Video with a usable pulse

### UBFC-rPPG -- the reference corpus

| | |
|---|---|
| on disk | 86 GB: all 50 labels, all 50 videos (complete as of 2026-08-23) |
| subjects | 50 (8 in DATASET_1, 42 in DATASET_2) |
| labels | HR and contact PPG |
| video | rawvideo AVI, 640x480, **uncompressed**, 1.0-2.9 GB per clip |
| duration | 46-118 s per clip, 55 min total |
| pass rate | 25/48 (52.1%), 20.8% peakless, median prominence 28.4 |

**Frame rate is not 30 and not constant.** DATASET_1 runs at 28.67 fps; DATASET_2
ranges 28.77-29.98 fps, except subjects **25, 26 and 27, which run at 23.2-23.4
fps**. Any fixed-30-fps assumption mistimes every clip and mistimes those three
badly.

**DATASET_2 labels align 1:1 with frames.** All 42 subjects have exactly as many
PPG/HR samples as video frames -- no resampling needed. DATASET_1 does not: its
oximeter runs at 62 Hz against 28.67 fps, so `gtdump.xmp` column 0 (ms) must be
used to resample onto frame times.

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
readout was not. `src/datasets/ubfc.py` takes a median over physiological values
rather than a mean, which sits **2.05 bpm** from a Welch estimate on the contact
PPG across all 42 subjects. The one real casualty is **subject24: named 96 bpm,
PPG states 127.2**.

DATASET_1's 62 Hz oximeter has no dropouts at all and reaches 148 bpm on
`after-exercise`.

**Two DATASET_1 videos are unlabelled.** Google Drive stripped the folder names
from five DATASET_1 clips; three were matched to their subject by duration, and
`6-gt` vs `12-gt` is an exact tie. Those two sit in
`DATASET_1/_UNRESOLVED_6gt_or_12gt/` and must stay out of any split until a
spectral HR estimate separates them -- their named HRs differ by 12 bpm.

The 40 source zips were deleted on 2026-08-24 after verifying all 45 zipped
videos against the archive manifest. UBFC-rPPG is now local-only: the Drive
folder was quota-blocked for over 24 h during the original download.

### MR-NIRP -- 44 sessions, ingested 2026-08-31

| | |
|---|---|
| on disk | 35 GB frame cache; ~245 GiB of source zips kept in `~/Downloads` |
| sessions | **44**: 29 Car (10 subjects) + 15 Indoor (8 subjects), 117.9 min |
| labels | contact PPG, epoch-timestamped, 32-60 Hz. No HR column, no BP. |
| video | 640x640 16-bit Bayer PGM stills, ~30 fps. RGB only; NIR is not ingested. |
| manifest | `build/clips_mrnirp.parquet` — **90.5 / 3.4 / 6.0** against a 90/3/7 request |

`uv run python -m src.cli mrnirp` reads the nested zips and writes the same
frame-cache artifacts the other corpora use, so MR-NIRP trains through the path
UBFC does with nothing decoded at training time.

**Drive split the download three ways, and each one loses sessions silently.**
Paired inner archives give 18 Car sessions with both streams; matching the 23
orphan `RGB-###.zip` archives on timestamps recovers 11 more; nine Indoor
sessions arrived as top-level `Subject3_still_940-015.zip` bundles. The 12
unmatched orphans have no PulseOX, so no label.

Orphan attribution is by clock: `pulseOxTime` against zip mtimes, with the
timezone offset **fitted** (-6 h, 92 votes to 37) because Car is October and
Indoor the preceding February. Matches must be unique both ways; ambiguity is
dropped. All 11 land within 1.0 s and imply 30.04-30.05 fps — neither of which
the matcher looks at.

Five things that had to be measured rather than assumed, each of which fails
silently if guessed:

- **Frame rate.** Indoor ships `CameraTimeLog*.txt`, one stamp per frame, exact.
  Car ships none, so its rate comes from frames over pulse span — valid only
  while both recordings cover the same window. `Subject6_still_940`'s oximeter
  stopped 13.25 s early, giving 32.19 fps against a true 29.98, which would slide
  the label 13 s by the end of the clip. Out-of-range rates fall back to nominal;
  `fps_source` records which applied.
- **Which stream is colour.** `Subject2_still_940` ships `cam_flea3_1/` and
  `RGB/` and no NIR directory — and the folder named `RGB` holds the *mono*
  frames (0.05% mosaic modulation against 29.67%). The stream is chosen by
  measuring modulation; a session with none is skipped.
- **The CFA pattern.** `BayerBG` on all 44, by green-parity and red-above-blue on
  the face. The test is **R > B, not R > G > B**: Car is IR-lit, where green
  outruns red on skin.
- **Bit alignment.** 12-bit left-shifted into 16 (`maxval 65535`, low nibble
  always zero). The other alignment reduced the same way yields a near-black clip.
- **Dropouts.** Zero samples are discarded. No Car session has any; every Indoor
  one does, worst 6.4%. Sessions above 20% dropout, a 2.0 s gap, or outside
  30-220 bpm are rejected — none hit those. Verified: **0 of 44 cached traces
  contain a zero**, HR spans 53.0-102.6 bpm.

**The cached box is capped at 256 px.** Indoor boxes reach 707, and one 300-frame
window would read 369 MB against the model's 14.7 MB input; capped, 59 MB. UBFC
and Car are under the cap already.

Indoor scored 0/6 on the pulse measurement taken when only 24 sessions were
ingested (Car was 8/18); the tool has since been removed, so treat it as a weak
prior, not a verdict.

Cameras: NIR is a Grasshopper3 GS3-U3-41C6NIR (mono, MONO12); RGB a Blackfly
BFLY-U3-23S6C (Sony IMX249, RAW12). Both crop 640x640 at even offsets, so the CFA
phase survives.

### CLBP-300 (sample) -- signal present, gone from disk

Five clips, 1.6 GB, 3/5 pass. **No longer on disk** -- `datasets/clbp-300-sample/`
is absent, so `build/clips.parquet` still lists five rows whose files do not
exist. The reader has been removed: it returned nothing, and the pooled manifest
`build/clips_all.parquet` never carried the corpus. The stale rows survive only in
`build/clips.parquet` and `build/clips_remux.parquet`, which carry their own
enum and still read. The full 300 subjects sit behind a data use agreement.

---

## B. Video without a recoverable pulse

### MCD-rPPG -- 249 GB, and 95.5% of it scored below chance

| | |
|---|---|
| on disk | **128 GiB actual** (`du` states 249 GiB; see below) |
| scale | 3600 recordings, 600 subjects, 180 h |
| labels | HR, respiration, SpO2, and cuff SBP/DBP -- complete, no nulls |
| video | **MPEG-4 Part 2 Simple Profile, 640x480, 0.12-0.39 bits/pixel** |
| pass rate | 159/3600 (4.4%), 76.1% peakless |

4.4% was below the ~12% chance rate. Three cameras recording one subject
simultaneously returned three different heart rates (77.3, 42.2, 77.3 for a true
100), which no measurement of one heart can do.

Cause was never fully isolated. Contributing factors, in the order the evidence
supports them:

- **Auto-exposure/auto-gain.** The worst clip swings 80 LSB (std 32) across 10 s
  with the subject nearly still -- a cliff then a 4 s exponential recovery. A
  pulse is 0.1-0.5 LSB.
- **Compression.** MPEG-4 SP at 0.12-0.39 bpp with 4:2:0 chroma is unfriendly to
  a sub-percent, spatially smooth signal. A controlled bitrate search was
  undecided.

**Still useful for:** 3.6 GB of real 12-lead ECG and 700 MB of contact PPG, both
with real pulses -- a valid waveform pretraining corpus.

**There is no 120 GB of duplication to reclaim, despite what `du` states.** The
working tree and `.git/lfs/objects` already share extents through btrfs reflinks,
so `du -sh` double-counts every video:

    $ btrfs filesystem du -s datasets/mcd_rppg
         Total   Exclusive  Set shared
     248.65GiB     6.80GiB   120.93GiB

Actual consumption is ~128 GiB. `git lfs prune` retains all 3600 objects because
every one is referenced by the checkout, and `git lfs dedup` re-shares data that
is already shared. Both are no-ops here.

---

## C. Waveforms without a camera

Neither can serve the primary target, but both carry real pulses and can
pretrain a waveform head.

| dataset | on disk | contents |
|---|---|---|
| **BIDMC** | 209 MB | 53 ICU records, 8 min each. PPG, ECG, RR, SpO2, arterial pressure waveform on 10 records, and two-annotator breath labels. |
| **BUT PPG** | 282 MB | 3888 records, 50 subjects. ECG at 1 kHz with R-peak annotations, smartphone PPG. |

---

## D. Synthetic

**SCAMPS** (3.6 GB). 2800 synthetic clips (labels as `.mat` and `.csv` -- the same
20 signals twice) plus 10 videos. PPG, ECG and breathing waveforms, pose and 13
action units. Official 2000/400/400 split. Synthetic, so pretraining only. No
reader is wired up.

---

## E. Unusable here

**Music / working-memory** (1.6 GB). No facial video -- only FaceReader
*expression scores* derived from videos that were never distributed. Carries
Empatica HR and IBI, Biopac ECG/EDA/EMG/RESP and 44-channel fNIRS, none of which
this project can use.

---

## Training on all three

`src.cli combine` pools UBFC, MR-NIRP and MCD into `build/clips_all.parquet` --
**3692 clips, 666 subjects, 183.1 h** -- with one split assigned over the pooled
table, grouped by subject and **stratified by source** so no corpus can be handed
a whole side. Achieved **90.06 / 2.93 / 7.01** in segments. `train` defaults to
this manifest and reads its split rather than deriving one.

MCD is **98.45%** of those segments against MR-NIRP's 1.09% and UBFC's 0.46%, so
an aggregate over any split is a measurement of MCD. Hence per-source reporting,
and `--stride` to subsample.

## Summary

| category | corpora | subjects with usable video |
|---|---|---|
| A. video + pulse | UBFC-rPPG, MR-NIRP | 50 UBFC, 18 MR-NIRP |
| B. video, no recoverable pulse | MCD-rPPG | 600, none usable from pixels |
| C. waveforms, no video | BIDMC, BUT PPG | n/a |
| D. synthetic | SCAMPS | n/a |
| E. unusable | music/working-memory | n/a |

UBFC remains the corpus the pipeline is built around. MR-NIRP adds **18 subjects
and 117.9 min of video** with per-frame contact PPG, in conditions UBFC has none
of: in-car, IR-illuminated, and with graded head motion.

## The process lesson

Three training runs, ~10 GPU hours, all stabilised to predicting the training
mean before the data was checked.

**Look at the data before training on it.** On MR-NIRP an afternoon of checking
turned up a wrong frame-rate assumption, an undocumented Bayer pattern, a 12-bit
alignment that would have trained on black frames, a monochrome stream labelled
`RGB`, and dropouts in every Indoor trace. None of them raises an exception.
