# rPPG: heart rate from facial video

A PyTorch implementation of CFMamba-Phys (Wang et al., *Biomedical Signal
Processing and Control* 126 (2026) 110996), a frequency-aware state space model
that predicts a blood-volume-pulse waveform from face video. Heart rate is derived
from the predicted waveform by band-pass and beat timing, not regressed directly.

0.9327M parameters, 79.15M MACs per frame. Published figures are 0.91M and 80.82M.

The scan is Mamba-3 (arXiv:2603.15569) rather than the Mamba-1 selective scan the
source papers call; see `ARCHITECTURE.md` section 4.1.

## Requirements

CUDA is required: Mamba-3's scan is a Triton kernel with no CPU path. All other
modules are plain PyTorch and are tested on CPU.

Three directories are not distributed with this repository:

| directory | contents | source |
|---|---|---|
| `datasets/` | video corpora and contact-PPG labels | see DATASETS.md |
| `research/` | the five source papers | published, obtain separately |
| `tools/rPPG-Toolbox` | POS and CHROM reference implementations | github.com/ubicomplab/rPPG-Toolbox |

`datasets/` is excluded because it holds human-subject facial video and health
biomarkers under data use agreements. `src/model/baselines.py` loads POS and CHROM
from the vendored toolbox rather than reimplementing them, so `tools/rPPG-Toolbox`
must be present for the `baseline` command. `src/model/postprocess.py` also reads
the toolbox's SNR and MACC definitions, so `evaluate` needs it too.

Both roots are relative to the working directory by default and neither is
hardcoded anywhere else in the source. Repoint either from the environment when
the corpora or the frame cache do not belong on the same volume as the code:

| variable | default | holds |
|---|---|---|
| `RPPG_DATA_ROOT` | `datasets` | the corpora, read only |
| `RPPG_BUILD_ROOT` | `build` | manifests, frame caches, masks, runs, figures |

```
RPPG_DATA_ROOT=/mnt/corpora RPPG_BUILD_ROOT=/scratch/build uv run python -m src.cli train
```

Both resolve once, at import of `src/paths.py`, and `~` is expanded. Every default
path shown by `--help` follows them, so a command's printed default is the path it
will actually use.

## Pipeline

```
video -> ffmpeg decode at 30 fps, face box cropped inside the filter graph
      -> random 87.5% sub-window, random horizontal flip
      -> one resample to 128x128     area if shrinking, linear if enlarging
      -> Fusion Stem                 raw frames fused with four temporal differences
      -> PGA                         Gaussian skin prior, channel-wise gating
      -> 4 x [Mamba + CAM, DF-FFN]   space is collapsed; a pure time series
      -> 1D conv head                one BVP sample per input frame
      -> two bands                   0.75-2.5 Hz to report a rate in
                                     0.75-4.0 Hz to find beats in, by MSPTD
      -> middle of three             spectral peak, median IBI, mean IBI
```

## Commands

```bash
uv run python -m src.cli --help

uv run python -m src.cli clips     # build the clip manifest: face boxes, skin masks
uv run python -m src.cli mrnirp    # ingest MR-NIRP from its zips; it ships stills
uv run python -m src.cli remux     # rewrite MCD's container so seeking is O(1)
uv run python -m src.cli combine   # pool the corpora into one manifest + split
uv run python -m src.cli cache     # decode each face box once to a uint8 cache
uv run python -m src.cli samples   # render contact sheets for visual review
uv run python -m src.cli info      # coverage, label spread, split sizes
uv run python -m src.cli check     # shapes, parameter budget, throughput
uv run python -m src.cli sanity    # fit a synthetic pulse, as a control
uv run python -m src.cli baseline  # POS and CHROM on the test windows
uv run python -m src.cli train     # fit the corpora, 50 epochs, then score
uv run python -m src.cli predict   # run one video through a model, plot the pulse
uv run python -m src.cli readout   # score every heart-rate readout on a labelled split
uv run python -m src.cli evaluate  # re-score a checkpoint on a split, without retraining
uv run python tools/plot_loss.py build/runs/<name>
```

`evaluate` exists because the tables under Results go stale without the model
changing. Every number in them passes through `postprocess.compare`, so altering the
reported readout or the beat detector invalidates them, and `train` -- which writes
them at the end of a fit -- is not something to re-run for a post-processing change.

`clips` runs YuNet once per clip for the face box and SegFace once for a median
skin mask. Both are too slow to run per batch, so their output is written to the
manifest and read from there at training time.

`train` takes no arguments for the standard run. Its options are the everyday
ones -- `--epochs`, `--batch`, `--frames`, `--workers`, `--seed`, the four loss and
schedule weights, `--sources`, `--stride`, `--resume`, `--profile`, `--manifest`
and `--run-dir`. The paper ablations (stem variant, PGA, CAM, FFN kind, PTS mode,
scan direction, frame normalisation, augmentation) are fields on `TrainConfig` in
`src/model/train.py` rather than flags.

`cache` decodes each clip's face box once at native resolution and native frame
rate into a `uint8` array. `remux` rewrites MCD-rPPG's AVI container with
`-c copy`, which copies every video packet unchanged and changes no pixel; those
files report `duration_ts=0` and no frame count, so `-ss` cannot seek and decodes
forward from frame 0. The output stays AVI because MP4 and MKV drop three frames
mid-stream on the 29.9167 fps clips, which shifts every subsequent frame and
desynchronises the contact-PPG target from the video.

`combine` pools UBFC, MR-NIRP and MCD into `build/clips_all.parquet` with one
90/3/7 split, grouped by subject and stratified by source so each corpus reaches
dev and test. `train` reads that column. A manifest with no `split` column gets a
subject-grouped 85/10/5 derived at load time, and the run prints which of the two
it used.

MCD is 180 h against the other two's 2.9, so it is ~98% of the segments under any
split. Results are reported per source, and `--stride` subsamples it.

`mrnirp` is a one-off ingest for the only corpus that ships stills rather than
video: it reads the nested zips, demosaics the Bayer frames, finds the face box
and writes the same cache artifacts `clips` and `cache` produce, so MR-NIRP then
trains through the ordinary path. It also writes its own split, persisted.

## Loss

CFMamba Eq. 19-21, verified against the paper and against the reference
implementation in `tools/rPPG-Toolbox`:

```
L = alpha * L_time + beta * L_freq        alpha = 0.8, beta = 1.0
L_time = 1 - Pearson(S_pred, S_gt)
L_freq = CE(PSD(S_pred), argmax(PSD(S_gt)))
```

`L_freq` is a cross-entropy over 105 candidate rates at 1 bpm spacing, covering
45-149 bpm, evaluated as a direct DFT at those frequencies rather than an FFT
followed by binning. The label is the target waveform's own dominant rate, read
from the contact PPG rather than from the manifest's heart-rate column, because
five UBFC subjects have a broken heart-rate readout and an intact waveform.

CFMamba states Eq. 19 with alpha and beta as symbols and gives no values.
RhythmFormer Section 3.4 supplied 0.2 and 1.0 for the same construction, and this
project ran that way until it measured the temporal term stopping at epoch 2 and
contributing ~1.5% of the total loss at `alpha=0.2`. **alpha is 0.8**, a departure
from RhythmFormer on that measurement; ARCHITECTURE.md section 6 records the
numbers. Loss values from runs before the change are on a different scale and are
not comparable term by term.

## Reading heart rate off the waveform

Heart rate is **the middle of three readouts**: the dominant spectral peak, the
median inter-beat interval, and the mean inter-beat interval. Band-pass, detect one
peak per cardiac cycle, interpolate each peak's position to sub-frame accuracy, read
the rate all three ways, return the median of the three.
`src.model.postprocess.reported_hr` is the function every number below comes through.

Two bands are in play. Rates are **reported** in 0.75-2.5 Hz, which is 45-150 bpm and
what both source papers specify. Beats are **detected** in 0.75-4 Hz by MSPTD, ported
from ppg-beats in `src/model/msptd.py` -- Bishop and Ercole 2018, benchmarked as one of
the two best open-source PPG beat detectors by Charlton et al., *Detecting beats in the
photoplethysmogram*, Physiol. Meas. 43(8) 085007 (2022),
[doi:10.1088/1361-6579/ac826d](https://doi.org/10.1088/1361-6579/ac826d). Detecting in
a wider band than the one rates are read in is that group's practice: harmonics are
what make a pulse a shape rather than a sinusoid, and a shape is what separates a beat
from a dicrotic notch.

Each member has one failure the other two do not share. The spectrum locks onto a
harmonic or a subharmonic and misses by tens of bpm. Beat timing counts a dicrotic
notch as a beat. The mean interval collapses to `(last - first) / (N - 1)`, so a
single spurious peak moves it by a whole 1/(N-1) of the rate -- 6 bpm on a 10 s window
at 72 bpm. Taking the middle value discards whichever member is furthest out, which
is the one that failed, without needing to know which it was.

`src.cli readout` runs one forward pass over a labelled split, caches it, and scores
every candidate readout against contact PPG. Over the 1569 test windows in
`build/readout_test_s900.npz`, strided so the sample spans all 265 test clips:

| readout | MAE | RMSE | rho |
|---|---|---|---|
| **median of three (reported)** | 3.41 | 7.28 | 0.834 |
| mean of three | 3.43 | **7.14** | **0.839** |
| spectral peak, rectangular, 8x pad | **3.25** | 8.01 | 0.800 |
| spectral peak, toolbox argmax | 3.35 | 8.18 | 0.793 |
| spectral peak, Hann, 8x pad | 3.39 | 8.04 | 0.804 |
| interval, median IBI | 3.86 | 7.73 | 0.821 |
| interval, mean IBI | 4.03 | 8.00 | 0.795 |

The vote beats each of its own members on RMSE and rho, which is what it is for: the
spectral group sits at RMSE 8.0-8.2 with rho 0.79-0.80 and the interval group at
7.7-8.0 with rho 0.80-0.82, and the middle value discards whichever one failed.

It does not beat the *mean* of the same three, which leads by 0.15 RMSE and 0.005 rho.
Under the previous beat detector the median led on all three, so this ordering is a
property of the current detection band rather than a settled result.

**These numbers are worse than the configuration they replaced.** With `find_peaks`
over 0.75-2.5 Hz the same sweep gave 3.29 / 6.23 / 0.870. Detecting beats in 0.75-4 Hz
was chosen anyway, for a reason the sweep cannot price: 2.5 Hz is 150 bpm, and a
readout that cannot report a tachycardia is wrong in a way this split does not
penalise. Its contact-PPG rate has a p99 of 118 bpm, and 7 of 1569 windows sit above
150 bpm. `src/model/postprocess.py` records the full band sweep beside
`DETECTION_LOW_HZ`.

Two consequences:

- **The results below are not comparable with the rPPG-Toolbox tables.**
  `postprocess.heart_rate` is still the toolbox's own argmax and is still tested
  for parity, but it is not what `evaluate` reports.
- Among the spectral variants the gain came from zero-padding, not from the window
  function and not from sub-bin interpolation: 8x padding moved MAE 3.35 -> 3.25,
  interpolation moved it 0.006, and a Hann window raised it. An earlier 600-window
  sample ranked Hann first; it was drawn from ~33 consecutive clips and the ranking
  did not hold on a sample spread across all of them.

### The dicrotic notch, counted as a beat

Beat timing has one failure the spectral readout does not, and it appears in the
labels as well as in the predictions. The dicrotic notch -- the aortic valve closing,
present in a normal pulse -- is a second local maximum on the diastolic decay. At
66 bpm it lands about 15 frames after the systolic peak, so a detector that separates
beats by a minimum spacing counts it as one: `find_peaks` with a 12-frame floor returns
17 peaks for 10 cycles, more than half the intervals are then half-cycles, and the
median reads 115 bpm for a 66 bpm pulse.

It is intermittent within one recording, because the notch grows and shrinks with
perfusion, so the same subject reads correctly in the next window. Measured under that
detector it affected 44 of 4482 contact-PPG windows: 39 MCD, 5 MR-NIRP, 0 UBFC. Those
5 were 12% of MR-NIRP's windows, enough to move that corpus to 6.79 bpm MAE and
rho -0.07 while the predicted waveforms matched their targets at MACC 0.9. Those counts
have not been re-measured under the current detector.

**MSPTD replaced it.** A minimum spacing cannot tell a notch from a beat, and the fix
that used to sit on top of it -- keep only peaks above half the median prominence, and
only when beat timing claims a rate 1.4x the spectral peak -- was a threshold patching
a detector with no notion of scale. Both constants are gone.

MSPTD asks a different question. For each half-width k it marks every sample larger
than the samples k before and k after it; the scale with the most marks is the one that
best matches the signal's own periodicity, and a sample is a beat only if it is marked
at every scale up to that one. A notch fails that at the scale the pulse rate wins. It
is `src/model/msptd.py`, a port of `msptdfastv2_beat_detector.m` from ppg-beats.

On a 20 s notched pulse with the notch at 0.4 of the cycle, against the true cycle
count:

| bpm | cycles | MSPTD | find_peaks + guard |
|---|---|---|---|
| 45 | 15.0 | **14** | 31 |
| 55 | 18.3 | **17** | 19 |
| 75 | 25.0 | **24** | 25 |
| 105 | 35.0 | **34** | 35 |
| 140 | 46.7 | **46** | 47 |
| 160 | 53.3 | **52** | 27 |

MSPTD is within one beat everywhere -- it drops the leading partial cycle, consistently.
The old pair breaks at 45 bpm, where 0.75 Hz *is* 45 bpm and its band-pass corner sits
on the signal, and again at 160.

**It has one limit, and no scale-based method can do better.** A notch at exactly half
the cycle is an evenly spaced second peak train at twice the pulse rate. No scale marks
the beats without also marking the notches, so the algorithm locks onto the doubled
rhythm: at 85 bpm it returns about twice the true count. Real notches fall at 0.35-0.45
of the cycle and at 0.4 the same signal reads correctly at every rate above.
`tests/test_msptd.py` pins the working range and that limit.

**On real predictions it costs accuracy rather than buying it.** Detector swapped with
the band held at 0.75-2.5 Hz, `interval` readout over the same 1569 windows:

| detector | MAE | RMSE | rho |
|---|---|---|---|
| find_peaks + guard | 3.65 | **6.42** | **0.866** |
| MSPTD | **3.62** | 6.86 | 0.849 |

Moving detection to 0.75-4 Hz costs more again, to 3.86 / 7.73 / 0.821. Both were
adopted anyway, for the headroom argued above: this split has a p99 of 118 bpm and
7 of 1569 windows above 150, so it prices a synthetic notch sweep at 45 and 160 bpm at
nothing. `tests/test_real_windows.py` pins that limitation as a test, and will fail if
a regenerated dump ever does sample a tachycardia -- at which point the comparison is
worth re-running rather than inherited.

## Results

Pooled test split, 4482 windows of 300 frames, batch 4, `alpha=0.8`, the
middle-of-three readout over 45-150 bpm with beats detected over 45-240 bpm.
Checkpoint at epoch 48 of a 50-epoch schedule. Regenerate with
`uv run python -m src.cli evaluate --split test`, which rewrites
`build/runs/cfmamba/eval_test_last.json`.

| split | MAE | RMSE | rho | MACC | SNR | n |
|---|---|---|---|---|---|---|
| test, all | 2.64 | 6.74 | +0.864 | 0.833 | +2.57 dB | 4482 |
| test, MCD | 2.67 | 6.79 | +0.860 | 0.832 | +2.54 dB | 4419 |
| test, MR-NIRP | 0.54 | 0.68 | +0.995 | 0.925 | +5.27 dB | 42 |
| test, UBFC | 0.82 | 1.06 | +0.999 | 0.832 | +3.63 dB | 21 |

MCD is 98.6% of those windows, so the aggregate is close to the MCD row by
construction. The MR-NIRP and UBFC rows are 42 and 21 windows.

**The readout change moved these two ways.** Against the previous configuration --
the median inter-beat interval alone, with beats found by `find_peaks` over
0.75-2.5 Hz -- the aggregate MAE improved from 2.75 to 2.64 while RMSE went from 5.20
to 6.74 and rho from 0.912 to 0.864. MACC is unchanged at 0.833, as it must be: it
compares waveforms and does not pass through a readout.

The two corpora with a recoverable pulse moved the other way from the aggregate:

| split | MAE | RMSE | rho |
|---|---|---|---|
| MR-NIRP, before | 1.34 | 2.27 | +0.951 |
| MR-NIRP, now | **0.54** | **0.68** | **+0.995** |
| UBFC, before | 1.44 | 1.96 | +0.997 |
| UBFC, now | **0.82** | **1.06** | **+0.999** |

MR-NIRP's RMSE fell by a factor of 3.3 and UBFC's by 1.8. The aggregate follows MCD
because MCD is 98.6% of the windows, and MCD is the corpus this project audited at a
4.4% pass rate and documents under "video without a recoverable pulse" in
DATASETS.md. Whether the aggregate or the two clean rows is the more informative
number is a judgement, not a measurement, and both are here rather than one.

Published in-dataset results on MCD-rPPG, from Egorov et al. (2025), which reports
MAE only:

| model | HR MAE |
|---|---|
| RhythmFormer | 2.82 |
| POS, training-free | 3.80 |
| PhysFormer | 4.08 |
| OMIT, training-free | 4.78 |
| iBVPNet | 4.83 |
| Egorov et al. | 4.86 |
| PBV, training-free | 15.37 |

The split and the heart-rate range differ from that work, so the comparison is
indicative rather than exact. That work reads heart rate over 0.5-3 Hz on 10-second
segments. This implementation reports over 0.75-2.5 Hz (45-150 bpm), detects beats
over 0.75-4 Hz, and reads the rate as the middle of three readouts -- none of which is
what those tables were produced with.

An earlier run at these settings settled at a constant output, with the temporal
term at 1.000 and dev MAE 30.97. Run-to-run variance at this learning rate is
large, and one run does not establish the result.

## Resuming

`train --resume` continues from `<run-dir>/last.pt`, restoring model weights, AdamW
moment estimates, and the learning-rate scheduler's step counter. The schedule is a
linear warmup into a cosine of length `steps_per_epoch * epochs`, so `--epochs`,
`--batch`, `--frames` and the step count per epoch must match the checkpoint. A
mismatch raises rather than continuing a different schedule without notice.

`<run-dir>/best.pt` is written whenever an epoch sets a new lowest dev loss.
`last.pt` is overwritten each epoch, so an earlier epoch cannot be recovered from
it. The reported result is still the last epoch; `best.pt` is scored only by
pointing `predict --model` or `readout --model` at it.

`history.json` and `last.pt` are written after every epoch.

## Tests

```bash
uv run python -m pytest tests/ -q     # 586 tests, requires an idle GPU
```

One test file per module, named for the paper equation it constrains. Several are
regressions for faults that produced no error:

- a frequency range selected by FFT bin index covered 45-202 bpm at T=160 and
  24-108 bpm at T=300 (`tests/test_band_mask.py`)
- a repointed manifest severed MCD's contact PPG from its clips, so `load_ppg`
  returned None and the target turned into zeros for a full epoch
  (`tests/test_remux.py`)
- the per-item augmentation RNG was seeded from a constant, so every segment saw
  one fixed crop and one fixed flip for a whole run (`tests/test_augment.py`)
- `samples --seconds` was read through a default argument bound at import, so the
  option was accepted and discarded (`tests/test_regressions.py`)
- `info` re-derived an 85/10/5 split rather than reading the manifest's own
  column, and so reported a partition no training run used
  (`tests/test_regressions.py`)

`check_targets_are_supervised` samples 64 training windows before the model is
built and raises if more than 20% have flat targets.

### Synthetic tests are not enough for the post-processing

`tests/test_real_windows.py` reads the cached forward pass in
`build/readout_test_s900.npz` and skips when it is absent, because `build/` is not
distributed. It exists because three post-processing decisions were made on signals
built from a formula and each one behaved differently on real predictions:

- a beat detector that read a synthetic notched pulse to within one beat at every rate
  from 45 to 160 bpm cost 0.44 bpm of RMSE on real predictions at the same band, and
  1.31 at a wider one
- a quality metric that cleanly separated a tone from white noise could not separate a
  prediction from white noise: medians 0.853 and 0.829, distributions overlapping
  across their whole range
- beat markers located in one band and drawn in another sat off the peak for 3152 of
  3785 real beats, while the synthetic test for the same fault passed before the fix

So that file pins the numbers the sweep above quotes -- MAE, RMSE and rho for the
reported readout, zero dropped windows, and the marker alignment -- and asserts the
dump is still the 1569-window split those thresholds were measured on. It also records
that the split has a p99 of 118 bpm, so a later dump that does sample a tachycardia
fails the test and prompts the band comparison to be re-run rather than inherited.

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | each module, the equation it implements, and where the papers are silent or inconsistent |
| [DATASETS.md](DATASETS.md) | what each corpus supports, and what was measured |
| [MODEL_CARD.md](MODEL_CARD.md) | the model card, for release alongside weights |
