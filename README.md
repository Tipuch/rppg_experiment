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
      -> Butterworth 0.75-2.5 Hz + beat intervals -> heart rate
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
uv run python tools/plot_loss.py build/runs/<name>
```

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

Heart rate is **the median inter-beat interval** of the predicted BVP: band-pass,
detect one peak per cardiac cycle, interpolate each peak's position to sub-frame
accuracy, difference them, take the median. This is what pulse oximeters and wrist
wearables display, and `src.model.postprocess.interval_hr` is the function every
number below comes through.

It replaced the dominant spectral peak on measurement. `src.cli readout` runs one
forward pass over a labelled split, caches it, and scores every candidate readout
against contact PPG. Over the 1569 test windows in `build/readout_test_s900.npz`,
strided so the sample spans all 265 test clips:

| readout | MAE | RMSE | rho |
|---|---|---|---|
| interval, median IBI | 3.70 | **6.60** | **0.857** |
| spectral peak, rectangular, 8x pad | **3.25** | 8.01 | 0.800 |
| spectral peak, toolbox argmax | 3.35 | 8.18 | 0.793 |
| spectral peak, Hann, 8x pad | 3.39 | 8.04 | 0.804 |

The interval readout has the highest MAE and the lowest RMSE, by 19%: fewer large
misses, slightly more small ones. rho -- whether the readout tracks the rate across
windows -- rises from 0.793 to 0.857. On the one clip inspected by hand with
contact PPG, the error fell from -28.1 to -6.5 bpm.

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
labels as well as in the predictions. The dicrotic notch -- the aortic valve
closing, present in a normal pulse -- is a second local maximum on the diastolic
decay. At 66 bpm it lands about 15 frames after the systolic peak, past the
12-frame spacing floor `beats` applies, so `find_peaks` returns 17 peaks for 10
cycles. More than half the intervals are then half-cycles and the median reads
115 bpm for a 66 bpm pulse.

It is intermittent within one recording, because the notch grows and shrinks with
perfusion, so the same subject reads correctly in the next window. On the pooled
test split it affects 44 of 4482 contact-PPG windows: 39 MCD, 5 MR-NIRP, 0 UBFC.
Those 5 were 12% of MR-NIRP's windows, enough to move that corpus to 6.79 bpm MAE
and rho -0.07 while the predicted waveforms matched their targets at MACC 0.9.

`beats` repairs it by keeping only peaks above half the median prominence, and only
when beat timing claims a rate 1.4x the spectral peak or higher. The spectrum of a
115 bpm pulse peaks at 115, so that ratio indicates double-counting. The floor is
conditional because on a noisy predicted waveform an unconditional one discards
real beats: over the 788 windows in `build/readout_test.npz` it cost 0.34 bpm MAE
and 1.4 bpm RMSE. Conditional, the same sweep improves (MAE 3.81 -> 3.79,
RMSE 6.71 -> 6.65, rho 0.858 -> 0.862), the guard fires on 0.56% of test windows,
and label inconsistency falls from 44 windows to 27.

## Results

Pooled test split, 4482 windows of 300 frames, batch 4, `alpha=0.8`, interval
readout, 45-149 bpm. Checkpoint at epoch 48 of a 50-epoch schedule
(`build/runs/cfmamba/eval_test_last.json`).

| split | MAE | RMSE | rho | MACC | SNR | n |
|---|---|---|---|---|---|---|
| test, all | 2.75 | 5.20 | +0.912 | 0.833 | +2.69 dB | 4482 |
| test, MCD | 2.77 | 5.23 | +0.909 | 0.832 | +2.66 dB | 4419 |
| test, MR-NIRP | 1.34 | 2.27 | +0.951 | 0.925 | +5.19 dB | 42 |
| test, UBFC | 1.44 | 1.96 | +0.997 | 0.832 | +3.93 dB | 21 |

MCD is 98.6% of those windows, so the aggregate is close to the MCD row by
construction. The MR-NIRP and UBFC rows are 42 and 21 windows.

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
indicative rather than exact. That work reads heart rate over 0.5-3 Hz on
10-second segments; this implementation uses 45-149 bpm and the interval readout,
which is not the readout those tables were produced with.

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
uv run python -m pytest tests/ -q     # 501 tests, requires an idle GPU
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

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | each module, the equation it implements, and where the papers are silent or inconsistent |
| [DATASETS.md](DATASETS.md) | what each corpus supports, and what was measured |
| [MODEL_CARD.md](MODEL_CARD.md) | the model card, for release alongside weights |
