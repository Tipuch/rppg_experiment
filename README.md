# rPPG: heart rate from facial video

A PyTorch implementation of CFMamba-Phys (Wang et al., *Biomedical Signal
Processing and Control* 126 (2026) 110996), a frequency-aware state space model
that predicts a blood-volume-pulse waveform from face video. Heart rate is derived
from the predicted waveform by band-pass and periodogram, not regressed directly.

0.9559M parameters, 79.19M MACs per frame. Published figures are 0.91M and 80.82M.

## Requirements

CUDA is required: `mamba_ssm`'s selective scan has no CPU kernel. All other modules
are plain PyTorch and are tested on CPU.

Two directories are not distributed with this repository:

| directory | contents | source |
|---|---|---|
| `datasets/` | video corpora and contact-PPG labels | see DATASETS.md |
| `research/` | the five source papers | published, obtain separately |
| `tools/rPPG-Toolbox` | POS and CHROM reference implementations | github.com/ubicomplab/rPPG-Toolbox |

`datasets/` is excluded because it contains human-subject facial video and health
biomarkers under data use agreements. `src/model/baselines.py` loads POS and CHROM
from the vendored toolbox rather than reimplementing them, so `tools/rPPG-Toolbox`
must be present for `baseline` and for the `--baselines` path of `train`.

## Pipeline

```
video -> ffmpeg decode at 30 fps, face box cropped inside the filter graph
      -> random 87.5% sub-window, random horizontal flip
      -> one resample to 128x128     area if shrinking, linear if enlarging
      -> Fusion Stem                 raw frames fused with four temporal differences
      -> PGA                         Gaussian skin prior, channel-wise gating
      -> 4 x [Mamba + CAM, DF-FFN]   space is collapsed; a pure time series
      -> 1D conv head                one BVP sample per input frame
      -> Butterworth 0.75-2.5 Hz + PSD -> heart rate
```

## Commands

```bash
uv run python -m src.cli --help

uv run python -m src.cli clips     # build the clip manifest: face boxes, skin masks
uv run python -m src.cli audit     # measure whether clips contain a recoverable pulse
uv run python -m src.cli remux     # rewrite MCD's container so seeking is O(1)
uv run python -m src.cli cache     # decode each face box once to a uint8 cache
uv run python -m src.cli samples   # render contact sheets for visual review
uv run python -m src.cli info      # coverage, label spread, split sizes
uv run python -m src.cli check     # shapes, parameter budget, throughput
uv run python -m src.cli sanity    # recover a synthetic pulse
uv run python -m src.cli baseline  # POS and CHROM on the test windows
uv run python -m src.cli train     # fit, then score the last epoch
uv run python -m src.cli predict   # run one video through a model, plot the pulse
uv run python tools/plot_loss.py build/runs/<name>
```

`clips` runs YuNet once per clip for the face box and SegFace once for a median
skin mask. Both are too slow to run per batch, so their output is written to the
manifest and read from there at training time.

`cache` decodes each clip's face box once at native resolution and native frame
rate into a `uint8` array. `remux` rewrites MCD-rPPG's AVI container with
`-c copy`, which copies every video packet verbatim and changes no pixel; those
files report `duration_ts=0` and no frame count, so `-ss` cannot seek and decodes
forward from frame 0. The output stays AVI because MP4 and MKV drop three frames
mid-stream on the 29.9167 fps clips, which shifts every subsequent frame and
desynchronises the contact-PPG target from the video.

`audit` reports the fraction of clips whose mean-skin-luma spectrum has a cardiac
peak agreeing with the labelled heart rate. Chance agreement is approximately 12%.

## Loss

CFMamba Eq. 19-21, verified against the paper and against the reference
implementation in `tools/rPPG-Toolbox`:

```
L = alpha * L_time + beta * L_freq        alpha = 0.2, beta = 1.0
L_time = 1 - Pearson(S_pred, S_gt)
L_freq = CE(PSD(S_pred), argmax(PSD(S_gt)))
```

`L_freq` is a cross-entropy over 105 candidate rates at 1 bpm spacing across
45-150 bpm, evaluated as a direct DFT at those frequencies rather than an FFT
followed by binning. The label is the target waveform's own dominant rate, read
from the contact PPG rather than from the manifest's heart-rate column, because
five UBFC subjects have a broken heart-rate readout and an intact waveform.

CFMamba states Eq. 19 with alpha and beta as symbols and does not give their
values. 0.2 and 1.0 are from RhythmFormer Section 3.4, whose frequency loss is the
same construction.

## Results

Full corpus, MCD-rPPG plus UBFC-rPPG, 6 epochs, 300-frame windows, batch 4,
subject-grouped 85/10/5 split, last epoch reported.

| split | MAE | RMSE | rho | MACC | SNR | n |
|---|---|---|---|---|---|---|
| test | 3.84 | 10.88 | +0.735 | 0.772 | +1.10 dB | 3163 |
| test, MCD | 3.84 | 10.89 | +0.734 | 0.772 | +1.10 dB | 3150 |
| test, UBFC | 2.97 | 6.96 | +0.938 | 0.844 | +1.49 dB | 13 |
| dev (full) | 4.49 | 11.17 | +0.741 | 0.743 | +0.23 dB | 6418 |

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

The split and the heart-rate readout band differ from that work, so the comparison
is indicative rather than exact. That work reads heart rate over 0.5-3 Hz on
10-second segments; this implementation uses 45-150 bpm.

Training loss fell 3.370, 2.560, 2.336, 2.175, 2.036, 1.949 across the six epochs.
Dev loss fell 2.919 to 2.200 and remained below training loss throughout. The
temporal term fell from 0.774 to 0.575, corresponding to a waveform Pearson
correlation of 0.226 to 0.425.

A run with identical settings collapsed to a constant-output minimum, with the
temporal term at exactly 1.000 and dev MAE 30.97. Run-to-run variance at this
learning rate is large, and a single run does not establish the result.

## Resuming

`train --resume` continues from `<out>/last.pt`, restoring model weights, AdamW
moment estimates, and the learning-rate scheduler's step counter. The schedule is a
linear warmup into a cosine of length `steps_per_epoch * epochs`, so `--epochs`,
`--batch`, `--frames` and the step count per epoch must match the checkpoint. A
mismatch raises rather than silently continuing a different schedule.

`history.json` and `last.pt` are written after every epoch.

## Tests

```bash
uv run python -m pytest tests/ -q     # 391 tests, requires an idle GPU
```

One test file per module, named for the paper equation it constrains. Several are
regressions for defects that produced no error:

- a frequency band selected by FFT bin index covered 45-202 bpm at T=160 and
  24-108 bpm at T=300 (`tests/test_band_mask.py`)
- a repointed manifest severed MCD's contact PPG from its clips, so `load_ppg`
  returned None and the target became zeros for a full epoch
  (`tests/test_remux.py`)
- the per-item augmentation RNG was seeded from a constant, so every segment saw
  one fixed crop and one fixed flip for an entire run (`tests/test_augment.py`)

`check_targets_are_supervised` samples 64 training windows before the model is
built and raises if more than 20% have flat targets.

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | each module, the equation it implements, and where the papers are silent or inconsistent |
| [DATASETS.md](DATASETS.md) | what each corpus supports, and audit results |
| [AGGREGATION_PLAN.md](AGGREGATION_PLAN.md) | the original data plan |
| [IMAGE_PIPELINE_PLAN.md](IMAGE_PIPELINE_PLAN.md) | skin isolation and brightness normalisation |
