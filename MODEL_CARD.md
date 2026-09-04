---
license: mit
library_name: pytorch
pipeline_tag: video-classification
tags:
  - rppg
  - remote-photoplethysmography
  - heart-rate
  - physiological-measurement
  - mamba
  - state-space-model
  - unofficial-reimplementation
---

# CFMamba-Phys (unofficial reimplementation) — rPPG heart rate from facial video

> ## Unofficial. Not affiliated with the original authors.
>
> This is an **independent, third-party reimplementation** of CFMamba-Phys. It is
> **not** an official release, **not** endorsed, reviewed, or supported by the
> paper's authors, by Beijing Sport University, or by Elsevier. No weights, code,
> or data came from the authors.
>
> The name "CFMamba-Phys" is used only to identify the architecture being
> reproduced.
>
> Several architecture details are **deliberate departures** from the paper — the
> scan, the clip length, and the loss weighting. See [Deviations](#deviations).

## Model summary

A PyTorch model that predicts a blood-volume-pulse (BVP) waveform from face video.
Heart rate is **not regressed**; it is read off the predicted waveform afterwards
by band-pass filtering and beat timing.

| | |
|---|---|
| task | remote photoplethysmography (rPPG) |
| input | `(B, T, 3, 128, 128)` RGB in `[0, 1]`, plus a skin mask `(B, 128, 128)` |
| output | `(B, T)` — one BVP sample per input frame |
| clip length | 300 frames = 10.0 s at 30 fps |
| parameters | 0.9327 M |
| MACs | 79.15 M per frame (excludes the fused scan kernel) |
| framework | PyTorch |
| code | <https://github.com/Tipuch/rppg_experiment> |
| hardware | **CUDA required** — the scan is a Triton kernel with no CPU path |

```
video -> ffmpeg decode 30 fps, face crop
      -> Fusion Stem      raw frames fused with four temporal differences
      -> PGA              Gaussian skin prior, channel-wise gating, space collapsed
      -> 4 x [Mamba + CAM, DF-FFN]
      -> 1D conv head     one BVP sample per frame
      -> Butterworth 0.75-2.5 Hz + beat intervals -> heart rate
```

## Source research

The architecture comes from:

> Wang, L., Su, X., Yang, Y., Ge, H., Shen, Y. **CFMamba-Phys: A frequency-aware
> state space model with channel enhancement for remote photoplethysmography.**
> *Biomedical Signal Processing and Control* **126** (2026) 110996.
> DOI: [10.1016/j.bspc.2026.110996](https://doi.org/10.1016/j.bspc.2026.110996)

The paper leaves parts unspecified. Four further papers fill them, as recorded in
`ARCHITECTURE.md`:

| paper | supplies |
|---|---|
| RhythmMamba — arXiv:2404.06483 | what CFMamba omits; the fusion stem |
| RhythmFormer — arXiv:2402.12788 | loss-term construction and original weights |
| FreTS — NeurIPS 36 | frequency-domain MLP formulation |
| CMamba — arXiv:2406.05316 | channel-adaptive modulation pooling |
| Mamba-3 — arXiv:2603.15569 | the scan actually used here |

## Deviations

These are known, intentional, and measured. They are why this is not a
reproduction.

1. **Mamba-3, not Mamba-1.** All source papers call the Mamba-1 selective scan.
   This uses Mamba-3.
2. **300-frame clips, not 160.** The papers use 160.
3. **Loss weight `alpha = 0.8`, not 0.2.** CFMamba Eq. 19 gives `alpha` and `beta`
   as symbols with no values. RhythmFormer section 3.4 supplied 0.2/1.0, and this
   project ran that way until measurement showed the temporal term stopping at
   epoch 2 and contributing ~1.5% of the total loss. Raised to 0.8 on that
   evidence; ARCHITECTURE.md section 6 records the numbers.
4. **Heart rate is read as the median inter-beat interval**, not the dominant
   spectral peak. This departs from the rPPG-Toolbox convention, so these numbers
   are not comparable with rPPG-Toolbox tables.
5. **Parameter and MAC budget differ by +2.5% / -2.0%** from the published
   0.91 M / 80.82 M.

## Loss

```
L = alpha * L_time + beta * L_freq        alpha = 0.8, beta = 1.0
L_time = 1 - Pearson(S_pred, S_gt)
L_freq = CE(PSD(S_pred), argmax(PSD(S_gt)))
```

`L_freq` is a cross-entropy over 105 candidate rates at 1 bpm spacing, covering
45-149 bpm, evaluated as a direct DFT at those frequencies. The label is the target
waveform's own dominant rate, taken from the contact PPG rather than from a
manifest heart-rate column, because five UBFC subjects have a broken HR readout and
an intact waveform.

## Training data

**No dataset files are uploaded here.** This repository holds code and weights
only. The corpora carry human-subject facial video and health biomarkers under data
use agreements and are **not** redistributed, mirrored, or bundled in any form.
Obtain each from its own source, under its own terms, yourself.

Three corpora, pooled into one 90/3/7 split, grouped by subject and stratified by
source.

| corpus | scale | note |
|---|---|---|
| MCD-rPPG | 3600 recordings, 600 subjects, 180 h | ~98% of segments whatever the split does |
| UBFC-rPPG | 50 subjects, 55 min | variable frame rate: 23.2-29.98 fps, not 30 |
| MR-NIRP | 44 sessions, 117.9 min | RGB stream only; NIR is not ingested |

MCD-rPPG:

> Egorov, K., Botman, S., Blinov, P., Zubkova, G., Ivaschenko, A., Kolsanov, A.,
> et al. **Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health
> Biomarkers Estimation.**
> DOI: [10.1145/3746027.3758255](https://doi.org/10.1145/3746027.3758255)
> Dataset: `huggingface.co/datasets/kyegorov/mcd_rppg`

*Unverified:* UBFC-rPPG (Bobbia et al.) and MR-NIRP (Nowara et al.) are used under
their own published terms.

## Evaluation

Pooled test split, 4482 windows of 300 frames, checkpoint at 48 of 50 epochs.
Heart rate is the middle of three readouts over 45-150 bpm, with beats detected by
MSPTD over 45-240 bpm. Regenerate with `uv run python -m src.cli evaluate`.

| split | MAE (bpm) | RMSE | rho | MACC | SNR | n |
|---|---|---|---|---|---|---|
| test, all | 2.64 | 6.74 | 0.864 | 0.833 | +2.57 dB | 4482 |
| test, MCD | 2.67 | 6.79 | 0.860 | 0.832 | +2.54 dB | 4419 |
| test, MR-NIRP | 0.54 | 0.68 | 0.995 | 0.925 | +5.27 dB | 42 |
| test, UBFC | 0.82 | 1.06 | 0.999 | 0.832 | +3.63 dB | 21 |

MCD is 98.6% of the windows, so the aggregate follows its row. Against the previous
configuration -- median inter-beat interval alone, beats from `find_peaks` over
45-150 bpm -- the aggregate MAE improved from 2.75 to 2.64 while RMSE went 5.20 to
6.74 and rho 0.912 to 0.864. The two corpora with a recoverable pulse moved the other
way: MR-NIRP's RMSE fell from 2.27 to 0.68 and UBFC's from 1.96 to 1.06. MACC is
unchanged, as it must be -- it compares waveforms and does not pass through a readout.
README.md sets out that trade.

### Readout comparison

Over 1569 strided test windows spanning all 265 test clips:

| readout | MAE | RMSE | rho |
|---|---|---|---|
| middle of three (**default**) | 3.41 | 7.28 | 0.834 |
| mean of three | 3.43 | **7.14** | **0.839** |
| spectral peak, rectangular, 8x pad | **3.25** | 8.01 | 0.800 |
| spectral peak, toolbox argmax | 3.35 | 8.18 | 0.793 |
| spectral peak, Hann, 8x pad | 3.39 | 8.04 | 0.804 |
| interval, median IBI | 3.86 | 7.73 | 0.821 |
| interval, mean IBI | 4.03 | 8.00 | 0.795 |

### Published comparison

MCD-rPPG in-dataset results from Egorov et al. (2025), MAE only:

| model | HR MAE |
|---|---|
| RhythmFormer | 2.82 |
| POS (training-free) | 3.80 |
| PhysFormer | 4.08 |
| OMIT (training-free) | 4.78 |
| iBVPNet | 4.83 |
| Egorov et al. | 4.86 |
| PBV (training-free) | 15.37 |

## Intended use

Research and engineering on remote photoplethysmography: reproducing the
architecture, comparing readouts, and studying dataset quality.

## Usage

The code is on GitHub, not here: **<https://github.com/Tipuch/rppg_experiment>**

```bash
git clone https://github.com/Tipuch/rppg_experiment.git
cd rppg_experiment
uv sync

uv run python -m src.cli --help
uv run python -m src.cli predict --video path/to/face.mp4 --model path/to/last.pt
uv run python -m src.cli readout   # score every readout on a labelled split
uv run python -m src.cli train     # the pooled corpora, 50 epochs, no arguments
```

`--model` defaults to the most recent `build/runs/*/final.pt` or `last.pt`, so
point it at the downloaded checkpoint. `predict` needs no manifest entry: the face
box and skin mask are built inline.

Window length is read from the checkpoint, not passed as an option. `CUDA` is
required for any forward pass.

`datasets/` and `tools/rPPG-Toolbox` are in neither repository. Obtain the corpora
yourself. The toolbox (`github.com/ubicomplab/rPPG-Toolbox`) supplies POS and CHROM
for `src.cli baseline`, and the SNR and MACC definitions `evaluate` reports.
`RPPG_DATA_ROOT` and `RPPG_BUILD_ROOT` repoint both roots.

`README.md`, `ARCHITECTURE.md` and `DATASETS.md` are in the repository.

## Reproducing

```
seed 20260822, AdamW lr 1e-3, betas (0.9, 0.999), weight decay 0.05
grad clip 1.0, batch 4, 300 frames, 128x128, 50 epochs
5% linear warmup into cosine, min lr 1% of peak
alpha 0.8, beta 1.0
```

`train --resume` restores weights, AdamW moments, and the scheduler step counter.
`--epochs`, `--batch`, `--frames` and steps per epoch must match the checkpoint; a
mismatch raises rather than continuing a different schedule without notice.

Tests: `uv run python -m pytest tests/ -q` — 501 tests, requires an idle GPU.

## Licence and redistribution

- **Code: MIT.** See `LICENSE` in the repository. Repository:
  `github.com/Tipuch/rppg_experiment`.
- **Weights:** trained on corpora governed by data use agreements. Confirm that
  each agreement permits releasing derived weights before redistributing them.
  MIT on the code does not settle this.
- **The paper is not licensed to this project.** The architecture is reimplemented
  from the published description.

## Citation

Cite the original work, not this repository:

```bibtex
@article{wang2026cfmambaphys,
  title   = {CFMamba-Phys: A frequency-aware state space model with channel
             enhancement for remote photoplethysmography},
  author  = {Wang, Lin and Su, Xinhua and Yang, Yaqing and Ge, Huanmin and
             Shen, Yanfei},
  journal = {Biomedical Signal Processing and Control},
  volume  = {126},
  pages   = {110996},
  year    = {2026},
  doi     = {10.1016/j.bspc.2026.110996}
}
```
