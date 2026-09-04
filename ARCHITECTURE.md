# CFMamba-Phys architecture

Rebuilt from Wang et al., *Biomedical Signal Processing and Control* 126 (2026)
110996, with RhythmMamba (arXiv:2404.06483) supplying what it omits and
RhythmFormer (arXiv:2402.12788), FreTS (NeurIPS 36) and CMamba (arXiv:2406.05316)
pinning values that both leave open. All five papers are in `research/`.

**0.9327M parameters, 79.15M MACs per frame.** Published: 0.91M and 80.82M, so
+2.5% and -2.0%. Those two numbers are the only quantitative constraints on the
parts the paper does not specify, and `tests/test_budget.py` enforces them. The MAC
figure excludes the scan, which is a fused kernel and opaque to
`FlopCounterMode`.

The scan itself is Mamba-3 rather than the Mamba-1 selective scan all three source
papers call; see §4.1.

Input `(B, T, 3, 128, 128)` in `[0, 1]` plus a skin mask `(B, 128, 128)`.
Output `(B, T)`: one BVP sample per input frame. Heart rate is not predicted; it is
read off the waveform afterwards by band-pass and beat timing -- the median
inter-beat interval. See section 8.

```
(B, T, 3, 128, 128)
  Fusion Stem   ->  (B, T, 80, 16, 16)     raw frames fused with four differences
  PGA           ->  (B, T, 80)             space collapsed into channels
  4 x Block     ->  (B, T, 80)             Mamba + CAM, then DF-FFN
  Predictor     ->  (B, T)                 the BVP waveform
```

| | |
|---|---|
| dim | 80 |
| depth | 4 |
| stem width | 16 |
| DF-FFN latent | 160 |
| clip | 300 frames = 10.0 s at 30 fps (the papers use 160; see below) |
| input | 128 x 128 |

---

## 1. Space is collapsed before the scan

This is the basis the rest of the design depends on, and the main difference from
PhysMamba, the other Mamba-based rPPG model. PhysMamba flattens **space and
time together** into one token sequence (`n_tokens = nf * H * W` in
`tools/rPPG-Toolbox/neural_methods/model/PhysMamba.py`). RhythmMamba measured what
that costs (its Table 4):

| tokens fed to Mamba | MAE | RMSE |
|---|---|---|
| 8x8 spatial | 4.90 | 10.14 |
| 4x4 | 4.62 | 8.87 |
| 4x4 + positional embedding | 5.47 | 10.34 |
| 2x2 | 4.69 | 9.97 |
| **1x1 (spatial average pool)** | **3.54** | **7.68** |

Spatial structure in the token sequence lowers the score. A selective scan's
state transition is a temporal phase shift; adding spatial dimensions raises the
dimensionality of that transition without giving the recurrence anything it can
use. So PGA pools space away at the earliest possible point, and every block after
it sees a pure time series.

---

## 2. Fusion Stem

`src/model/cfmamba/fusion_stem.py`. 0.0824M parameters.

A face is ~127 LSB of static appearance and the pulse is a 0.1-0.5 LSB change on
top of it, so raw frames ask the network to find a 0.3% modulation under a signal
400x larger. Frame differences cancel the static term but amplify every other
artifact. The stem uses both.

```
temporal shift        X[t-2..t+2], clamped at the clip boundary
four differences      concatenated to 12 channels
stem_raw    Conv2d(3,  16, 7x7, s2, p3) -> BN -> ReLU -> MaxPool(3, s2, p1)   128 -> 32
stem_diff   Conv2d(12, 16, 7x7, s2, p3) -> BN -> ReLU -> MaxPool(3, s2, p1)
stem2       Conv2d(16, 16, 5x5, s1, p2) -> BN -> ReLU -> MaxPool(3, s2, p1)    32 -> 16
X_fusion  = stem2(0.5*X_raw + 0.5*X_diff) + stem2(X_diff)
stem3       Conv3d(16, 80, (2,5,5), pad (0,2,2)) -> BN3d -> ReLU
```

Boundaries **clamp, they do not wrap**. Wrapping would join a recording's last
frame to its first and manufacture a step discontinuity at exactly the frequencies
rPPG reads.

`stem3`'s time kernel is 2, so the time axis is padded by one at the front. Without
that pad the output is T-1 frames, every layer downstream accepts it, and every
target sits one frame out of step with its features. Nothing raises.

**Three values here are not free parameters.** RhythmFormer ablates them: fusion
ratio 5:5 (its Table 8 — 3.07 MAE against 4.56 for raw alone and 4.39 for
differences alone), ±2 adjacent frames (Table 6 — 3.07 against 4.19 at ±1), frame
step 1 (Table 7).

**The two source papers disagree on the geometry** and CFMamba references only
RhythmMamba, so both are implemented and the cost budget arbitrates:

| | RhythmMamba §3.2 | RhythmFormer §3.2 |
|---|---|---|
| stem1 kernel | 7x7 | 5x5 |
| stem2 kernel | 7x7 in text | 3x3 |
| diff order | reverse chronological | chronological |
| MaxPool in stem2 | yes | no |
| output | H/8 = 16x16 | H/4 = 32x32 |

`stem2` at 5x5 is neither paper's given value. RhythmMamba's text only constrains
that convolution to stride 1 — the 16x16 arithmetic fails otherwise — so its kernel
was never pinned, and 5 is what reproduces the published cost. Measured over the
whole model at the time of that search: 82.5M MACs/frame at 5, against 94.5M at 7
and 74.1M at 3. The current model, after the Mamba-3 swap, is 79.15M.

The diff ordering makes no difference to what the stem can represent: reverse is
chronological negated, and the first convolution is linear. It follows
RhythmMamba's text regardless.

---

## 3. PGA — physiology-guided attention

`src/model/cfmamba/pga.py`. **Zero parameters.**

Once space is pooled away, *where* the signal was is gone. RhythmMamba's
answer was a learned sigmoid gate; CFMamba's objection is that a purely data-driven
gate has nothing anchoring it to anatomy, so under motion it strays onto hair,
edges and background (its Fig. 3a). PGA multiplies two maps before pooling.

```
M_prior(h,w) = exp( -((h-c_h)^2 + (w-c_w)^2) / (2 sigma^2) )        Eq. 1
A_feat[c,t]  = X[c,t] / (eps + spatial_mean(X[c,t])) * gamma        Eq. 2
A_tilde      = A_feat * M_prior                                    Eq. 3
A_spatial    = H*W * A_tilde / (2 * ||A_tilde||_1)                  Eq. 4
x_seq[t]     = spatial_mean( X[t] * A_spatial[t] )                  Eq. 5
```

`(c_h, c_w)` and `sigma` are the **skin mask's own first and second moments**, so
the prior tracks the subject rather than assuming a centred face, and it is
parameter-free so it cannot be trained away. An empty mask degrades to a centred
bias rather than a NaN.

Eq. 2 is a divisive normalisation rather than a sigmoid, which is the substantive
difference from RhythmMamba. A sigmoid saturates and loses the ratio between two
bright regions; a divisive normalisation keeps it, so a channel can represent
"twice as much here".

Eq. 4's constant is confirmed against the source it references: EfficientPhys and TS-CAN
both implement `x / sum(x) * H * W * 0.5`
(`tools/rPPG-Toolbox/neural_methods/model/EfficientPhys.py`).

Removing PGA costs the paper 0.36 -> 0.50 MAE on UBFC and 4.03 -> 5.72 on VIPL-HR.

---

## 4. Block: Mamba + CAM, then DF-FFN

`src/model/cfmamba/block.py`. Post-norm, following RhythmMamba Fig. 3.

```
mixed = mamba(x); if cam: mixed = cam(mixed)
x     = norm1(x + mixed)
x     = norm2(x + ffn(x))
```

CAM modulates the **Mamba branch before it joins the residual stream**, not after
the addition. Section 3.2 states it "operates on the temporal representations
produced by the state space model"; applying it after would let it rescale the skip
connection too, compounding the attenuation across 4 layers.

### 4.1 Multi-temporal constraint Mamba

`src/model/cfmamba/mamba_layer.py`. RhythmMamba §3.3.

One Mamba block sees the clip whole, in halves, and in quarters — three paths,
`2^(i-1)` slices each, **one shared set of weights**. RhythmMamba is explicit that
this is a *constraint*, not a fusion: "we replace multi-temporal fusion with
multi-temporal constraint". Allocating a Mamba per path would turn it into a
multi-scale ensemble and change what the layer means, so the sharing is checked in
the tests.

```
X_mamba = sum_i X_path_i * silu(Proj(X_stem))          Eq. 4
```

**The scan inside is Mamba-3** (Lahoti et al., arXiv:2603.15569), not the Mamba-1
selective scan all three source papers call. The block around it is unchanged:
same `(B, T, C)` in and out, same CAM, same DF-FFN, same residuals. Four changes to
the recurrence:

| | Mamba-1 | Mamba-3 |
|---|---|---|
| discretisation | exponential-Euler, `h_t = α_t h_{t-1} + Δ_t B_t x_t` | exponential-trapezoidal, `h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t` |
| state transition | real decay | decay × data-dependent rotation, applied to B and C as a RoPE |
| B, C | unnormalised | RMS-normalised, then a learnable bias initialised to 1 |
| short conv | width-4 depthwise | **none** |

`β_t = (1-λ_t) Δ_t exp(Δ_t A_t)`, `γ_t = λ_t Δ_t`, `λ_t = σ(u_t)` projected from
the input (Prop. 1). The trapezoid is second-order where Euler is first, which
matters at 1–3 Hz sampled at 30 Hz: ten to thirty samples per cycle, so the
discretisation error is *phase* error. The rotation is RhythmMamba's own
justification for using Mamba — "a state transition is the temporal phase shift of
the rPPG signal" — taken literally.

The conv is gone because the trapezoid *is* an implicit width-2 convolution on
`B_t x_t` (§3.1.2). Table 5a measures adding it back as worse: 15.72 ppl against
15.85 for "Mamba-3 + conv".

`d_state=16, expand=2, headdim=32, mimo_rank=1, rope_fraction=1.0, chunk_size=32`.
`d_state` carries over from the Mamba-1 fit rather than rising to Mamba-3's default
of 128, which is sized for a 1.5B language model. `mimo_rank=1` keeps the
single-input single-output recurrence Mamba-1 and Mamba-2 have; Mamba-3's MIMO
extension (its section 3.2) raises the rank of the `B`/`C` outer product, and at
rank 1 the two are the same operator. `rope_fraction=1.0` rotates the whole state
for `d_state/2 = 8` angles (Prop. 2); 0.5 would leave four, too coarse for a heart
rate. `chunk_size=32` because the quartered path is `T/4` -- 75 frames at the 300
this trains on -- and 64 leaves it no cross-chunk recurrence.

`mimo_rank` is a constructor argument on `CFMambaPhys` and `MultiTemporalMamba`,
not a `TrainConfig` field: nothing in the lineage measures it.

Measured, this costs less than what it replaced: 0.9559M -> 0.9327M, and the error
against the published 0.91M, +5.0% -> +2.5%.

**Direction.** Unidirectional by default. Neither paper states which, and the
evidence splits: RhythmMamba's phase-shift argument reads as directional, and
neither `mamba_ssm.Mamba` nor `Mamba3` has a `bimamba` flag (PhysMamba's came from a
fork); against that, a pulse is not causal in any physical sense and an offline
model has the entire clip. `TrainConfig.direction` leaves it to measurement:
`"shared"` scans the reversed sequence with the same weights and adds no
parameters, `"separate"` is Vim-style and roughly doubles the SSM parameter count.

Measured: unidirectional influence of the last frame on every earlier position is
**exactly 0.0**; `"shared"` puts 3.7e-2 three steps back, decaying to ~1e-4 at 31.

### 4.2 CAM — channel-adaptive modulation

`src/model/cfmamba/cam.py`. CFMamba Eqs. 6-8.

Mamba runs each channel through its own recurrence and mixes only by linear
projection, so it cannot down-weight a channel bringing a motion artifact. CAM
estimates channel reliability from the whole clip.

```
z = AvgPool_T(X), MaxPool_T(X)
W = sigmoid(MLP_scale(z));  B = sigmoid(MLP_shift(z))
X' = W * X + B                                          broadcast over T
```

Two things come from reading CMamba, which CFMamba ports this from, and neither is
inferable from CFMamba alone:

- **The hidden layer is set by an expansion rate, not an SE-net bottleneck ratio.**
  CMamba calls it "expansion rate r" and reports that too small a value underfits.
  Sizing it as a bottleneck costs ~28k parameters per block at dim=96.

  **The value is not recoverable from either paper.** CFMamba states only "two
  lightweight multilayer perceptrons". CMamba uses "expansion rate" for two
  different modules and prints a number for the wrong one: its A.2 fixes "the
  expansion rate of the linear layer at 1" for **M-Mamba** (that is Mamba's
  `expand`, not the MLP), while the GDD-MLP's `r` appears only in a figure, never as a number
  whose axis labels are rasterised. `r = 1.0` is used here on the word rather than
  a number: it is the smallest setting that is really an *expansion*, making the
  hidden layer as wide as the stream.

  It costs +5.0% on the published parameter count, and that is the whole of the
  overshoot. Re-fitting the other widths around it is worse, not better -- with `r`
  pinned the best available is `dim=76 / ffn=152 / stem=16` at -4.42% parameters and
  -3.2% MACs, 7.6% total against 7.0% for keeping `dim=80`. So the widths stay where
  every configuration search put them.

  Note CMamba's M-Mamba `expand=1` does **not** transfer: CFMamba references [14,15] for
  its SSM and CMamba only for the channel-descriptor idea, so `expand=2` (Mamba's
  default, corroborated by PhysMamba's vendored source) is correct here.
- **CMamba runs each descriptor through the shared MLP and sums the outputs;**
  CFMamba Eq. 6 sums the descriptors first. These are not equivalent — summing
  first lets a large average cancel a large max before either reaches a
  non-linearity. **CMamba's is the default here**, because it is the version with an
  ablation behind it. `pooling="cfmamba"` restores the literal Eq. 6.

Removing CAM costs 0.62 -> 0.78 RMSE on UBFC, 6.59 -> 8.63 on VIPL-HR.

### 4.3 DF-FFN — dual-frequency feed-forward

`src/model/cfmamba/df_ffn.py`, over `cs_ffn.py`, `pts_ffn.py`, `band_mask.py`,
`complex_linear.py`.

A pointwise time-domain FFN cannot see a periodicity spread over a whole clip. So
`Linear(80 -> 160)`, two spectral stages, `Linear(160 -> 80)`.

**CS-FFN** (Eqs. 9-12) — FFT along the **channel** axis, complex linear `160x160`
shared across timestamps, iFFT. RhythmMamba's frequency FFN unchanged.

#### Section 3.3 omits the activation

Every operation the section names is linear: linear expansion, FFT, complex linear,
iFFT, Gaussian gating, iFFT, linear projection. The only sigmoid in it constrains
`f_c` and `b_w`, which are parameters rather than activations. Implemented as
written, the DF-FFN is one linear operator -- measured by superposition, relative
affine error `2.2e-07`, against `6.4e-01` for a two-layer MLP. Four stacked linear
blocks add nothing the Mamba layers do not already do, which does not match an
ablation measured at 0.36 against 0.59 MAE.

FreTS, the work CFMamba references for the complex linear, has one: its Eq. 7
applies the activation to the real and imaginary parts separately and restacks
them. That is where it goes here, inside both complex linears, so `ComplexLinear`
stays Eqs. 10-11 and `complex_activation` is the other half of what FreTS calls a
FreMLP.

`ffn_activation` defaults to **GELU** (`src/model/cfmamba/model.py`,
`df_ffn.py`, `TrainConfig`). `"relu"` is FreTS's own choice and `None` restores the
literal Section 3.3 reading, which is linear. ReLU on an imaginary part discards
half the phase plane, so which of the three a pulse wants is left to the ablation.

**PTS-FFN** (Eqs. 13-18) — FFT along **time**, the learnable Gaussian band mask,
complex linear, iFFT.

```
f_c  = 0.75 + sigmoid(theta) * (2.5 - 0.75)   Hz        Eq. 14
b_w  = 0.2  + sigmoid(theta) * (1.0 - 0.2)    Hz
M(f) = exp(-(f-f_c)^2 / 2 b_w^2) + exp(-(f+f_c)^2 / 2 b_w^2)   Eq. 15
```

The sigmoid is a hard constraint: no setting of the weights can centre the filter
outside 45-150 bpm, so it stays a physiological prior rather than a parameter that
can settle on a motion artifact.

Eq. 15 is two Gaussians because a real signal's spectrum is conjugate symmetric:
the peak appears at +f_c and -f_c. A single lobe would keep one, halve the energy
and leave the inverse transform complex. Hence full `fft`, not `rfft`.

Frequencies are Hz, never bin indices. Bin *k* means `k*fps/T`, so a mask built from
bin numbers is a different filter at every clip length while raising nothing. In the
model this replaces, a fixed bin list meant 45-202 bpm at T=160 and 24-108 bpm at
T=300. `tests/test_band_mask.py` holds the peak to the same Hz across
T = 100/160/300/450.

#### Eq. 17, where the paper conflicts with itself

Its text — "applied to each channel individually, with weights shared across all N
channels" — reads as a `(T, T)` matrix over the frequency axis. But the same
sentence points at Eqs. 10-11 for the operation, and those define `W in C^(NxN)`.
Two outside facts break the tie:

- **FreTS**, which CFMamba references for the method, transforms along one axis while
  applying `W in C^(dxd)` to a *different* one (its Eq. 4). The transformed axis
  acts as a batch dimension. CFMamba folded FreTS's three axes into two,
  which is how the text came to read ambiguously.
- **CFMamba Table 4 measured cost on a 900-frame clip** while §4.1 trained on
  160-frame segments. A `(T, T)` weight cannot do both.

So `pts_mode="channel"` is the default. `"full"` implements the literal-text
alternative, and `tests/test_budget.py` shows it inconsistent with the paper's own
experiment. Two further values exist for the ablation: `"diagonal"`, a per-channel
complex gain rather than a full `(N, N)` mixing matrix, and `"none"`, which drops
the projection and leaves the Gaussian mask alone. All four are fields on
`TrainConfig`.

---

## 5. Predictor

`Conv1d(80, 80, k=5) -> SiLU -> Conv1d(80, 1, k=1)` -> `(B, T)`. 0.0322M
parameters. A convolution rather than a Linear: the map from local temporal context
to pulse amplitude is the same at every instant, so it shares weights across time
and holds for any T.

---

## 6. Loss

`src/model/losses.py`. `L = 0.8 * L_time + 1.0 * L_freq` (Eq. 19).

`L_time` is negative Pearson — scale- and offset-invariant, which is required: the
network sees skin brightness and cannot know the contact sensor's gain, so demanding
absolute agreement would penalise a correctly shaped prediction for being the wrong
size.

`L_freq` is cross-entropy between the predicted spectrum and the target's dominant
rate, over 105 candidates at 1 bpm resolution covering 45-149 bpm. It exists because
correlation is indifferent to *which* periodicity was locked onto: a prediction at
double the true rate can correlate respectably while being wrong by 72 bpm.

### The weights

CFMamba Eq. 19 states `alpha` and `beta` as symbols and gives no values.
RhythmFormer Section 3.4 supplies 0.2 and 1.0 for the same construction, and its
Table 13 records what the balance does:

| terms | MAE | RMSE |
|---|---|---|
| time only | 3.56 | 8.17 |
| freq only | 13.32 | 16.54 |
| both | 3.13 | 6.98 |

This project ran at `alpha = 0.2` and measured, over a 15-epoch UBFC run, that the
temporal term stopped moving from epoch 2 at 0.674 train / 0.448 dev, and that at
that weight it was ~1.5% of the total loss. **`alpha` is 0.8 here**, a departure
from RhythmFormer on that measurement. `beta` stays at 1.0. Loss totals from runs
before the change are on a different scale and are not comparable term by term.

The frequency term is written here rather than imported from the toolbox because
`TorchLossComputer.Frequency_loss` obtains its label through
`calculate_metric_per_video`, which hardcodes a first-order 0.6-3.3 Hz band-pass.
Importing it would put a 36-198 bpm band inside the *loss*.

---

## 7. Data

Loader: `src/model/dataset.py`.

```
video -> ffmpeg decode at 30*k fps, area filter if shrinking
      -> YuNet median face box, square, at SOURCE resolution
      -> random 87.5% sub-window of the box, random horizontal flip
      -> ONE resample to 128            area if shrinking, linear if enlarging
      -> CFMamba-Phys
      -> BVP waveform
```

The SegFace skin mask is cached once per clip at 256, and the same sub-window is
mapped into that frame and resampled alongside. Frames are never routed through
256; the mask's resolution is the mask's business.

### One resample

The chain used to be `box -> 256 -> crop 224 -> 128`: three resampling steps to
serve one crop. Two things were wrong with it.

**94% of clips have a face box smaller than 256**, so the first step was an
*enlargement* -- median box side is 210 px for UBFC and 191 px for MCD. And
`cv2.INTER_AREA` enlarging is **exactly equal to `INTER_NEAREST`**: measured, a 4x4
ramp taken to 8x8 comes back with 16 distinct values against `INTER_LINEAR`'s 44.
So 94% of clips were pixel-duplicated up to 256 and then area-averaged back down to
128, for nothing.

Measured impact on the recovered pulse: **none**. Across eight UBFC clips the
spectral peak was identical to the tenth of a bpm and prominence moved a few percent
in both directions. Resampling artefacts are high-frequency and PGA's spatial pooling
averages them away. So this was costing precision, not signal -- which is still worth
not paying.

Three rules now hold, in order of how much they matter for a 0.1-0.5 LSB signal:

1. **Float before any resampling.** The 8-bit decode is the camera's quantisation.
   Resizing in uint8 rounds interpolated values and injects +/-0.5 LSB, the same
   size as the thing being measured.
2. **Area filter when shrinking.** It integrates every source pixel landing in an
   output pixel, so the local mean -- which is the entire signal -- is preserved
   exactly. Verified: a uniform +0.3 step survives a 235->128 reduction as exactly
   +0.3 everywhere. Linear point-samples and aliases.
3. **Linear when enlarging.** 12.7% of MCD clips have an 87.5% crop below 128 px
   (smallest is 94), and for those area filtering would be nearest-neighbour.

`ffmpeg` gets `scale=...:flags=area` when it is shrinking, for the same reason --
swscale defaults to bicubic, which point-samples. UBFC and MCD are already 640x480
so it is a no-op for them; it matters for anything higher-resolution.

Augmentation, all train-only:

- **Random horizontal flip**, frames and skin mask together. Separately would aim
  PGA's prior at the mirror image of the face the model is looking at, and nothing
  would raise.
- **HR-balanced temporal resampling** (RhythmFormer §4.3): a window whose own rate
  exceeds 90 bpm is stretched, below 75 bpm compressed. The apparent rate is
  `hr_true / k`, so the decode timebase and the PPG sampling timebase must move
  together — that alignment is what `tests/test_augment.py` exists for.
- **Segment jitter — removed.** Enumerated starts used to move by up to ±0.5 s in
  training, on the argument that a segment boundary is an artifact of the
  enumeration. It was the only thing breaking the strict non-overlap both source
  papers score under: at ±0.5 s two adjacent 5.33 s windows could share a second of
  footage, so each window escaped into its neighbours. Starts are now exact.

**The augmentation used to be fixed.** The per-item RNG was seeded
`seed * 1_000_003 + index`, and nothing altered `seed` between epochs, so every
segment saw one fixed crop, one fixed flip and one fixed `k` for a whole run — a
50/50 partition assigned once rather than an augmentation. Measured: segment 0 took
crop (22, 19) and flip `False` in every one of epochs 0, 1, 2 and 3. Training now draws from
one continuous per-worker stream. Reseeding from `torch.initial_seed()`, the usual
fix, does not work here: `persistent_workers=True` keeps workers alive across
epochs so their base seed never changes either. Evaluation keeps the per-index seed,
because a scored window has to be the same window every time.

**Skin masking is off by default.** None of the three papers masks, and the mask's
job here is to supply PGA's prior. `TrainConfig.apply_skin_mask` zeroes non-skin
pixels for the ablation.

### The face box, measured

The toolbox all three papers used enlarges the detected box by `LARGE_BOX_COEF=1.5`
and detects on the **first frame only** with a Haar Cascade. This pipeline uses
YuNet, a median over 24 sampled frames, and 1.25x. Haar Cascade falls back to the
entire frame when it fails, and a first-frame box is why RhythmFormer's Table 11
caps input at 160 frames ("subsequent frames may lose the face due to motion").

**This project trains at 300 frames, not 160.** It halves the FFT bin spacing, from
11.25 bpm to 6.0 bpm. It also means numbers produced
here are not directly comparable with the published ones, and RhythmFormer's
Table 11 measures 160 as the better length (3.07 MAE against 3.86 at 320). The
band mask is unaffected -- it is parameterised in Hz precisely so that changing T
cannot move it.

The crop size was the open question, so it was measured with POS, which needs no
training and therefore reads how much pulse survives each crop directly. 120
random UBFC segments, same windows both times:

| crop | box side | skin | POS MAE | POS RMSE | POS rho | CHROM MAE |
|---|---|---|---|---|---|---|
| 1.25x (ours) | 210 px | 34.5% | 3.34 | 9.32 | +0.845 | **6.56** |
| 1.50x (toolbox) | 253 px | 24.0% | **3.22** | **8.82** | **+0.870** | 6.68 |

1.5x is 4% better on POS and 2% worse on CHROM, which at n=120 is within noise.
1.25x is kept, because the difference does not justify rebuilding 3692 SegFace masks
at roughly 90 s each. `median_face_box(pad=0.5)` switches to 1.5x if a rebuild
happens for another reason.

For reading any result: POS scores 1.66 bpm MAE on the UBFC paper protocol's 12 test
subjects and 3.34 bpm on a random sample across all 48. The floor moves by 2x
depending on which subjects land in the split, which is why metrics are reported per
subject and per source rather than as one number.

**Ground-truth heart rate comes from the contact PPG, never the label column.**
`DATASETS.md` records subject24 labelled 96 bpm against a PPG reading of 127.2, plus
four more UBFC subjects whose HR readout drops out while the waveform stays intact.

---

## 8. Protocol

One split over the pooled corpora, assigned once and persisted. `src.cli combine`
pools UBFC, MR-NIRP and MCD into `build/clips_all.parquet` at 90/3/7, grouped by
subject and stratified by source so each corpus reaches dev and test. `train` reads
that column rather than deriving a split, so the partition cannot move when the
manifest grows. A manifest with no `split` column gets a subject-grouped 85/10/5
derived at load time, and the run prints which of the two it used.

The published UBFC protocol (first 30 subjects train, last 12 test) was removed:
it only understands UBFC's `subjectN` ids, and on the pooled manifest it silently
placed 606 of 666 subjects in train. Results here are not comparable with the
papers' tables in any case — they use 160-frame windows and this trains at 300.

Clips are not filtered by a pulse-prominence screen. Selecting clips where a simple
spectral estimator already agrees with the label makes any later result circular.

**The reported result is the last epoch.** Both papers report that, so the epoch
budget is chosen in advance -- 50 by default. A dev split is scored each epoch on a
fixed subsample of 1500 segments for the trajectory, and `<run-dir>/best.pt` is
written whenever an epoch sets a new lowest dev loss, because `last.pt` is
overwritten each epoch and cannot be rolled back. `best.pt` is a second artefact and
is scored only if `--model` points at it; the reported number stays the last
epoch's.

Evaluation lists every non-overlapping segment, not one centred window per clip, and
reports per source: MCD is 98.45% of the segments, so an aggregate over any split is
largely a measurement of MCD.

POS and CHROM are the classical floor. Neither needs training, and both publish
~4.06-4.08 bpm MAE on UBFC in every table in both papers. `src.cli baseline` scores
them on the same windows the model is scored on.

---

## 9. What the reconstruction does and does not establish

The published 0.91M parameters and 80.82M MACs/frame are two equations in several
unknowns. They cannot recover the architecture; they can and do rule configurations
out.

Reading the referenced sources moved the fit. Before FreTS and CMamba were read,
PTS-FFN was a `(T, T)` frequency matrix and CAM was an SE-net bottleneck. Both were
wrong in the direction of too few parameters, and correcting them took the fit from
-2.9%/+2.1% to -0.6%/-2.0% under the Mamba-1 scan. The current Mamba-3
configuration sits at +2.5%/-2.0%.

What held across every configuration search, before and after those corrections, is
depth 4 and a stem width of 16. `dim=80` with a 2x FFN latent is one fit among
several that land inside tolerance.

**Where the budget gives no signal.** CAM is the only module that is
parameter-heavy and compute-free, because Eq. 6 pools over T before the MLPs, so
they run once per clip on a `(B, C)` vector rather than per frame. Measured, MACs
per frame are identical to the second decimal from `r = 0.25` to `r = 4.0` while
parameters move 0.879M -> 1.265M. The FLOP budget says nothing about that setting
and the parameter budget is the only evidence, which is why it is the one place a
textual argument overrules it.

**Unverified and open:** whether the SSM is bidirectional (see section 4.1); the
learning rate, which no paper in the lineage states; CAM's expansion rate, which
CMamba sweeps in a figure without printing the chosen value; and which activation
the DF-FFN's omitted non-linearity should be (section 4.3).

**Where the paper conflicts with itself.** Section 3.1 states the facial region is
detected "for every frames"; Section 4.1 states "facial recognition was executed
solely on the first frame of each segment". This pipeline does neither: a median box
over 24 sampled frames, measured as no worse than a 1.5x first-frame crop (see the
face box table above).

---

## 10. Known limits

**The corpus, not the architecture, is the binding constraint.** UBFC-rPPG ships 50
subjects across two releases: 8 in DATASET_1 and 42 in DATASET_2, ~55 minutes of
video, of which 48 clips reach the pooled manifest. The paper reports 0.36 bpm MAE;
this copy of UBFC differs -- three subjects run at 23.2 fps, five have a broken HR
readout, and the decode and crop path is this project's rather than the toolbox's.

**The skin mask is one median mask per clip**, so it does not track subject motion.
Stable for UBFC's seated recordings.

**Blood pressure is out of scope.** The target is the pulse waveform and the
heart rate read off it. The manifest still carries nullable `sbp_mmhg` and
`dbp_mmhg` columns so MCD's cuff readings are retained, but nothing reads them and
no head predicts them.
