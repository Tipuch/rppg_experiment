> **Status.** Built, with three decisions changed after measurement:
> **256, not 512** (MambaVision's windows divide 224 exactly, and MCD's native face
> box is only ~126 px, so 512 was upscaling nothing into 4x the memory);
> **brightness normalisation off by default** (subtracting each frame's mean skin
> luma removes the spatially uniform component of the pulse by construction, so the
> `mean_Y` branch is off too); **YuNet for the box**, not MambaVision, which has no
> detection head. See ARCHITECTURE.md and DATASETS.md for current state.

# Plan: skin-isolated, brightness-normalised frame pipeline

Requested: MambaVision face box -> 512x512 resize -> mean Y -> SegFace skin mask,
subtract mean Y, black out non-skin. Output: frames of brightness-normalised
facial skin.

The goal is sound and most of it is buildable today. Four things need a decision
first: two are blocking, two change the result quality.

## 1. Blocking: MambaVision cannot find the face box

Verified against the installed package. `mambavision` exposes exactly two things,
`create_model` and `models`, and its factories are all classifiers:

    mamba_vision_T / T2 / S / B / L / L2 / L3 (+ _21k, _512_21k variants)

Source contains no `bbox`, `anchor`, `roi_head` or `RPN`; `num_classes=1000`.
It is an ImageNet **backbone**, not a detector. To get boxes from it you would
have to bolt on a detection head and train it on named face boxes -- weeks of
work, and it would end up worse than an off-the-shelf detector.

Three ways forward:

**(a) Derive the box from SegFace itself (recommended).** SegFace already
produces a skin mask. The bounding box of that mask is the face box. This drops
the detector from the pipeline entirely -- one model instead of two, and the box
is guaranteed consistent with the mask we later apply. Needs a coarse first pass
(SegFace on a downscaled full frame) to locate, then a second pass at 512 on the
crop.

**(b) Keep YuNet**, already working in `src/aggregation/face.py`, median box per
clip. Cheap, stable, confirmed on this data.

**(c) Use MambaVision as the rPPG *model* backbone**, which is what it is good
for, and let (a) or (b) handle detection. This is likely what was intended --
MambaVision fits downstream of this pipeline, not inside it.

## 2. Blocking: 512x512 does not fit on disk

Measured, at 180 s / 30 fps per MCD recording, 3600 recordings:

| resolution | per recording | all 3600 |
|---|---|---|
| 512x512x3 | 4.25 GB | **15.3 TB** |
| 256x256x3 | 1.06 GB | 3.8 TB |
| 128x128x3 | 0.27 GB | 1.0 TB |

641 GB free. Even the current 128x128 store does not scale to full MCD.

Lossy video compression is not a way out. The rPPG signal is a sub-percent
fluctuation in pixel value; H.264 at any normal CRF quantises it away. Lossless
(FFV1) on mostly-black masked frames would help, but the remaining skin region
still outweighs and 3-5x is optimistic.

Options, cheapest first:

| option | all 3600 | keeps |
|---|---|---|
| 3 windows/recording @128 | 159 GB | plenty of data, current resolution |
| 3 windows/recording @256 | 637 GB | borderline, no margin |
| 3 windows/recording @512 | 2.5 TB | no |
| full recording @128 | 1.0 TB | no |

Recommendation: **sample N windows per recording rather than taking all 18.**
A 180 s clip at rest is highly redundant; 3 well-spread 10 s windows carry nearly
the same information as 18. That also rebalances the dataset -- MCD would
otherwise swamp CLBP-300 by 18:1 on window count alone.

If 512 is required for a specific reason, it is affordable on a **subset**
(e.g. 300 recordings @512, 3 windows = 212 GB).

## 3. Reorder: mean Y must be measured over skin, not the whole frame

The requested order is resize -> mean Y -> segment. Computing mean Y before
segmentation averages background, hair, clothing and any wall behind the subject
into the number. Those move independently of the face and inject illumination
noise straight into the value we then subtract from every pixel.

Correct order:

    detect -> square crop -> resize 512 -> SegFace mask -> mean Y over MASK ONLY
      -> subtract -> black out non-skin

Same steps, one swap, and the saved mean-Y trace becomes a clean skin-brightness
signal instead of a room-brightness signal.

## 4. Result to be intentional about: the subtraction removes the pulse

This is the part worth being explicit about, because it determines how the output
gets consumed.

The rPPG signal *is* a small, spatially near-uniform brightness change across
skin. Subtracting the per-frame spatial mean of Y over the skin removes exactly
that global component. After this step the frames no longer carry the dominant
pulse signal.

That is not a bug in the design, because the mean Y is being saved. The pipeline
factorises each clip into:

    mean_Y[t]        1-D trace, carries most of the pulse
    frames[t]        spatially detrended skin, carries local/regional variation
                     (forehead and cheek do not pulse in perfect synchrony)

Both halves are informative and the split is justifiable -- it is close to what
CHROM and POS do analytically. **But a model trained on `frames` alone will
underperform badly**, because the signal it needs was moved into `mean_Y`. The
downstream model must consume both. Worth determining now, since it shapes the
model interface, not just storage.

Two routine details that follow:

- `Y - mean_Y` goes negative. Clipping to uint8 ruins half the residual.
  Store as `Y - mean_Y + 128` in uint8 (offset, invertible, keeps the existing
  store format), or as int16/float16 if the extra precision is wanted.
- Recentre only Y. Convert back to RGB with the original Cb/Cr so colour
  information survives, since chrominance carries pulse too.

## 5. Geometry: avoiding deformation

Least deformation is achieved by **expanding the face box to a square before
resizing**, not by letterboxing. A square crop resampled to 512x512 has an
aspect ratio of exactly 1:1 -- zero distortion, no padding, no wasted pixels.

Letterbox padding is only needed when the square would run past the frame edge.
Clamp the square inward instead; the existing `face.py` already does this.

Use `INTER_AREA` for downscaling -- it averages rather than samples, which
matters when the signal is a sub-percent brightness fluctuation.

## 6. Mask stability

Per-frame segmentation flickers at the mask boundary. A mask whose area changes
frame to frame makes `mean_Y[t]` jump for reasons that have nothing to do with
blood volume -- a fake signal at exactly the frequencies we care about.

Mitigate by computing the mask on sampled frames and taking a **median mask per
clip**, matching the median-box approach already used for detection. Faces are
near-stationary in these recordings, so a fixed mask is reasonable. Re-estimate
per window if motion is significant (MCD has left/right view recordings).

Record `mask_area_px` per frame regardless, so instability is measurable rather
than invisible.

## 7. Proposed build order

1. `segface.py` -- fetch weights from `kartiknarayan/SegFace` (MIT licence,
   Swin-Base 512, CelebAMask-HQ, 88.96 mean F1), wrap inference, expose
   `skin_mask(frame) -> bool array`. Verify the skin class index against
   CelebAMask-HQ's label map on real frames before trusting it.
2. `roi.py` -- square-expand + `INTER_AREA` resize to the chosen resolution.
   Box from SegFace mask (option 1a) or YuNet (1b).
3. `normalise.py` -- RGB->YCbCr, median mask, mean Y over mask, `Y-mean+128`,
   back to RGB, zero non-skin.
4. Extend the frame store: alongside `frames.npy`, write `mean_y.npy` (float32,
   one value per frame) and `mask_area.npy`. Add `mean_y_path` to the Polars
   schema so the trace moves with the row.
5. Re-run over MCD/CLBP-300 with the sampling policy from section 2.

## 8. Decisions needed

1. Face box: SegFace-derived (a), YuNet (b), or something else?
2. Resolution and sampling: 128 with 3 windows/recording (159 GB) is the only
   combination with real margin. Accept, or pick a different point?
3. Confirm the downstream model will consume `mean_Y[t]` alongside the frames.
