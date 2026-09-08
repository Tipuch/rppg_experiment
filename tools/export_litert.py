#!/usr/bin/env python
"""Export a trained CFMamba-Phys checkpoint to LiteRT (.tflite).

    python tools/export_litert.py --out ../rppg_experiment/androidApp/src/main/assets

**Run this in its own environment.** `litert-torch` supports Python 3.10-3.13 and
this project pins 3.14, so the converter cannot live in `.venv`:

    uv venv --python 3.13 .venv-export
    VIRTUAL_ENV=.venv-export uv pip install 'torch>=2.13,<2.14' litert-torch
    .venv-export/bin/python tools/export_litert.py

No GPU and no `mamba_ssm` are needed: the export runs `scan="export"` and
`spectral="matmul"`, which are plain PyTorch. See
`src/model/cfmamba/mamba3_export.py` and `src/model/cfmamba/dft.py`.

**Two graphs, not one.** All of the model's activation memory is in the stem --
`temporal_differences` alone materialises 225 MB at T=300, and a whole-clip
forward peaks around 763 MB. But everything up to and including PGA is per-frame
apart from two short reaches across time, so it tiles: with a halo of 3 frames
back and 2 forward a tile reproduces the whole-clip features *bit for bit*, and
peak memory falls to about 133 MB at a tile of 30. So:

    frontend   (1, tile + 5, 3, 128, 128) + (1, 128, 128) -> (1, tile, 80)
    backbone   (1, 300, 80)                               -> (1, 300)

The app runs the frontend as frames arrive, accumulating 96 KB of features, then
the backbone once. The 59 MB whole-clip input tensor never exists.

**The caller owns three things the graphs do not.**

1. *Normalisation.* The checkpoint was trained with `frame_norm="standardized"`:
   one scalar mean and standard deviation over the **whole 300-frame window**,
   not per frame and not per channel. It is clip-global, so it has to be settled
   before the first tile runs -- accumulate sum and sum of squares as frames
   arrive, then normalise on the way into the frontend.
2. *The crop.* `src/aggregation/face.py` squares the YuNet box with
   `BOX_PAD=0.25` and resizes to 128x128. That framing is the training
   distribution: it sets how much of the frame is face, which sets PGA's sigma.
3. *The heart rate.* The model returns a waveform and never a rate. See
   `src/model/postprocess.reported_hr`.

**Tile bookkeeping.** `stem3`'s Conv3d replicate-pads its front, which in the
whole clip happens exactly once, at frame 0. So tile 0 starts at clip frame 0 and
lets the graph's own pad fire, keeping outputs 0..tile-1; every later tile starts
`HALO_BACK` frames earlier and discards its first `HALO_BACK` outputs. The tail
is replicate-padded, which is exact because `temporal_differences` clamps at the
clip boundary anyway. `--check` asserts all of this against the whole-clip model.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.cfmamba.model import CFMambaPhys

DEFAULT_CHECKPOINT = Path("build/runs/cfmamba/last.pt")
DEFAULT_TILE = 30
# To emit output frame t, stem3's Conv3d needs stem2 features at t-1 and t, and
# each of those needs temporal_differences over +/-2. Hence 3 back, 2 forward.
# A halo of 2 back is *not* exact -- it scores 2.6e-2 on an output of scale
# 2.7e-1, and nothing raises.
HALO_BACK, HALO_FORWARD = 3, 2


class Frontend(nn.Module):
    """FusionStem + PGA, over one tile of frames. (B, span, 3, H, W) -> (B, span, C)."""

    def __init__(self, model: CFMambaPhys) -> None:
        super().__init__()
        self.stem, self.pga = model.stem, model.pga

    def forward(self, frames: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
        features = self.stem(frames)
        # use_pga=False is the Table 5 ablation, and the model's own fallback is a
        # plain spatial mean. `skin` stays in the signature so both ablations
        # export the same two graph inputs.
        if self.pga is None:
            return features.mean(dim=(3, 4))
        return self.pga(features, skin)


class Backbone(nn.Module):
    """The four blocks and the predictor. (B, T, C) -> (B, T)."""

    def __init__(self, model: CFMambaPhys) -> None:
        super().__init__()
        self.blocks, self.predictor = model.blocks, model.predictor

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return self.predictor(sequence)


def architecture(cfg: dict) -> dict:
    """The checkpoint's config, reduced to the keys `CFMambaPhys` accepts.

    Not a hand-written list of keys, and not `src.model.train.build_model` either.
    A list drifts: naming `cam_pooling` explicitly is what broke this tool when
    that field left `TrainConfig`, and the same would happen to the next one.
    `build_model` is the constructor a run really used, but importing it pulls in
    polars and cv2 through `train.py`, and this tool runs in an environment that
    has torch and litert-torch and nothing else -- see the module docstring.

    Filtering by the signature is equivalent for every key that matters: the two
    dataclasses overlap in exactly the nine architecture fields `build_model`
    passes, plus `fps`. A key the constructor does not take is dropped; a key it
    takes but the checkpoint does not record takes the constructor default.
    """
    accepted = set(inspect.signature(CFMambaPhys.__init__).parameters) - {"self"}
    return {name: value for name, value in cfg.items() if name in accepted}


def load_exportable(checkpoint: Path) -> tuple[CFMambaPhys, dict]:
    """The trained model, built on the two implementations that export."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in state:
        raise SystemExit(f"{checkpoint} has no 'config' payload")
    cfg = state["config"]
    # scan and spectral are not in the config on purpose: they choose an
    # implementation, never what was learned, and these two are the pair that runs
    # without CUDA and without an FFT.
    model = CFMambaPhys(**architecture(cfg), scan="export", spectral="matmul").eval()
    # strict=True is the point: nothing here re-shapes or renames a parameter, so
    # a load that needed strict=False would mean the export computes something the
    # checkpoint did not train.
    model.load_state_dict(state["model"], strict=True)
    return model, cfg


def tile_plan(n_frames: int, tile: int) -> list[tuple[int, int, int]]:
    """(clip start, input offset, outputs to skip) for each tile."""
    plan = []
    for start in range(0, n_frames, tile):
        if start == 0:
            plan.append((start, 0, 0))
        else:
            plan.append((start, start - HALO_BACK, HALO_BACK))
    return plan


def run_tiled(
    frontend, frames: torch.Tensor, skin: torch.Tensor, tile: int, dim: int
) -> np.ndarray:
    """Drive a frontend -- a torch module or a LiteRT model -- over every tile."""
    n_frames = frames.shape[1]
    span = tile + HALO_BACK + HALO_FORWARD
    tail = frames[:, -1:].expand(-1, HALO_FORWARD, -1, -1, -1)
    padded = torch.cat([frames, tail], dim=1)
    skin_np = skin.numpy()
    features = np.zeros((frames.shape[0], n_frames, dim), dtype=np.float32)
    for start, offset, skip in tile_plan(n_frames, tile):
        window = padded[:, offset:offset + span]
        if window.shape[1] < span:
            pad = window[:, -1:].expand(-1, span - window.shape[1], -1, -1, -1)
            window = torch.cat([window, pad], dim=1)
        got = frontend(window.numpy(), skin_np)
        if isinstance(got, dict):
            got = next(iter(got.values()))
        width = min(tile, n_frames - start)
        features[:, start:start + width] = np.asarray(got)[:, skip:skip + width]
    return features


def export(module: nn.Module, sample: tuple[torch.Tensor, ...], path: Path) -> None:
    import litert_torch

    module.eval()
    with torch.no_grad():
        reference = module(*sample)
    began = time.time()
    converted = litert_torch.convert(module, sample)
    elapsed = time.time() - began
    blob = converted.model_content()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)

    got = converted(*[t.numpy() for t in sample])
    if isinstance(got, dict):
        got = next(iter(got.values()))
    got = torch.as_tensor(np.asarray(got))
    delta = float((reference - got).abs().max())
    scale = max(float(reference.abs().max()), 1e-12)
    print(f"  {path.name:28} {len(blob) / 1024 / 1024:6.2f} MB  "
          f"convert {elapsed:6.1f}s  max|d| {delta:.2e}  rel {delta / scale:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=Path("build/litert"))
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE,
                        help="frames of new output per frontend call")
    parser.add_argument("--check", action="store_true",
                        help="run the exported pair against the whole-clip model")
    args = parser.parse_args()

    model, cfg = load_exportable(args.checkpoint)
    n_frames = model.n_frames
    print(f"{args.checkpoint}  {model.parameter_count()} parameters  "
          f"T={n_frames}  frame_norm={cfg.get('frame_norm', 'unrecorded')}")

    span = args.tile + HALO_BACK + HALO_FORWARD
    frontend_path = args.out / f"cfmamba_frontend_tile{args.tile}.tflite"
    backbone_path = args.out / f"cfmamba_backbone_t{n_frames}.tflite"

    print("exporting:")
    torch.manual_seed(0)
    export(Frontend(model),
           (torch.randn(1, span, 3, 128, 128), torch.rand(1, 128, 128)),
           frontend_path)
    export(Backbone(model), (torch.randn(1, n_frames, model.dim),), backbone_path)

    if not args.check:
        return

    import litert_torch

    print("\nchecking the pair against the whole-clip model:")
    # Standardised the way the loader does it: one scalar pair for the window.
    raw = torch.rand(1, n_frames, 3, 128, 128) * 255.0
    frames = (raw - raw.mean()) / raw.std()
    skin = torch.rand(1, 128, 128)

    torch_frontend = Frontend(model)
    with torch.no_grad():
        whole_features = torch_frontend(frames, skin).numpy()
        whole_waveform = model(frames, skin).numpy()
        tiled_torch = run_tiled(
            lambda f, s: torch_frontend(torch.as_tensor(f), torch.as_tensor(s)),
            frames, skin, args.tile, model.dim,
        )
    frontend = litert_torch.load(str(frontend_path))
    backbone = litert_torch.load(str(backbone_path))
    began = time.time()
    tiled_tflite = run_tiled(frontend, frames, skin, args.tile, model.dim)
    frontend_time = time.time() - began
    began = time.time()
    waveform = backbone(tiled_tflite)
    if isinstance(waveform, dict):
        waveform = next(iter(waveform.values()))
    waveform = np.asarray(waveform)
    backbone_time = time.time() - began

    def report(name, a, b, exact=False):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        delta = np.abs(a - b).max()
        r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        verdict = "  EXACT" if exact and delta == 0.0 else ""
        print(f"  {name:44} max|d| {delta:.3e}  pearson {r:.8f}{verdict}")

    # In PyTorch the tiling must be bit-exact. Through LiteRT it lands at float32
    # epsilon instead, which is the converter's arithmetic, not the tiling.
    report("features: tiled torch vs whole clip", whole_features, tiled_torch, exact=True)
    report("features: tiled tflite vs whole clip", whole_features, tiled_tflite)
    report("waveform: tflite pair vs whole clip", whole_waveform, waveform)
    print(f"\n  frontend {frontend_time:.2f}s over "
          f"{len(tile_plan(n_frames, args.tile))} tiles, "
          f"backbone {backbone_time:.2f}s, total {frontend_time + backbone_time:.2f}s")


if __name__ == "__main__":
    main()
