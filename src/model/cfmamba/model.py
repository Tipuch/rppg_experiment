"""CFMamba-Phys, assembled. Wang et al., Biomed. Signal Process. Control 126 (2026).

    (B, T, 3, 128, 128)
      -> FusionStem          (B, T, C, 16, 16)     raw frames fused with 4 diffs
      -> PGA                 (B, T, C)             space collapsed into channels
      -> L x Block           (B, T, C)             Mamba+CAM, then DF-FFN
      -> Predictor           (B, T)                the BVP waveform

Everything after PGA is a time series, which is the basis the design depends
on: RhythmMamba Section 4.5 measured that leaving spatial structure in the token
sequence makes Mamba *worse* (4.90 MAE at 8x8 tokens against 3.54 at 1x1), because
spatial information raises the dimensionality of the state transition without
adding anything the recurrence can use.

Heart rate is never predicted. It is read off the returned waveform by band-pass
and Welch PSD, exactly as both source papers do -- which is why a clip-level HR
label is not needed for training and its known faults (DATASETS.md: five UBFC
subjects with a broken HR column) cannot reach the loss.
"""

from __future__ import annotations

import torch
from torch import nn

from .block import ChannelAdaptiveMambaBlock
from .df_ffn import DualFrequencyFFN
from .fusion_stem import FusionStem
from .mamba_layer import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_D_STATE,
    DEFAULT_DIRECTION,
    DEFAULT_HEADDIM,
    DEFAULT_MIMO_RANK,
    DEFAULT_ROPE_FRACTION,
)
from .pga import PhysiologyGuidedAttention
from .vanilla_ffn import VanillaFFN

# Section 4.1 / 4.5: 128x128 at 30 Hz. 300 frames is a 10 s window.
#
# This is a deliberate departure from both papers, which use 160-frame segments,
# and from RhythmFormer's Table 11, which measures 160 as the optimum (3.07 MAE
# against 3.53 at 80 and 3.86 at 320). Numbers produced at 300 are therefore not
# directly comparable with the published ones. The published UBFC protocol has
# been removed with it: it only understands UBFC's subjectN ids, and on the
# pooled manifest `combine` writes it silently put 606 of 666 subjects in train.
#
# What 300 buys: the FFT bin spacing halves, from 11.25 bpm at 160 to 6.0 bpm at
# 300, so a rate read off a window is resolved twice as finely before any
# zero-padding interpolates it.
DEFAULT_N_FRAMES = 300
DEFAULT_FPS = 30.0

# Widths neither paper states. These are not guesses: they are the configuration
# that reproduces both published cost figures -- 0.91M parameters and 80.82M
# multiply-accumulates per frame -- to within 2% on each, once every module is
# shaped the way the referenced sources shape it. tests/test_budget.py is the gate and
# records what else fits.
DEFAULT_DIM = 80
DEFAULT_DEPTH = 4
# RhythmMamba builds its stem at dim//4; the cost budget puts CFMamba's at dim//5
# (16 channels at dim=80), which is where the stem stops dominating the MAC count.
DEFAULT_STEM_DIVISOR = 5
# "N > C denotes the expanded channel dimension" (Section 3.3).
DEFAULT_FFN_RATIO = 2
# CMamba's GDD-MLP expansion rate (its Appendix C.1), which CFMamba inherits
# without naming a value -- CMamba sweeps it in a rasterised figure and the only
# numeric "expansion rate" it prints, 1, fits to a different module (M-Mamba's
# linear, its A.2). 1.0 makes the hidden layer as wide as the stream, which is the
# smallest value that is really an *expansion* rather than a bottleneck and so
# matches CMamba's own word for it. It costs +5.0% against the published parameter
# count; see tests/test_budget.py for what that trade gains and gives up.
DEFAULT_CAM_EXPANSION = 1.0


class Predictor(nn.Module):
    """(B, T, C) -> (B, T). "a lightweight 1D convolutional-based prediction head".

    A convolution rather than a Linear, because the map from local temporal context
    to pulse amplitude is the same at every instant: it should share weights across
    time and stay valid for any T.
    """

    def __init__(self, dim: int, kernel: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(1)


class CFMambaPhys(nn.Module):
    """The full model. Ablation flags mirror CFMamba Table 5 and RhythmFormer T4."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        depth: int = DEFAULT_DEPTH,
        stem_dim: int | None = None,
        ffn_hidden: int | None = None,
        stem_variant: str = "rhythmmamba",
        stem_k1: int | None = None,
        stem_k2: int | None = None,
        fps: float = DEFAULT_FPS,
        n_frames: int = DEFAULT_N_FRAMES,
        pts_mode: str = "channel",
        pts_bins: int = 64,
        ffn_activation: str | None = "gelu",
        d_state: int = DEFAULT_D_STATE,
        headdim: int = DEFAULT_HEADDIM,
        expand: int = 2,
        mimo_rank: int = DEFAULT_MIMO_RANK,
        rope_fraction: float = DEFAULT_ROPE_FRACTION,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        paths: int = 3,
        direction: str = DEFAULT_DIRECTION,
        cam_expansion: float = DEFAULT_CAM_EXPANSION,
        cam_pooling: str = "cmamba",
        use_pga: bool = True,
        use_cam: bool = True,
        ffn: str = "df",
        fuse_stem: bool = True,
    ) -> None:
        super().__init__()
        if ffn not in ("df", "vanilla", "none"):
            raise ValueError(f"unknown ffn {ffn!r}, expected df, vanilla or none")
        self.dim = dim
        self.depth = depth
        self.fps = fps
        self.n_frames = n_frames
        self.stem_dim = stem_dim if stem_dim is not None else max(1, dim // DEFAULT_STEM_DIVISOR)
        self.ffn_hidden = ffn_hidden if ffn_hidden is not None else dim * DEFAULT_FFN_RATIO
        self.ffn_kind = ffn

        self.stem = FusionStem(
            dim=dim, stem_dim=self.stem_dim, variant=stem_variant, fuse=fuse_stem,
            k1=stem_k1, k2=stem_k2,
        )
        # use_pga=False is the Table 5 ablation: a plain spatial mean, which is
        # what RhythmMamba's frame stem does before its own attention. It weights
        # the background exactly as heavily as the cheeks.
        self.pga = PhysiologyGuidedAttention() if use_pga else None

        def make_ffn() -> nn.Module:
            if ffn == "df":
                return DualFrequencyFFN(
                    dim, self.ffn_hidden, fps=fps, pts_mode=pts_mode,
                    n_frames=n_frames, pts_bins=pts_bins,
                    activation=ffn_activation,
                )
            if ffn == "vanilla":
                return VanillaFFN(dim, self.ffn_hidden)
            return nn.Identity()

        self.blocks = nn.ModuleList(
            ChannelAdaptiveMambaBlock(
                dim, make_ffn(), use_cam=use_cam, cam_expansion=cam_expansion,
                cam_pooling=cam_pooling, d_state=d_state, headdim=headdim,
                expand=expand, mimo_rank=mimo_rank, rope_fraction=rope_fraction,
                chunk_size=chunk_size, paths=paths, direction=direction,
            )
            for _ in range(depth)
        )
        self.predictor = Predictor(dim)

    def forward(self, frames: torch.Tensor, skin: torch.Tensor | None = None) -> torch.Tensor:
        """frames (B, T, 3, H, W) in [0, 1], skin (B, H, W) in [0, 1] -> (B, T).

        `skin` may be omitted, in which case PGA falls back to an all-ones mask and
        its Gaussian prior degrades to a centred bias. That is a real degradation,
        not a neutral default, so the caller passes the cached SegFace mask.
        """
        features = self.stem(frames)
        if self.pga is None:
            sequence = features.mean(dim=(3, 4))
        else:
            if skin is None:
                skin = frames.new_ones(frames.shape[0], *frames.shape[-2:])
            sequence = self.pga(features, skin)
        for block in self.blocks:
            sequence = block(sequence)
        return self.predictor(sequence)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        """Per-component parameter counts, for reading against the 0.91M target."""
        groups = {
            "stem": self.stem,
            "pga": self.pga,
            "blocks": self.blocks,
            "predictor": self.predictor,
        }
        header = (
            f"CFMambaPhys  dim={self.dim} depth={self.depth} "
            f"stem_dim={self.stem_dim} ffn_hidden={self.ffn_hidden} ffn={self.ffn_kind}"
        )
        lines = [header]
        for name, module in groups.items():
            count = 0 if module is None else sum(p.numel() for p in module.parameters())
            lines.append(f"  {name:10} {count / 1e6:7.4f} M")
        lines.append(f"  {'total':10} {self.parameter_count() / 1e6:7.4f} M")
        return "\n".join(lines)
