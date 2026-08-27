"""Fusion Stem: raw frames fused with frame differences.

A face is roughly 127 LSB of static appearance and the pulse is a 0.1-0.5 LSB
change on top of it, so raw frames ask the network to find a 0.3% modulation
under a signal 400x larger. Frame differences cancel the static term outright but
amplify every other artefact with it. The stem uses both: differences to expose
the change, raw frames to say where the skin is.

RhythmFormer measures the effect directly. Dropping it in to models that never had
it moves PhysFormer 11.99 -> 8.72 MAE on MMPD and 16.17 -> 2.55 on COHFACE
(Tables 5 and 2), which is a larger swing than most architectures produce.

CFMamba cites this stem to RhythmMamba (Section 3.2) but the two source papers
describe different geometry, so both are built here and selected by `variant`.
Three values RhythmFormer ablates are not free parameters and are fixed
accordingly: a 5:5 raw/difference ratio (its Table 8, 3.07 MAE against 4.56 for
raw alone and 4.39 for differences alone), +/-2 adjacent frames (Table 6, 3.07
against 4.19 at +/-1 and 3.74 at +/-3), and a frame step of 1 (Table 7).
"""

from __future__ import annotations

import torch
from torch import nn

# RhythmMamba 3.2 / RhythmFormer 3.2 disagree; the parameter budget decides.
VARIANTS = {
    # kernel1, kernel2, pool in stem2, output stride relative to the input.
    #
    # rhythmmamba's k2 is 5 rather than the 7 its prose implies. The prose only
    # constrains stem2 to stride 1 -- the 16x16 arithmetic does not work otherwise
    # -- so its kernel was never pinned, and the two source papers disagree anyway
    # (7 against RhythmFormer's 3). 5 is what reproduces the published cost: it
    # lands the model at 82.5M MACs per frame against a stated 80.82M, where 7
    # gives 94.5M and 3 gives 74.1M. See tests/test_budget.py.
    "rhythmmamba": {"k1": 7, "k2": 5, "pool2": True, "stride": 8},
    "rhythmformer": {"k1": 5, "k2": 3, "pool2": False, "stride": 4},
}


def temporal_differences(x: torch.Tensor, reverse: bool = True) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T, 4C, H, W): differences across five frames.

    The five-frame stack X[t-2..t+2] is built by clamping at the clip boundary
    rather than wrapping. Wrapping would join the last frame of a recording to its
    first and manufacture a discontinuity at exactly the frequencies rPPG reads.

    `reverse` selects RhythmMamba's stated ordering over RhythmFormer's. The two
    differ only by a sign on all four channels, and the first convolution is linear
    and unbiased with respect to sign, so this cannot change what the stem is able
    to represent -- it is kept faithful, not because it matters.
    """
    n_frames = x.shape[1]
    index = torch.arange(n_frames, device=x.device)
    shifted = [x.index_select(1, (index + k).clamp(0, n_frames - 1)) for k in (-2, -1, 0, 1, 2)]
    forward = [shifted[i + 1] - shifted[i] for i in range(4)]
    if reverse:
        forward = [-d for d in forward]
    return torch.cat(forward, dim=2)


class FusionStem(nn.Module):
    """(B, T, 3, H, W) -> (B, T, dim, H/stride, W/stride)."""

    def __init__(
        self,
        dim: int,
        stem_dim: int,
        variant: str = "rhythmmamba",
        alpha: float = 0.5,
        beta: float = 0.5,
        fuse: bool = True,
        k1: int | None = None,
        k2: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {list(VARIANTS)}")
        spec = dict(VARIANTS[variant])
        # RhythmMamba's text gives 7x7 for both convolutions but only constrains
        # stem2 to stride 1, so its kernel is genuinely free; RhythmFormer says
        # 5x5 then 3x3. These overrides exist so the published cost budget can
        # arbitrate rather than the prose. See tests/test_budget.py.
        if k1 is not None:
            spec["k1"] = k1
        if k2 is not None:
            spec["k2"] = k2
        self.k1, self.k2 = spec["k1"], spec["k2"]
        self.variant = variant
        self.stride = spec["stride"]
        self.alpha = alpha
        self.beta = beta
        # fuse=False is the "vanilla stem" ablation: raw frames only, no difference
        # branch. RhythmFormer's Table 4 puts that at 3.56 MAE against 3.07.
        self.fuse = fuse
        self.reverse_diff = variant == "rhythmmamba"

        def block(in_ch: int, kernel: int, stride: int, pool: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, stem_dim, kernel, stride=stride,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm2d(stem_dim),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(3, stride=2, padding=1))
            return nn.Sequential(*layers)

        self.stem_raw = block(3, spec["k1"], 2, pool=True)
        self.stem_diff = block(12, spec["k1"], 2, pool=True) if fuse else None
        self.stem2 = block(stem_dim, spec["k2"], 1, pool=spec["pool2"])

        # "a 3D convolution to capture local spatiotemporal correlations"
        # (CFMamba 3.1). The (2, 5, 5) kernel consumes one frame, so the time axis
        # is padded by one at the front: letting T become T-1 here would shift
        # every frame against its PPG target and nothing downstream would notice.
        self.stem3 = nn.Sequential(
            nn.Conv3d(stem_dim, dim, (2, 5, 5), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_frames = x.shape[:2]
        raw = self.stem_raw(x.flatten(0, 1))

        if self.stem_diff is None:
            fused = self.stem2(raw)
        else:
            diff = self.stem_diff(
                temporal_differences(x, reverse=self.reverse_diff).flatten(0, 1)
            )
            # X_fusion = Stem2(alpha*X_raw + beta*X_diff) + Stem2(X_diff). The
            # second term is not redundant: it gives the difference branch a path
            # that the raw frames cannot dilute.
            fused = self.stem2(self.alpha * raw + self.beta * diff) + self.stem2(diff)

        spatial = fused.shape[-2:]
        volume = fused.view(batch, n_frames, -1, *spatial).permute(0, 2, 1, 3, 4)
        volume = nn.functional.pad(volume, (0, 0, 0, 0, 1, 0), mode="replicate")
        out = self.stem3(volume)
        return out.permute(0, 2, 1, 3, 4)
