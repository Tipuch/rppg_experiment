"""The DFT as a constant matrix, for the runtimes that have no FFT.

LiteRT has no complex tensor type and no FFT: `aten._fft_c2c.default` has no
lowering, so `cs_ffn.py` and `pts_ffn.py` cannot be exported as written.

They do not need to be. Both transform over a *fixed* axis -- 160 channels in
CS-FFN, 300 frames in PTS-FFN -- and a DFT of fixed length is a constant linear
operator. So each `fft`/`ifft` pair becomes two real matmuls against precomputed
cosine and sine matrices, which every runtime has.

Three properties make this a good trade rather than a grudging one:

- **It is cheap.** Everything after PGA is a (300, 80) time series, so the O(N^2)
  transform costs about 353 M MACs per clip against the model's 23.7 G -- under
  2%. The O(N log N) it replaces was never the expensive part.
- **The inverse needs only two matmuls, not four.** Both stages discard the
  imaginary half of the inverse transform (`.real`), so the imaginary output is
  never computed.
- **The weights are untouched.** `ComplexLinear` already carries the real and
  imaginary parts as two real tensors and already multiplies them out by hand, so
  nothing here changes what is learned. The matrices are constants, registered
  non-persistently so a trained checkpoint still loads `strict=True`.

What it costs: the transform length is baked in. `cs_ffn.py` was already fixed in
its axis, but PTS-FFN becomes locked to one clip length, which is the property
`pts_ffn.py`'s docstring uses to argue *against* `mode="full"`. For an
inference-only export at T=300 that costs nothing; for training it would.

Sign conventions follow `torch.fft`, so the two paths are interchangeable:

    analysis     X_k = sum_n x_n (cos(2 pi k n / N) - i sin(2 pi k n / N))
    synthesis    x_n = (1 / N) sum_k X_k (cos(2 pi k n / N) + i sin(2 pi k n / N))
"""

from __future__ import annotations

import math

import torch
from torch import nn

# The two transform implementations, named in one place: `model.py` validates the
# flag, `cs_ffn.py` and `pts_ffn.py` each accept it.
SPECTRAL = ("fft", "matmul")


class FixedDFT(nn.Module):
    """A length-`n` DFT and its inverse, as two constant matrices.

    Both matrices are symmetric -- cos(2 pi k n / N) does not care which index is
    which -- so the same buffer serves the transform and its inverse and no
    transpose is needed anywhere.

    Every method operates on the **last** axis. Callers transforming another axis
    move it last first; a transpose is free next to a matmul of this size, and it
    keeps the op list to `transpose` and `matmul`.
    """

    def __init__(self, length: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        self.length = length
        index = torch.arange(length, dtype=torch.float64)
        # float64 for the angle, then cast: at n=300 the largest argument is
        # 2 pi * 299 * 299 / 300, and building it in float32 loses bits in the
        # product before the cosine ever sees it.
        angle = 2.0 * math.pi * index[:, None] * index[None, :] / length
        self.register_buffer("cos", torch.cos(angle).to(dtype), persistent=False)
        self.register_buffer("sin", torch.sin(angle).to(dtype), persistent=False)

    def extra_repr(self) -> str:
        return f"length={self.length}"

    def analyse(
        self, real: torch.Tensor, imag: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`torch.fft.fft` over the last axis, carried as two real tensors.

        `imag=None` is the real-input case, which halves the work: two matmuls
        instead of four.
        """
        cos, sin = self.cos.to(real.dtype), self.sin.to(real.dtype)
        if imag is None:
            return real @ cos, -(real @ sin)
        return real @ cos + imag @ sin, imag @ cos - real @ sin

    def synthesise_real(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """The real part of `torch.fft.ifft` over the last axis.

        Only the real part, because both callers take `.real` -- a complex linear
        does not preserve conjugate symmetry, and the imaginary leftover is the
        part of the learned map no real signal can express.
        """
        cos, sin = self.cos.to(real.dtype), self.sin.to(real.dtype)
        return (real @ cos - imag @ sin) / self.length


def share_fixed_dfts(root: nn.Module) -> int:
    """Collapse every duplicate `FixedDFT` under `root` onto one instance per length.

    Each block builds its own, so at depth 4 the model holds four identical
    (160, 160) pairs and four identical (300, 300) pairs: 3.70 MB of tables
    against 3.73 MB of parameters, where 0.92 MB is distinct.

    This is a host-memory saving only. The exported `.tflite` is *byte-identical*
    either way -- measured -- because the LiteRT converter deduplicates equal
    constants itself. Nothing here reaches the device.

    Sharing is safe because the tables are constants: nothing writes to them, and
    they are registered non-persistently, so no `state_dict` key moves. Returns
    how many were collapsed.
    """
    canonical: dict[tuple[int, torch.dtype], FixedDFT] = {}
    duplicates = []
    for module in root.modules():
        dft = getattr(module, "dft", None)
        if isinstance(dft, FixedDFT):
            first = canonical.setdefault((dft.length, dft.cos.dtype), dft)
            if first is not dft:
                duplicates.append((module, first))
    for module, first in duplicates:
        module.dft = first
    return len(duplicates)
