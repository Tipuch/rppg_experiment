"""CFMamba-Phys: a frequency-aware state space model with channel enhancement.

Wang et al., Biomedical Signal Processing and Control 126 (2026) 110996,
doi:10.1016/j.bspc.2026.110996 -- rebuilt from the paper, with RhythmMamba
(arXiv:2404.06483) supplying what it omits and RhythmFormer (arXiv:2402.12788)
pinning values that both leave open.

One file per module the papers name, one test file per module, so a claim in a
paper can be tracked to the code that implements it and to the test that keeps it
in place:

| file               | paper                          | test                    |
|--------------------|--------------------------------|-------------------------|
| fusion_stem.py     | CFMamba 3.1 / RM 3.2 / RF 3.2  | test_fusion_stem.py     |
| pga.py             | CFMamba Eqs. 1-5               | test_pga.py             |
| mamba_layer.py     | RM 3.3 Eq. 4 / Mamba-3 3.1-3.4 | test_mamba_layer.py     |
| cam.py             | CFMamba Eqs. 6-8               | test_cam.py             |
| complex_linear.py  | CFMamba Eqs. 10-11             | test_complex_linear.py  |
| cs_ffn.py          | CFMamba Eqs. 9-12              | test_cs_ffn.py          |
| band_mask.py       | CFMamba Eqs. 14-15             | test_band_mask.py       |
| pts_ffn.py         | CFMamba Eqs. 13-18             | test_pts_ffn.py         |
| df_ffn.py          | CFMamba 3.3 (the two stages)   | test_df_ffn.py          |
| block.py           | CFMamba 3.2-3.3 / RM Fig. 3    | test_block.py           |
| vanilla_ffn.py     | CFMamba Table 5 ablation       | test_vanilla_ffn.py     |
| model.py           | CFMamba Fig. 2                 | test_cfmamba_model.py   |
|                    | budget: 0.91M par, 80.82M FLOP | test_budget.py          |

Every value the papers do not state is a named constructor argument with its
justification beside it.

**The scan is Mamba-3.** All three papers call `mamba_ssm.Mamba`, the Mamba-1
selective scan. `mamba_layer.py` swaps the recurrence for Mamba-3 (Lahoti et al.,
arXiv:2603.15569) and leaves the block around it alone; see that module for what
changes and why.

`mamba_layer.py` is the only module that needs CUDA: Mamba-3's scan is a Triton
kernel with no CPU path. Everything else is plain PyTorch and is tested on CPU,
which is why the package is split at that seam.
"""

from .band_mask import GaussianBandMask
from .block import ChannelAdaptiveMambaBlock
from .cam import ChannelAdaptiveModulation
from .complex_linear import ComplexLinear
from .cs_ffn import ChannelSpectralFFN
from .df_ffn import DualFrequencyFFN
from .fusion_stem import FusionStem
from .model import CFMambaPhys, Predictor
from .pga import PhysiologyGuidedAttention
from .pts_ffn import PhysiologyTemporalSpectralFFN
from .vanilla_ffn import VanillaFFN

__all__ = [
    "CFMambaPhys",
    "ChannelAdaptiveMambaBlock",
    "ChannelAdaptiveModulation",
    "ChannelSpectralFFN",
    "ComplexLinear",
    "DualFrequencyFFN",
    "FusionStem",
    "GaussianBandMask",
    "PhysiologyGuidedAttention",
    "PhysiologyTemporalSpectralFFN",
    "Predictor",
    "VanillaFFN",
]
