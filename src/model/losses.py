"""The composite waveform loss. CFMamba Eq. 19-21.

    L = alpha * L_time + beta * L_freq

`L_time` is the negative Pearson correlation, which is scale- and offset-invariant
-- required here, because the network sees skin brightness and cannot know the
contact sensor's gain, so demanding absolute agreement would penalise a perfectly
shaped prediction for being the wrong size.

`L_freq` is a cross-entropy between the predicted spectrum and the ground truth's
dominant frequency. It exists because correlation alone is indifferent to *which*
periodicity it locked onto: a prediction at twice the true rate can correlate
respectably while being wrong by 70 bpm.

**Weights.** CFMamba leaves alpha and beta unstated. RhythmFormer Section 3.4 uses
0.2 and 1.0, and its Table 13 shows the balance is not minor -- the temporal
term alone reaches 3.56 MAE, the frequency term alone reduces to 13.32, and
together they reach 3.13. Those are the defaults.

**Why the frequency term is written here rather than imported.** The vendored
`TorchLossComputer.Frequency_loss` is the same construction, but it obtains its
ground-truth rate through `calculate_metric_per_video`, which hardcodes a
first-order 0.6-3.3 Hz band-pass. Importing it would put a 36-198 bpm band inside
the *loss*, where it is harder to notice than in a metric. The candidate range
below is CFMamba's own 45-150 bpm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .waveform import neg_pearson

# CFMamba Section 4.2: "The frequency range [0.75, 2.5] Hz covers the standard
# physiological range of human heart rates (45-150 bpm)." One candidate per bpm,
# so the cross-entropy is over 105 classes at 1 bpm resolution -- finer than the
# 11.25 bpm an unpadded 160-frame DFT would give.
BPM_MIN, BPM_MAX = 45, 150

# CFMamba Eq. 19 states alpha and beta as symbols and gives no values.
# RhythmFormer Section 3.4 Table 13 supplied 0.2 and 1.0, and this project ran that
# way until REPORT_cfmamba.md finding 2: the temporal term went flat from epoch 2,
# and at alpha=0.2 it was ~1.5% of the final loss -- the optimiser had almost no
# reason to fix the waveform the model exists to predict. beta stays at 1.0.
DEFAULT_ALPHA = 0.8
DEFAULT_BETA = 1.0


def bpm_candidates(device: torch.device | None = None) -> torch.Tensor:
    return torch.arange(BPM_MIN, BPM_MAX, dtype=torch.float32, device=device)


def spectral_power(signal: torch.Tensor, fps: float, bpm: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, K): differentiable power at each candidate rate in `bpm`.

    A DFT evaluated only at the frequencies of interest, rather than an FFT
    followed by binning. Two reasons: the candidates land exactly on whole bpm
    instead of on whatever grid T happens to impose, and every step stays
    differentiable, which an argmax over FFT bins would not be.
    """
    signal = signal - signal.mean(dim=1, keepdim=True)
    n_frames = signal.shape[1]
    time = torch.arange(n_frames, device=signal.device, dtype=signal.dtype) / fps
    phase = 2 * math.pi * (bpm.to(signal) / 60.0).unsqueeze(1) * time.unsqueeze(0)
    real = signal @ torch.cos(phase).transpose(0, 1)
    imag = signal @ torch.sin(phase).transpose(0, 1)
    return real.square() + imag.square()


def frequency_loss(
    predicted: torch.Tensor, target: torch.Tensor, fps: float = 30.0
) -> torch.Tensor:
    """CFMamba Eq. 21: CE(PSD(S_pre), argmax(PSD(S_gt))).

    The label is the target waveform's own dominant rate, taken under no_grad --
    it is a label, not a quantity to optimise. Reading it from the contact PPG
    rather than from a manifest column is intentional: five UBFC subjects have a
    broken HR readout and an intact waveform (DATASETS.md).
    """
    bpm = bpm_candidates(predicted.device)
    with torch.no_grad():
        label = spectral_power(target.float(), fps, bpm).argmax(dim=1)
    return F.cross_entropy(spectral_power(predicted.float(), fps, bpm), label)


def composite_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    fps: float = 30.0,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Eq. 19. Returns the loss and its two terms, for logging them separately.

    Logged separately because they fail differently: a temporal term that stalls
    near 1.0 means the waveform is uncorrelated, while a frequency term that
    stalls means the model found *a* periodicity and picked the wrong one. One
    summed number cannot distinguish those.

    The terms come back as **detached device tensors, not floats**. Calling
    `float()` on them here would be three device-to-host syncs per step, sitting
    between the forward pass and `.backward()` and stalling the pipeline at the one
    point where the GPU should be running ahead of the loader. The caller
    accumulates them on the device and reads the total once a logging window, which
    is one sync every `log_every` steps instead of three every step.
    """
    time_term = neg_pearson(predicted.float(), target.float())
    freq_term = frequency_loss(predicted, target, fps)
    total = alpha * time_term + beta * freq_term
    return total, {
        "loss": total.detach(),
        "time": time_term.detach(),
        "freq": freq_term.detach(),
    }
