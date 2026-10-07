"""Respiratory rate from the pulse-interval series of a 30 s PPG: run AH's network,
frozen (trained 2026-10-07 on `rr_table`'s table, kept epoch 28).

    (B, 2, 900)   tach, ls from waves_pp.parquet (`INPUTS`), 30 Hz
    -> per signal, its own ConvDownStem       (B, 16, 225) each
         signal, 1st and 2nd derivative -> Conv1d(3 -> 16, k 5, stride 2) -> GELU
                                        -> Conv1d(16 -> 16, k 5, stride 2) -> GELU
    -> stacked on a signal axis               (B, 16, 2, 225)
    -> Conv2d(16 -> 16, kernel (2, 3), padding (0, 1)) -> GELU
    -> mean over the signal axis              (B, 225, 16) = s
    -> m = Mamba-3(LayerNorm(s))              (B, 225, 16), SISO, 2 heads of 16
    -> cat(mean_T(s), mean_T(m))              (B, 32)
    -> Dropout(0.5) -> GELU -> Linear(32 -> 1), bias starting at the natural train mean
    -> (B,) breaths/min

**Inputs.** Breathing reaches the PPG through the beat-to-beat interval, which
shortens on the in-breath. `tach` is that interval series on the 30 Hz grid and `ls`
its Lomb-Scargle power (src/vitals/waves.py); each is z-scored per row, so the
network sees the breathing in the pulse timing and not the heart rate or the
dataset. The wave itself is not an input: earlier runs fit it through dataset and
heart-rate shortcuts that did not carry over to unseen sources. A row with too few
beats holds zeros in both, so it gets one constant estimate.

**The stem** (`Multi2dStem`). TimeMixer++ (arXiv 2410.16032) builds each coarser
scale with a stride-2 convolution; `ConvDownStem` does that twice per signal, 30 ->
15 -> 7.5 Hz, whose Nyquist (3.75 Hz) still holds the cardiac band. Each stage's 5
learned taps are its only anti-aliasing filter. The derivative channels
(`derivative_channels`) are central differences, each divided by its own
per-segment std so the three share a scale. The Conv2d spans both signals unpadded
(kernel height 2), so tach and ls keep their own weights and the signal axis
collapses to 1; its 3 time taps are zero-padded to keep 225 steps. On `ls` the
steps are frequency bins read in order, not time.

**One direct Mamba-3 scan.** The rPPG model scans the clip whole, in halves and in
quarters; at 6-8 breaths/min a quarter of a 30 s segment holds at most one breath,
so here: one scan, forward only, stock init (dt log-uniform over [0.001, 0.1]),
d_state 16, expand 2, chunk 64. Head size 16 (`HEADDIM`): the CUDA Triton kernel
needs at least 16. The LayerNorm hands the scan a fixed scale.

**Concatenated pooling.** The stem's features and the scan's output are each
averaged over T and concatenated, not summed: in a sum the scan's output was 35-72x
the stem's (run M), so the stem's features barely counted. The mean is the right
prior for a rate, a property of the whole segment. Run AC pooled the sum
(`pool_concat` False); `load` still reads its checkpoint.

**Dropout** in the head only (`HEAD_DROPOUT`, 0.5), before the output layer. Runs
H-J overfit from epoch 2 without it; dropout on the scan's input was tried and
turned off (run Q onward).

**The last bias starts at the natural train mean**, so the first steps go into shape
rather than into finding the target's level. Natural, not the oversampled table's
mean, which `balance` moves on purpose.

**`scan`** picks the recurrence, as in src/model/cfmamba/mamba_layer.py:

    "kernel"    `mamba_ssm.Mamba3`, the Triton kernel. Training runs on it.
    "export"    `mamba3_export.Mamba3Sequential`, the same recurrence in plain
                PyTorch. Same state_dict keys, runs on CPU, about 80x slower.

`mamba_ssm` is imported only for "kernel", so CPU runs never need Triton. `main`
picks "kernel" on a GPU and "export" on the CPU.

## Training (`main`)

    <table dir>/features.parquet   the balanced RR table (`rr_table`), labels, splits
    <table dir>/waves_pp.parquet   each key's wave, tach and ls
      -> inner join on `KEYS`, null labels dropped    (`join`, drops printed)
      -> any non-finite input refused
      -> 64 waves' skewness checked against the table's     (`check_waves`)
      -> train on "train", pick the epoch on "dev", score train, dev and test
      -> <out>/ (default <table dir>/rr_mamba/)
           rr_mamba_rr_bpm.pt                 the kept epoch
           checkpoints/epoch_NNN.pt           every epoch
           rr_mamba_rr_bpm_history.parquet    one row per epoch
           rr_mamba_metrics.parquet           `score`
           rr_mamba_test_predictions.parquet  test estimates
           rr_mamba_rr_bpm_loss.png           `plot.rr_mamba_loss`, written last

An output dir whose checkpoints/ already holds epoch files is refused before any
data is read: a new run would mix its epochs with the old ones. They are never
deleted here.

**Rows.** The join is many-to-one, so the oversampled train copies keep their
repeats and the model sees the balanced label spread. `fit` is handed only the train
and dev rows, so test rows are never seen until the checkpoint is chosen. Train
scores use only the "natural" rows, once per key (`natural_rows`), so they compare
with dev. All rows fit in memory (145k x 2 x 900 float32 is about 1 GB), so batches
are slices of tensors already on the device, not a DataLoader.

**Stale waves.** The table and the waves file are written apart, so waves of other
keys, flipped or shifted, would still join and train quietly on the wrong input.
`check_waves` recomputes `skewness` on 64 seeded keys and stops if any differs from
the table's by more than 1e-4.

**Loss and optimiser.** Huber with delta 3 /min (`HUBER_DELTA`): squared below 3, so
the tail misses weigh as in RMSE, linear above, so a few wild labels can't steer the
fit. AdamW with weight decay on the matrices only, as src/model/train.py: 1-D tensors
(biases, norm gains, Mamba-3's `dt_bias` and `D`), names ending in "bias" and tensors
marked `_no_weight_decay` get none, since decaying them shifts or rescales a layer
rather than regularising it. Linear warmup over the first 5% of steps, then cosine to
1% of the peak. Gradients clipped to 1. fp32 throughout.

**The kept epoch** is the one with the highest pooled dev slope (`fit`): the slope of
the estimate on the label over all dev rows, 1 for no pull to the mean, 0 for a
constant. The goal is low error and no pull to the mean; with narrow labels the
lowest-RMSE epoch was the one closest to a constant. Every epoch's weights are saved
too, each with its history row, so any epoch can be reloaded and scored.

**Repeatability.** The seed fixes the initial weights and the batch order, but the
Mamba-3 kernel's backward pass is not bitwise repeatable, so two runs with one seed
drift apart slowly. The "export" scan on the CPU repeats exactly.

**Reports.** No per-epoch messages: the history prints as one table at the end,
then the scores, and `plot.rr_mamba_loss` draws train and dev MAE per epoch with the
kept epoch marked, and dev slope and spread.

**Scores** (`score`): per split, over all sources together (source "all") and then
per source. MAE, RMSE, r, slope and spread, the MAE of the lowest and highest 10%
of labels (`TAIL`), and the MAE of predicting the natural train mean.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from src.model.cfmamba.mamba_layer import NO_DECAY_PARAMETERS, SCANS
from src.vitals import plot
from src.vitals.rr_table import NATURAL_MEAN, OUT_DIR, TARGET, skewness
from src.vitals.waves import KEYS, PP_FILE, WAVE_FPS

INPUTS = ("tach", "ls")
DIM = 16
DOWN_KERNEL = 5
DOWN_STAGES = 2
CONV2D_KERNEL = 3
# Added to each derivative channel's per-segment std: a flat segment gives zeros.
DERIVATIVE_EPS = 1e-6
HEAD_DROPOUT = 0.5
# The Mamba-3 settings, saved in every checkpoint's config and checked by `load`.
DESIGN = {
    "dim": DIM, "d_state": 16, "expand": 2, "headdim": 16, "rope_fraction": 1.0,
    "chunk_size": 64, "pooling": "mean", "scan_block": True, "head_widths": (),
}  # fmt: skip
SCAN_SETTINGS = ("d_state", "expand", "headdim", "rope_fraction", "chunk_size")

NAME = "rr_mamba"
TABLE = OUT_DIR / "features.parquet"
HUBER_DELTA = 3.0
LR = 2e-5
EPOCHS = 50
BATCH = 32
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05
LR_FLOOR = 0.01
GRAD_CLIP = 1.0
SEED = 20261003
# Inference only: no activations kept for a backward pass, so larger batches fit.
EVAL_BATCH = 256
# The stale-wave check: keys sampled, and how far a recomputed skewness may sit from
# the table's. Matching waves agree to about 1e-6; a flipped wave is off by units.
CHECK_ROWS = 64
CHECK_TOLERANCE = 1e-4
# Share of a split's labels at each end that `score` scores apart.
TAIL = 0.1
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}
loss_fn = nn.HuberLoss(delta=HUBER_DELTA)


def derivative_channels(x: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, 3, T): the signal, its 1st and its 2nd derivative (central
    differences, applied twice), each derivative divided by its own std."""
    d1 = torch.gradient(x, dim=-1)[0]
    d2 = torch.gradient(d1, dim=-1)[0]
    scaled = lambda d: d / (d.std(-1, keepdim=True) + DERIVATIVE_EPS)
    return torch.stack([x, scaled(d1), scaled(d2)], dim=1)


class ConvDownStem(nn.Module):
    """(B, T) -> (B, DIM, T / 4): the derivative channels through 2 stride-2
    convolutions, each followed by GELU. Padding kernel // 2 gives ceil(L / 2) steps."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(
                3 if i == 0 else DIM,
                DIM,
                DOWN_KERNEL,
                stride=2,
                padding=DOWN_KERNEL // 2,
            )
            for i in range(DOWN_STAGES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = derivative_channels(x)
        for conv in self.convs:
            h = nn.functional.gelu(conv(h))
        return h


class Multi2dStem(nn.Module):
    """(B, S, T) -> (B, T / 4, DIM): a ConvDownStem per signal, stacked on a signal
    axis, one Conv2d over all signals and 3 time taps, GELU, mean over signals."""

    def __init__(self, signals: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(ConvDownStem() for _ in range(signals))
        self.mix = nn.Conv2d(
            DIM, DIM, (signals, CONV2D_KERNEL), padding=(0, CONV2D_KERNEL // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.branches):
            raise ValueError(f"expected {len(self.branches)} signals, got {x.shape[1]}")
        image = torch.stack([b(x[:, i]) for i, b in enumerate(self.branches)], dim=2)
        return nn.functional.gelu(self.mix(image)).mean(2).transpose(1, 2)


class RRMamba(nn.Module):
    """(B, 2, T) z-scored `INPUTS` -> (B,) respiratory rate, breaths/min (module
    docstring). `pool_concat` False is run AC's pooling, mean_T(s + m)."""

    def __init__(
        self, init_mean: float, scan: str = "kernel", pool_concat: bool = True
    ) -> None:
        super().__init__()
        if scan not in SCANS:
            raise ValueError(f"unknown scan {scan!r}, expected one of {SCANS}")
        # Imported here: mamba_ssm pulls in Triton and needs a GPU.
        if scan == "kernel":
            from mamba_ssm import Mamba3 as Scan
        else:
            from src.model.cfmamba.mamba3_export import Mamba3Sequential as Scan
        self.pool_concat = pool_concat
        # Saved with each checkpoint; `load` rebuilds the network from it.
        self.config = DESIGN | {
            "init_mean": float(init_mean),
            "scan": scan,
            "pool_concat": pool_concat,
        }
        self.stem = Multi2dStem(len(INPUTS))
        self.mamba = Scan(
            d_model=DIM, is_mimo=False, **{k: DESIGN[k] for k in SCAN_SETTINGS}
        )
        for name in NO_DECAY_PARAMETERS:
            getattr(self.mamba, name)._no_weight_decay = True
        self.scan_norm = nn.LayerNorm(DIM)
        self.head_dropout = nn.Dropout(HEAD_DROPOUT)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(DIM * (2 if pool_concat else 1), 1)
        )
        nn.init.constant_(self.head[-1].bias, init_mean)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2 DIM): cat(mean_T(s), mean_T(m)); (B, DIM) mean_T(s + m) for run AC."""
        s = self.stem(x)
        m = self.mamba(self.scan_norm(s))
        return (
            torch.cat([s.mean(1), m.mean(1)], -1)
            if self.pool_concat
            else (s + m).mean(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_dropout(self.pooled(x))).squeeze(-1)


def load(path: Path, scan: str | None = None) -> tuple[RRMamba, dict]:
    """The network a checkpoint holds, on the CPU in eval mode, and the checkpoint.

    Any checkpoint whose config names other `DESIGN` settings is refused: the code of
    the older designs is gone. `scan` overrides the saved scan: "export" scores a
    kernel-trained checkpoint without a GPU, since the two share state_dict keys.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["config"]["model"]
    if any(saved.get(k) != v for k, v in DESIGN.items()):
        raise ValueError(f"{path} was trained with another design than run AH's")
    model = RRMamba(
        saved["init_mean"], scan or saved["scan"], saved.get("pool_concat", False)
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


_in_split = lambda *splits: pl.col("split").is_in(splits)
# The pull to the mean, in float64: slope of the estimate on the label, and
# sd(estimate) / sd(label). 1 for none.
_slope = lambda e, y: (
    ((e - e.mean()) * (y - y.mean())).sum() / ((y - y.mean()) ** 2).sum()
).item()
_spread = lambda e, y: (e.std() / y.std()).item()
_mae_rmse = lambda e, y: (
    (e - y).abs().mean().item(),
    (e - y).pow(2).mean().sqrt().item(),
)
# Inputs (N, 2, T), the `INPUTS` columns stacked in order; labels.
_tensors = lambda rows, device: (
    torch.from_numpy(np.stack([rows[c].to_numpy() for c in INPUTS], axis=1)).to(device),
    torch.from_numpy(rows[TARGET].to_numpy(writable=True)).float().to(device),
)  # fmt: skip


def _refuse_old_epochs(checkpoint_dir: Path) -> None:
    """Raise if `checkpoint_dir` holds epoch files: a new run would mix its epochs
    with an earlier run's. Never deletes them; that is the user's call."""
    if old := sorted(checkpoint_dir.glob("epoch_*.pt")):
        raise FileExistsError(
            f"{checkpoint_dir} already holds {len(old)} epoch checkpoints from an "
            f"earlier run ({old[0].name} to {old[-1].name}), which a new run would "
            "mix with its own. Choose a new --out, or remove the old files first."
        )


def join(table: pl.DataFrame, waves: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`table`'s labelled rows that have a wave and both `INPUTS`, with those columns
    added, and per split and source the rows dropped for no label or no wave. Prints
    the latter. Many-to-one on `KEYS`: each oversampled copy gets its key's wave."""
    columns = ["wave", *INPUTS]
    has_label = pl.col(TARGET).is_not_null()
    has_wave = pl.all_horizontal(pl.col(c).is_not_null() for c in columns)
    joined = table.join(
        waves.select(*KEYS, *columns), on=KEYS, how="left", validate="m:1"
    )
    report = (
        joined.group_by("split", "source")
        .agg(rows=pl.len(), no_label=(~has_label).sum(), no_wave=(has_label & ~has_wave).sum(),
             kept=(has_label & has_wave).sum())
        .sort(pl.col("split").replace_strict(SPLIT_ORDER, return_dtype=pl.Int8), "source")
    )  # fmt: skip
    with pl.Config(tbl_rows=-1):
        print(f"{TARGET} rows per split and source, dropped for no label or no wave:")
        print(report)
    return joined.filter(has_label, has_wave), report


def natural_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per natural key of `rows`: the shifted rows and the oversampled
    repeats dropped. The rows train scores are taken on."""
    return rows.filter(pl.col("augment") == "natural").unique(KEYS, maintain_order=True)


def check_waves(
    rows: pl.DataFrame, sample: int = CHECK_ROWS, tolerance: float = CHECK_TOLERANCE
) -> None:
    """Raise if `rows`' waves (`join` output) are not the ones the table's `skewness`
    was computed from: `sample` keys, drawn with `SEED`, are recomputed."""
    first = rows.select(KEYS).with_row_index().unique(KEYS, keep="first").sort(KEYS)
    index = first.sample(n=min(sample, first.height), seed=SEED)["index"]
    picked = rows.select(pl.col("wave", "skewness").gather(index))
    got = skewness(picked["wave"].to_numpy())
    want = picked["skewness"].cast(pl.Float64).to_numpy()
    off = ~np.isclose(got, want, rtol=0, atol=tolerance, equal_nan=True)
    if off.any():
        raise ValueError(
            f"{off.sum()} of {off.size} sampled waves have a skewness more than "
            f"{tolerance} from the table's (largest gap {np.nanmax(np.abs(got - want)):.3g}): "
            "the waves were built for other keys, flipped or misaligned. Rebuild the "
            "table with `python -m src.vitals.rr_table`."
        )


def optimiser(
    model: nn.Module, steps_per_epoch: int, epochs: int, lr: float = LR
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW, decay on matrices only, then linear warmup into cosine decay to
    `LR_FLOOR` of `lr` (module docstring, "Loss and optimiser")."""
    decayed, exempt = [], []
    for _, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            no_decay = (parameter.ndim <= 1 or getattr(parameter, "_no_weight_decay", False)
                        or name.endswith("bias"))  # fmt: skip
            (exempt if no_decay else decayed).append(parameter)
    adamw = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": WEIGHT_DECAY},
            {"params": exempt, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(total * WARMUP_FRAC))
    cosine = lambda step: (
        0.5
        * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    )
    factor = lambda step: (
        (step + 1) / warmup
        if step < warmup
        else LR_FLOOR + (1 - LR_FLOOR) * cosine(step)
    )
    return adamw, torch.optim.lr_scheduler.LambdaLR(adamw, factor)


def predict(model: RRMamba, x: torch.Tensor, batch: int = EVAL_BATCH) -> torch.Tensor:
    """Estimates (N,) on the CPU, in eval mode."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return torch.cat([model(chunk.to(device)).cpu() for chunk in x.split(batch)])


def fit(
    model: RRMamba,
    train: pl.DataFrame,
    dev: pl.DataFrame,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    lr: float = LR,
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
    train_mean: float | None = None,
) -> tuple[dict, pl.DataFrame]:
    """Train `model` on `train`, choosing the epoch on `dev` (both `join` output).
    Returns the kept checkpoint, the highest pooled dev slope, and one history row
    per epoch. Test rows are not an argument, so they can't reach training.

    A checkpoint holds `state_dict` (on the CPU), `epoch`, `dev_mae`, `dev_rmse`,
    `dev_slope`, its `history` row and `config`: "model" rebuilds the network
    (`load`), "train" records the run. With `checkpoint_dir`, every epoch's is also
    written there as epoch_NNN.pt. `train_mean`, the dev baseline the history logs,
    defaults to the model's `init_mean`.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if checkpoint_dir is not None:
        _refuse_old_epochs(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_mean = model.config["init_mean"] if train_mean is None else train_mean
    torch.manual_seed(SEED)
    model.to(device)
    train_x, train_y = _tensors(train, device)
    natural_x, natural_y = _tensors(natural_rows(train), "cpu")
    dev_x, dev_y = _tensors(dev, "cpu")
    config = {
        "model": model.config,
        "train": {
            "epochs": epochs, "batch": batch, "lr": lr, "weight_decay": WEIGHT_DECAY,
            "warmup_frac": WARMUP_FRAC, "floor": LR_FLOOR, "clip": GRAD_CLIP, "seed": SEED,
            "loss": f"Huber(delta={HUBER_DELTA})", "train_rows": len(train_x),
        },
    }  # fmt: skip
    baseline_mae, baseline_rmse = _mae_rmse(
        torch.full_like(dev_y, train_mean, dtype=torch.float64), dev_y.double()
    )
    adamw, scheduler = optimiser(model, math.ceil(len(train_x) / batch), epochs, lr)
    # On the CPU, so the order of batches doesn't depend on the device.
    order = torch.Generator().manual_seed(SEED)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        start, first_lr = time.perf_counter(), scheduler.get_last_lr()[0]
        model.train()
        total = torch.zeros((), device=device)
        for index in torch.randperm(len(train_x), generator=order).split(batch):
            index = index.to(device)
            loss = loss_fn(model(train_x[index]), train_y[index])
            adamw.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            adamw.step()
            scheduler.step()
            total += loss.detach() * len(index)
        estimate = predict(model, dev_x)
        train_mae, train_rmse = _mae_rmse(predict(model, natural_x), natural_y)
        dev_mae, dev_rmse = _mae_rmse(estimate, dev_y)
        row = {
            "epoch": epoch, "train_loss_running": total.item() / len(train_x),
            "train_mae": train_mae, "train_rmse": train_rmse, "dev_mae": dev_mae, "dev_rmse": dev_rmse,
            "baseline_dev_mae": baseline_mae, "baseline_dev_rmse": baseline_rmse,
            "dev_slope": _slope(estimate.double(), dev_y.double()),
            "dev_spread": _spread(estimate.double(), dev_y.double()),
            "lr": first_lr, "seconds": time.perf_counter() - start,
        }  # fmt: skip
        history.append(row)
        saved = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "epoch": epoch, "dev_mae": dev_mae, "dev_rmse": dev_rmse, "dev_slope": row["dev_slope"],
            "history": row, "config": config,
        }  # fmt: skip
        if checkpoint_dir is not None:
            torch.save(saved, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if best is None or row["dev_slope"] > best["dev_slope"]:
            best = saved
    return best, pl.DataFrame(history)


def score(predictions: pl.DataFrame, train_mean: float) -> pl.DataFrame:
    """Scores of `predictions` (split, source, rr_bpm, predicted) per split, over all
    sources (source "all") and then per source (module docstring, "Scores")."""
    y, error = pl.col(TARGET), pl.col("predicted") - pl.col(TARGET)
    both = pl.concat([predictions.with_columns(source=pl.lit("all")), predictions])
    return (
        both.group_by("split", "source")
        .agg(rows=pl.len(), mae=error.abs().mean(), rmse=error.pow(2).mean().sqrt(),
             r=pl.corr("predicted", TARGET), slope=pl.cov("predicted", TARGET) / y.var(),
             spread=pl.col("predicted").std() / y.std(),
             mae_low_tail=error.abs().filter(y <= y.quantile(TAIL, "lower")).mean(),
             mae_high_tail=error.abs().filter(y >= y.quantile(1 - TAIL, "higher")).mean(),
             mae_train_mean=(y - train_mean).abs().mean())
        .sort(pl.col("source") != "all", "source",
              pl.col("split").replace_strict(SPLIT_ORDER, return_dtype=pl.Int8))
        .select(pl.lit(TARGET).alias("target"), pl.all())
    )  # fmt: skip


def main(
    table: Path = TABLE,
    out_dir: Path | None = None,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    lr: float = LR,
    device: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Train, keep the epoch with the highest dev slope, score train, dev and test;
    write the files the module docstring lists. Returns the scores.

    The waves are `PP_FILE` next to `table`; `out_dir` defaults to rr_mamba/ there.
    `device` defaults to "cuda" when there is one ("cuda" is also ROCm's name); the
    scan is the kernel there and "export" on the CPU. `limit` keeps that many train
    rows, drawn at random: for smoke runs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scan = "kernel" if str(device).startswith("cuda") else "export"
    waves = table.parent / PP_FILE
    out_dir = out_dir or table.parent / NAME
    # Before any data is read, so a stale output dir fails in seconds.
    _refuse_old_epochs(out_dir / "checkpoints")
    frame = pl.read_parquet(table)
    natural = frame[NATURAL_MEAN].unique()
    if natural.len() != 1:
        raise ValueError(
            f"{table}: expected one natural train mean, got {natural.to_list()}"
        )
    train_mean = float(natural[0])
    rows, _ = join(frame, pl.read_parquet(waves))
    check_waves(rows)
    for column in INPUTS:
        if bad := int((~np.isfinite(rows[column].to_numpy()).all(axis=1)).sum()):
            raise ValueError(
                f"{bad} of {rows.height} rows of {waves} hold a non-finite {column}"
            )
    train = rows.filter(_in_split("train"))
    if limit is not None:
        train = train.sample(n=min(limit, train.height), seed=SEED)

    torch.manual_seed(SEED)
    model = RRMamba(init_mean=train_mean, scan=scan)
    print(
        f"{NAME}: {sum(p.numel() for p in model.parameters())} parameters, scan {scan}, "
        f"{device}, {train.height} train rows, natural train mean {train_mean:.3f}, "
        f"{WAVE_FPS / 2**DOWN_STAGES:g} Hz after the stem",
        flush=True,
    )
    checkpoint, history = fit(
        model, train, rows.filter(_in_split("dev")), epochs, batch, lr,
        device=device, checkpoint_dir=out_dir / "checkpoints", train_mean=train_mean,
    )  # fmt: skip
    torch.save(checkpoint, out_dir / f"{NAME}_{TARGET}.pt")
    history_path = out_dir / f"{NAME}_{TARGET}_history.parquet"
    history.write_parquet(history_path)

    model.load_state_dict(checkpoint["state_dict"])
    scored = pl.concat([natural_rows(train), rows.filter(_in_split("dev", "test"))])
    estimate = predict(model, _tensors(scored, "cpu")[0])
    predictions = scored.select(*KEYS, "source", "split", TARGET).with_columns(
        predicted=pl.Series(estimate.numpy()), train_mean=pl.lit(train_mean)
    )
    scores = score(predictions, train_mean)
    scores.write_parquet(out_dir / f"{NAME}_metrics.parquet")
    predictions.filter(_in_split("test")).write_parquet(
        out_dir / f"{NAME}_test_predictions.parquet"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(f"baseline: the natural train mean on dev, MAE {history['baseline_dev_mae'][0]:.3f} "
              f"RMSE {history['baseline_dev_rmse'][0]:.3f}")  # fmt: skip
        print(history.drop("baseline_dev_mae", "baseline_dev_rmse"))
        print(f"kept epoch {checkpoint['epoch']} (highest dev slope {checkpoint['dev_slope']:.3f}, "
              f"RMSE {checkpoint['dev_rmse']:.3f}, MAE {checkpoint['dev_mae']:.3f})")  # fmt: skip
        print(scores)
    # Last: every data file is on disk, so a failing plot loses nothing.
    plot.rr_mamba_loss(history_path)
    print(f"wrote {out_dir}")
    return scores
