"""Training loop for CFMamba-Phys.

Waveform supervision: one target per frame, taken from the contact PPG. Heart rate
is read off the prediction afterwards rather than learned. A clip-level label
departs from the true rate of a given 5.33 s window by 4.02 bpm on this corpus.

**The reported result is the last epoch.** Both source papers report that --
"due to the absence of the validation set, we selected the checkpoint from the
last epoch" (RhythmMamba 4.3) -- so the epoch budget is fixed in advance rather
than tuned on the result.

A dev split is scored each epoch, subsampled to `dev_eval_segments`, for the
trajectory. `<out>/best.pt` is written whenever an epoch sets a new lowest dev
score, because `last.pt` is overwritten every epoch and cannot be rolled back. It
is a second artefact; the reported number stays the last epoch's.

POS and CHROM are available as a floor through `src.cli baseline`, which scores
them on the same windows. They publish ~4.06-4.08 bpm MAE on UBFC-rPPG.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from ..aggregation.splits import DEFAULT_SEED, RATIOS
from ..paths import BUILD_ROOT
from .cfmamba import CFMambaPhys
from .dataset import (
    DEFAULT_CLIP_FRAMES,
    MODEL_RES,
    TARGET_FPS,
    WindowDataset,
    expand_to_segments,
    load_manifest,
    prepare_splits,
)
from .evaluate import evaluate, format_metrics, per_source, per_subject, summarise
from .losses import DEFAULT_ALPHA, DEFAULT_BETA, composite_loss


@dataclass
class TrainConfig:
    # RhythmFormer 4.2: batch 4, 30 epochs, at 128x128. No paper in the lineage
    # states an optimiser setting, so these are AdamW's own defaults where it has
    # them (lr, betas, eps) and the ViT/MambaVision-class convention where it does
    # not (weight decay 0.05, cosine decay with a linear warmup). See _optimiser.
    epochs: int = 50
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.05
    # Fraction of total steps spent warming the learning rate up from zero. The
    # first steps of a freshly initialised model produce the largest gradients it
    # will ever see, and at batch 4 they are also the noisiest.
    warmup_frac: float = 0.05
    # Cosine floor, as a fraction of lr. Decaying to exactly zero wastes the last
    # epoch; 1% keeps it doing something without letting it wander.
    min_lr_frac: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 4
    workers: int = 8
    seed: int = DEFAULT_SEED
    n_frames: int = DEFAULT_CLIP_FRAMES
    resolution: int = MODEL_RES

    # Loss weights, RhythmFormer 3.4 / Table 13.
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA

    # Augmentation.
    hr_balance: bool = True
    flip: bool = True
    apply_skin_mask: bool = False

    # Architecture and ablations. Defaults are the reconstruction; each flag
    # corresponds to a row of CFMamba Table 5 or RhythmFormer Table 4.
    stem_variant: str = "rhythmmamba"
    fuse_stem: bool = True
    use_pga: bool = True
    use_cam: bool = True
    ffn: str = "df"
    pts_mode: str = "channel"
    direction: str = "none"
    ffn_activation: str | None = "gelu"

    # How the split is obtained. "auto" reads the manifest's own `split` column
    # when it has one and derives a subject-grouped split when it does not, and
    # says which it used. "manifest" and "random" force one or the other; a
    # forced "manifest" against a manifest with no column raises rather than
    # silently training on a different partition.
    protocol: str = "auto"
    # Cap the per-epoch dev pass. Dev is 11,984 segments on the full corpus, so
    # scoring all of it every epoch costs as much as the training does -- and the
    # trajectory does not need that precision, it needs to be visible. The full dev
    # and test splits are scored once at the end, unlimited.
    dev_eval_segments: int = 1500
    frame_norm: str = "standardized"
    sources: tuple[str, ...] = ()          # empty means every source with a waveform
    stride_frames: int | None = None        # None means non-overlapping segments
    log_every: int = 25
    # Synchronise inside the training loop so compute_s is compute rather than
    # queue-submission time. Off by default: it serialises the pipeline, so it
    # measures the split truthfully and makes the run slower while it does.
    profile: bool = False
    # Continue from out_dir/last.pt instead of starting over. The schedule inputs
    # must match what the checkpoint was written under; see check_resumable.
    resume: bool = False
    # Metric on the per-epoch dev pass that <out>/best.pt tracks; see best_index.
    # Lower is better, so BEST_METRICS only. The dev pass is a subsample
    # (dev_eval_segments), which makes this a noisy criterion.
    best_metric: str = "loss"
    out_dir: Path = field(default_factory=lambda: BUILD_ROOT / "runs" / "cfmamba")


# Evaluation loaders get a fraction of the worker budget, and never persist.
#
# A run builds four loaders -- train, dev, test, and the per-epoch watch split. At
# `num_workers=12, persistent_workers=True` on each, 48 worker processes stay alive
# at once, each holding an ffmpeg child, decode buffers and pinned host memory.
# Measured on a 16-thread machine: load average 23.5, GPU utilisation 1-7%, 95% of
# VRAM consumed, and batch 10 running slower (2.2 clip/s) than batch 12 (2.8 clip/s)
# under the contention. Only the training loader stays warm.
EVAL_WORKER_SHARE = 3


def _loader(segments: pl.DataFrame, cfg: TrainConfig, train: bool) -> DataLoader:
    """`segments` is already expanded -- see build_splits."""
    dataset = WindowDataset(
        segments,
        n_frames=cfg.n_frames, train=train, seed=cfg.seed,
        resolution=cfg.resolution, apply_skin_mask=cfg.apply_skin_mask,
        frame_norm=cfg.frame_norm, hr_balance=cfg.hr_balance, flip=cfg.flip,
        return_waveform=True,
    )
    workers = cfg.workers if train else max(1, cfg.workers // EVAL_WORKER_SHARE)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
        # Persist only the training loader. An evaluation loader that stays warm
        # keeps its workers idle for the whole epoch it is not being used in.
        persistent_workers=bool(workers) and train,
        pin_memory=train,
        # Uniform training batches. A short final batch is a second tensor shape,
        # so a second set of cuFFT plans and a second workspace. cuFFT reports the
        # resulting allocation failure as CUFFT_INTERNAL_ERROR, which does not name
        # the cause. Evaluation keeps every window, so it does not drop.
        drop_last=train,
    )


def build_model(cfg: TrainConfig) -> CFMambaPhys:
    return CFMambaPhys(
        n_frames=cfg.n_frames, fps=TARGET_FPS, stem_variant=cfg.stem_variant,
        fuse_stem=cfg.fuse_stem, use_pga=cfg.use_pga, use_cam=cfg.use_cam,
        ffn=cfg.ffn, pts_mode=cfg.pts_mode, direction=cfg.direction,
        ffn_activation=cfg.ffn_activation,
    )


def _optimiser(
    model: CFMambaPhys, cfg: TrainConfig, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with two parameter groups, then linear warmup into cosine decay.

    Two groups, because weight decay does not apply to every parameter. Decaying a
    LayerNorm gain pulls it toward zero, which rescales the layer rather than
    regularising it; decaying a bias shifts it. The convention is to exempt every
    1-D parameter, and two tensors here need it specifically:

      `theta_fc`, `theta_bw`   the Gaussian band's centre and width (Eq. 14). They
                               are 0-dim, and decay would drag them toward zero --
                               which after the sigmoid is the *midpoint* of the
                               physiological range, 1.625 Hz. That is a prior on
                               heart rate masquerading as regularisation.
      `dt_bias`, `D`           Mamba-3's step-size and skip parameters, which
      `B_bias`, `C_bias`       `mamba_ssm` marks `_no_weight_decay`, plus the B and
                               C biases, which mamba_layer.py marks. The latter two
                               are 3-D, so a dimension rule alone would miss them.
                               The flag is respected directly.

    Warmup then cosine. No paper in the lineage states a schedule. Cosine decay
    after a short linear warmup is the AdamW convention for vision models of this
    class. At batch 4 the first steps carry both the largest gradients of the run
    and the noisiest estimate of them, and a cosine schedule alone applies its
    highest learning rate there.
    """
    decay, no_decay = [], []
    for _, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            exempt = (
                parameter.ndim <= 1
                or getattr(parameter, "_no_weight_decay", False)
                or name.endswith("bias")
            )
            (no_decay if exempt else decay).append(parameter)

    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
    )

    total = max(1, steps_per_epoch * cfg.epochs)
    warmup = max(1, int(total * cfg.warmup_frac))
    floor = cfg.min_lr_frac

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, factor)
    print(f"\nAdamW lr {cfg.lr:.1e} betas {cfg.betas} eps {cfg.eps:.0e}  "
          f"wd {cfg.weight_decay} on {len(decay)} tensors, 0.0 on {len(no_decay)}")
    print(f"cosine to {floor:.0%} of lr over {total} steps, "
          f"linear warmup over the first {warmup}  |  grad clip {cfg.grad_clip}")
    return optimiser, scheduler


# Fields the learning-rate schedule is a function of. `_optimiser` builds a cosine
# of length `steps_per_epoch * epochs`, and `steps_per_epoch` follows from the
# manifest, the batch size and the window length. A checkpoint stores the
# scheduler's step counter, so resuming with any of these changed would apply that
# counter to a differently-shaped schedule.
SCHEDULE_FIELDS = ("epochs", "batch_size", "n_frames")


def save_checkpoint(
    path: Path, *, model, optimiser, scheduler, epoch: int,
    history: list[dict], config: dict, steps_per_epoch: int,
) -> None:
    """Everything needed to continue this run, not only to reuse its weights.

    Weights alone give a warm restart: AdamW's moments are rebuilt from scratch and
    the scheduler restarts its warmup, returning the learning rate to peak. For a
    30-epoch run that is a different experiment from the interrupted one.

    `epoch` is the **next** epoch to run, so a resume needs no arithmetic.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": config,
        "steps_per_epoch": steps_per_epoch,
        "torch_rng": torch.get_rng_state(),
    }, path)


# Where `build_splits` takes the split from. "auto" reads the manifest's column
# when it has one and derives a subject-grouped split when it does not.
PROTOCOLS = ("auto", "manifest", "random")


def _ratio_text(ratios: dict[str, float]) -> str:
    return "/".join(f"{100 * ratios[n]:.0f}" for n in ("train", "dev", "test"))


# Metrics best.pt can select on. All lower-is-better; a higher-is-better metric
# (corr, macc, snr) would select the worst epoch under this comparison, so it is
# refused rather than minimised.
BEST_METRICS = ("loss", "mae", "rmse")


def best_index(
    history: list[dict], split: str, metric: str = "loss"
) -> int | None:
    """Index in `history` of the epoch with the lowest dev `metric`, or None.

    `last.pt` is overwritten every epoch, so the weights of a better earlier epoch
    are gone the moment the next one finishes. Selection therefore has to happen
    during the run: each epoch asks whether it is the best dev score so far.

    Ties keep the earlier epoch. Only a strict improvement rewrites best.pt, so a
    plateau does not rewrite 11 MB for the same score.
    """
    if metric not in BEST_METRICS:
        raise ValueError(
            f"cannot select a checkpoint on {metric!r}: best.pt takes the lowest "
            f"score, and only {', '.join(BEST_METRICS)} are lower-is-better."
        )
    best, at = float("inf"), None
    for i, record in enumerate(history):
        value = (record.get(split) or {}).get(metric)
        # An absent or NaN metric is a measurement that did not happen rather than
        # a good one: an epoch predating the dev pass, or one whose readout produced
        # nothing.
        if value is None or not math.isfinite(value):
            continue
        if value < best:
            best, at = float(value), i
    return at


def load_checkpoint(path: Path) -> dict:
    """Read a checkpoint written by `save_checkpoint`.

    `weights_only=False` because the payload carries the history and config, not
    only tensors. These files are produced by this project, never downloaded.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def check_resumable(
    saved_config: dict, cfg: TrainConfig, saved_steps: int, steps_per_epoch: int
) -> None:
    """Reject a resume whose schedule is not the one that was saved.

    Without this the run restores a step counter into a cosine of a different
    length and continues, producing a learning-rate curve matching neither the
    interrupted run nor a fresh one, with nothing in the log to say so.
    """
    for name in SCHEDULE_FIELDS:
        was, now = saved_config.get(name), getattr(cfg, name)
        if was is not None and was != now:
            raise RuntimeError(
                f"cannot resume: {name} was {was} in the checkpoint and is {now} "
                f"now. The learning-rate schedule is a function of {name}, so the "
                f"saved step counter refers to a different curve. Start a new run, "
                f"or restore {name}={was}."
            )
    if saved_steps and saved_steps != steps_per_epoch:
        raise RuntimeError(
            f"cannot resume: steps per epoch was {saved_steps} in the checkpoint "
            f"and is {steps_per_epoch} now, so the manifest or split has changed "
            f"and the schedule no longer matches."
        )


def write_progress(
    out_dir: Path, result: dict, history: list[dict], complete: bool = False
) -> None:
    """Write `history.json` for the epochs finished so far.

    Called after each epoch rather than at the end of the run. An epoch here costs
    over an hour, so an interrupted run that wrote nothing would leave its completed
    epochs only in a terminal log.

    `complete` distinguishes two finished epochs of six from a finished two-epoch
    run, which the history alone cannot express. The caller's `result` is not
    altered, so the final write builds on it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = result | {"history": history, "complete": complete}
    (out_dir / "history.json").write_text(json.dumps(partial, indent=2))


def check_targets_are_supervised(
    dataset, sample: int = 64, tolerance: float = 0.2
) -> None:
    """Raise if the contact PPG is not reaching the loss.

    A missing waveform raises nowhere: `load_ppg` returns None, `_waveform` returns
    zeros, and every tensor downstream keeps its dimensions. The run then trains
    against a flat target, where `neg_pearson` sits at exactly 1.0 and the frequency
    term reduces onto the constant label that argmax of a zero PSD produces. In the
    log that reads as fast progress.

    Repointing `video_path` to the remuxed containers severed MCD's labels from its
    clips, because the PPG is located relative to the video, and a 1.3-hour epoch
    trained on zeros before the dropped-window count showed it.

    Only the PPG is read, never the video, so this costs a few hundred
    interpolations and runs before the first batch is decoded.
    """
    import numpy as np

    rows = dataset.rows
    if not rows:
        raise RuntimeError("the training split is empty")
    step = max(1, len(rows) // sample)
    picked = rows[::step][:sample]

    flat = 0
    for row in picked:
        start = float(row.get("window_start_s") or 0.0)
        wave = dataset._waveform(row, start, 1.0)
        if not np.isfinite(wave).all() or float(np.std(wave)) < 1e-6:
            flat += 1

    if flat > tolerance * len(picked):
        raise RuntimeError(
            f"{flat} of {len(picked)} sampled training windows have a flat or "
            f"non-finite target, so the contact PPG is not reaching the loss. "
            f"Check that the manifest's labels still resolve -- a repointed "
            f"video_path needs ppg_video_path alongside it."
        )


def build_splits(cfg: TrainConfig, manifest_path: Path) -> dict[str, pl.DataFrame]:
    """Segments per split, already expanded and filtered to what can be supervised.

    Each split is a segment table, not a clip table. The ratios are in segments
    because that is what the model sees, and clip length varies four-fold across
    this corpus: splitting clips and expanding afterwards gives the right number of
    recordings per side and the wrong number of examples.

    `cfg.protocol` selects where the split comes from -- see TrainConfig.
    """
    manifest = load_manifest(manifest_path)
    usable = tuple(manifest["source"].unique().to_list())

    keep = cfg.sources or usable
    if cfg.sources:
        excluded = [s for s in usable if s not in cfg.sources]
        if excluded:
            n = manifest.filter(pl.col("source").is_in(excluded)).height
            print(f"  excluded {n} clips by --sources ({', '.join(sorted(excluded))})")
    manifest = manifest.filter(pl.col("source").is_in(list(keep)))
    if manifest.height == 0:
        raise ValueError(f"no clips left after filtering to sources {keep}")

    if cfg.protocol not in PROTOCOLS:
        raise ValueError(
            f"unknown protocol {cfg.protocol!r}, expected one of {PROTOCOLS}"
        )
    has_column = "split" in manifest.columns
    if cfg.protocol == "manifest" and not has_column:
        raise ValueError(
            f"protocol 'manifest' needs a `split` column; {manifest_path} has none. "
            f"Build one with `src.cli combine`, or use protocol 'random'."
        )

    if cfg.protocol == "random" or (cfg.protocol == "auto" and not has_column):
        why = "" if has_column else f"; {manifest_path} carries no split column"
        print(f"  split derived here, subject-grouped {_ratio_text(RATIOS)}{why}")
        return prepare_splits(
            manifest, n_frames=cfg.n_frames, fps=TARGET_FPS, seed=cfg.seed,
            stride_frames=cfg.stride_frames,
        )

    # The split the manifest carries, rather than one derived here. `src.cli
    # combine` balances it in segments and stratifies it by source, then persists
    # the result so the assignment cannot move when the manifest grows.
    # Re-deriving it here would discard that and train on a different partition.
    missing = [n for n in ("train", "dev", "test")
               if manifest.filter(pl.col("split") == n).height == 0]
    if missing:
        raise ValueError(f"manifest split is missing {', '.join(missing)}")
    print(f"  split read from {manifest_path}")
    # Expanded after partitioning: the persisted split was already balanced in
    # segments, so re-balancing would undo it.
    return {
        name: expand_to_segments(
            manifest.filter(pl.col("split") == name),
            cfg.n_frames, TARGET_FPS, cfg.stride_frames,
        )
        for name in ("train", "dev", "test")
    }


def train(
    cfg: TrainConfig,
    manifest_path: Path = BUILD_ROOT / "clips_all.parquet",
    splits: dict[str, pl.DataFrame] | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: Mamba-3's scan kernel has no CPU path.")
    device = "cuda"
    torch.manual_seed(cfg.seed)
    # Every batch is the same shape -- (batch, n_frames, 3, resolution, resolution)
    # -- so cuDNN's autotuner covers for itself on the first step and the stem's
    # convolutions run on the plan it picks for the rest of the run.
    torch.backends.cudnn.benchmark = True
    # TF32 for the float32 matmuls the autocast region does not cover. The signal
    # is a 0.1-0.5 LSB change on an 8-bit source, so 10 mantissa bits is far more
    # precision than the input carries.
    torch.set_float32_matmul_precision("high")

    if splits is None:
        splits = build_splits(cfg, manifest_path)

    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a not in splits or b not in splits:
            continue
        overlap = set(splits[a]["subject_id"]) & set(splits[b]["subject_id"])
        if overlap:
            raise RuntimeError(f"subject leakage {a}/{b}: {sorted(overlap)[:5]}")

    loaders = {name: _loader(part, cfg, train=(name == "train"))
               for name, part in splits.items() if part.height}
    # A fixed subsample of dev for the per-epoch curve. Deterministic, so the
    # trajectory is comparable across runs rather than a different sample each time;
    # subject coverage follows from sampling segments evenly.
    watch_split = "dev" if "dev" in splits and splits["dev"].height else "test"
    watching = splits[watch_split]
    if cfg.dev_eval_segments and watching.height > cfg.dev_eval_segments:
        watching = watching.sample(cfg.dev_eval_segments, seed=cfg.seed, shuffle=True)
        print(f"  per-epoch {watch_split} pass limited at {watching.height} of "
              f"{splits[watch_split].height} segments "
              f"({watching['subject_id'].n_unique()} subjects); "
              f"the full split is scored once at the end")
    loaders["watch"] = _loader(watching, cfg, train=False)
    total = sum(part.height for part in splits.values())
    for name, part in splits.items():
        share = 100 * part.height / max(total, 1)
        print(f"  {name:5} {part.height:6} segments ({share:5.2f}%)  "
              f"{part['clip_id'].n_unique():4} clips  "
              f"{part['subject_id'].n_unique():4} subjects  "
              f"{sorted(part['source'].unique().to_list())}")

    # Before anything expensive: confirm the labels reach the loss.
    check_targets_are_supervised(loaders["train"].dataset)

    model = build_model(cfg).to(device)
    print(f"\n{model.describe()}")

    result: dict = {"config": {k: str(v) if isinstance(v, Path) else v
                               for k, v in asdict(cfg).items()}}

    total_steps = len(loaders["train"])
    optimiser, scheduler = _optimiser(model, cfg, total_steps)

    # --- resume -------------------------------------------------------------
    start_epoch = 0
    history: list[dict] = []
    checkpoint = cfg.out_dir / "last.pt"
    best_path = cfg.out_dir / "best.pt"
    # Checked before the first epoch, not at its save, which is an hour in.
    if cfg.best_metric not in BEST_METRICS:
        raise RuntimeError(
            f"best_metric {cfg.best_metric!r} is not one of {BEST_METRICS}."
        )
    if cfg.resume:
        if not checkpoint.exists():
            raise RuntimeError(f"--resume given but {checkpoint} does not exist")
        state = load_checkpoint(checkpoint)
        check_resumable(state.get("config", {}), cfg,
                        state.get("steps_per_epoch", 0), total_steps)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"].cpu())
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        if start_epoch >= cfg.epochs:
            raise RuntimeError(
                f"nothing to resume: the checkpoint has finished {start_epoch} of "
                f"{cfg.epochs} epochs. Raise --epochs for a longer schedule, which "
                f"starts a new run, or point --out somewhere else."
            )
        print(f"\nresumed from {checkpoint}: starting epoch {start_epoch} of "
              f"{cfg.epochs}, scheduler at step {scheduler.last_epoch}")
        # The inherited history includes epochs whose weights are gone, and best.pt
        # is compared against all of them. If nothing from here beats the score
        # below, no best.pt is written.
        at = best_index(history, watch_split, cfg.best_metric)
        if at is not None:
            score = history[at][watch_split][cfg.best_metric]
            print(f"best {watch_split} {cfg.best_metric} so far is {score:.4f} at "
                  f"epoch {history[at]['epoch']}, from before this resume. "
                  f"{best_path.name} is written only by an epoch that beats it.")
    elif checkpoint.exists():
        print(f"\nnote: {checkpoint} exists and will be overwritten. "
              f"Pass --resume to continue it instead.")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{total_steps} steps/epoch at batch {cfg.batch_size}, "
          f"{cfg.epochs} epochs, last epoch is the result\n")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        started = time.time()
        # Accumulated on the device. Summing Python floats here would mean a
        # device-to-host sync every step; see composite_loss's docstring.
        sums = {k: torch.zeros((), device=device) for k in ("loss", "time", "freq")}
        optimiser.zero_grad(set_to_none=True)
        window = {"loss": torch.zeros((), device=device), "n": 0}
        window_started = time.time()
        # Data-wait against compute. Epoch time alone says the run is slow without
        # saying which side is slow, which is what decides whether to change the
        # loader or the model.
        fetch_s = compute_s = 0.0
        waiting = time.perf_counter()

        for step, batch in enumerate(loaders["train"]):
            fetch_s += time.perf_counter() - waiting
            stepped = time.perf_counter()
            frames = batch["frames"].to(device, non_blocking=True)
            # non_blocking on all three: pin_memory is set on the train loader, so
            # without it these two are synchronous copies that undo the overlap the
            # first one just bought.
            skin = batch["skin"].to(device, non_blocking=True)
            target = batch["wave"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(frames, skin)
            loss, parts = composite_loss(
                predicted, target, fps=TARGET_FPS, alpha=cfg.alpha, beta=cfg.beta
            )
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            # Per step, not per epoch: an epoch here is 100-25000 steps depending
            # on the corpus, so an epoch-level schedule would be a different curve
            # for each.
            scheduler.step()
            for key in sums:
                sums[key] += parts[key]
            window["loss"] += parts["loss"]
            window["n"] += 1
            if cfg.profile:
                torch.cuda.synchronize()
            compute_s += time.perf_counter() - stepped

            if (step + 1) % cfg.log_every == 0:
                rate = window["n"] / (time.time() - window_started)
                # The one sync a logging window, rather than three a step.
                mean = float(window["loss"]) / window["n"]
                print(f"  e{epoch:02d} {step + 1:4d}/{total_steps}  "
                      f"loss {mean:.4f}  {rate * cfg.batch_size:5.1f} clip/s"
                      f"  (fetch {fetch_s:.1f}s compute {compute_s:.1f}s)", flush=True)
                window = {"loss": torch.zeros((), device=device), "n": 0}
                window_started = time.time()
            waiting = time.perf_counter()

        # Dev each epoch, test only at the end. Dev exists to make the trajectory
        # visible without touching the number that gets reported.
        watch = watch_split
        metrics = summarise(evaluate(model, loaders["watch"], device, TARGET_FPS,
                                     alpha=cfg.alpha, beta=cfg.beta))
        record = {
            "epoch": epoch,
            **{f"train_{k}": float(v) / max(total_steps, 1) for k, v in sums.items()},
            watch: metrics,
            "seconds": time.time() - started,
            # Wall clock split. fetch_s is time blocked on the dataloader;
            # compute_s is the step itself, and is only a true compute time when
            # cfg.profile forced a synchronise inside the loop.
            "fetch_s": fetch_s,
            "compute_s": compute_s,
        }
        # Pulled out of the nested metrics dict as well, so a loss curve is
        # `train_loss` against `dev_loss` at the top level -- the schema the
        # plotting tools already read.
        for term in ("loss", "time", "freq"):
            if term in metrics:
                record[f"{watch}_{term}"] = metrics[term]
        history.append(record)
        record["lr"] = optimiser.param_groups[0]["lr"]
        gap = (f" {watch} {record[f'{watch}_loss']:.4f}"
               if f"{watch}_loss" in record else "")
        write_progress(cfg.out_dir, result, history)
        save_checkpoint(checkpoint, model=model, optimiser=optimiser,
                        scheduler=scheduler, epoch=epoch + 1, history=history,
                        config=result["config"], steps_per_epoch=total_steps)

        # A second copy of the same payload whenever this epoch is the best dev
        # score of the run, because last.pt cannot be rolled back to it later. Full
        # checkpoint, not weights alone, so best.pt is loadable by the same readers.
        is_best = best_index(history, watch, cfg.best_metric) == len(history) - 1
        if is_best:
            save_checkpoint(best_path, model=model, optimiser=optimiser,
                            scheduler=scheduler, epoch=epoch + 1, history=history,
                            config=result["config"], steps_per_epoch=total_steps)

        print(f"epoch {epoch:3d}  lr {record['lr']:.2e}  train {record['train_loss']:.4f} "
              f"(time {record['train_time']:.3f} freq {record['train_freq']:.3f}){gap}  "
              f"{format_metrics(watch, metrics)}  "
              f"({record['seconds']:.0f}s, fetch {fetch_s:.0f}s)"
              f"{'  *best' if is_best else ''}",
              flush=True)

    torch.save({"model": model.state_dict(), "epoch": cfg.epochs,
                "config": result["config"]}, cfg.out_dir / "final.pt")

    print("\n--- final, last epoch ---")
    if "dev" in loaders:
        full = summarise(evaluate(model, loaders["dev"], device, TARGET_FPS,
                                  alpha=cfg.alpha, beta=cfg.beta))
        print(format_metrics("dev (full)", full))
        result["dev_full"] = full
    rows = evaluate(model, loaders["test"], device, TARGET_FPS,
                    alpha=cfg.alpha, beta=cfg.beta)
    final = summarise(rows)
    print(format_metrics("test", final))
    print("\nby source:")
    for name, metrics in per_source(rows):
        print(f"  {format_metrics(name, metrics)}")
    print("\nworst subjects:")
    for name, metrics in per_subject(rows)[:5]:
        print(f"  {format_metrics(name, metrics)}")

    result |= {"history": history, "test": final,
               "per_source": dict(per_source(rows)),
               "per_subject": dict(per_subject(rows)),
               "complete": True}
    (cfg.out_dir / "history.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {cfg.out_dir / 'history.json'} and {cfg.out_dir / 'final.pt'}")
    return result | {"model": model, "loaders": loaders, "splits": splits}
