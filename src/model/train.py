"""Training loop for CFMamba-Phys.

Waveform supervision only: one target per frame, taken from the contact PPG, and
heart rate read off the prediction afterwards rather than learned. The scalar
regression this file used to run is gone -- a clip-level label departs from the
true rate of any given 5.33 s window by 4.02 bpm on this corpus, so it was the
wrong target before it was a hard optimisation.

**No validation split, and no checkpoint selection.** Neither source paper has one
-- "due to the absence of the validation set, we selected the checkpoint from the
last epoch" (RhythmMamba 4.3) -- and inventing one over twelve test subjects would
make every number incomparable while inviting selection on the thing being
measured. Test is scored each epoch so the trajectory is visible, and the weights
kept are always the last epoch's. That means the epoch budget has to be decided in
advance and not tuned afterwards.

**POS and CHROM are scored on the same windows before training starts.** They need
no training and publish ~4.06-4.08 bpm MAE on UBFC-rPPG. A run that ends above
them has not learned anything a decade-old algorithm does not already do, and
the baseline is therefore printed before the first epoch.
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

from ..paths import BUILD_ROOT
from .baselines import run as run_baselines
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

# CLBP-300's five clips ship as plain .mov files with the labels encoded in the
# filename and no waveform at all, so they cannot support per-frame supervision.
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
    accum_steps: int = 1
    workers: int = 8
    seed: int = 20260822
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
    cam_pooling: str = "cmamba"
    ffn_activation: str | None = "gelu"

    # "paper" is both source papers' UBFC split, for comparability. "random" is a
    # subject-grouped 85/10/5 over every segment in the manifest, which is what to
    # use when the manifest is more than UBFC.
    # Matches the CLI. The pooled manifest carries its own split, and
    # re-deriving one here would silently train on a different partition.
    protocol: str = "manifest"
    # Cap the per-epoch dev pass. Dev is 11,984 segments on the full corpus, so
    # scoring all of it every epoch costs as much as the training does -- and the
    # trajectory does not need that precision, it needs to be visible. The full dev
    # and test splits are scored once at the end, unlimited.
    dev_eval_segments: int = 1500
    frame_norm: str = "standardized"
    sources: tuple[str, ...] = ()          # empty means every source with a waveform
    stride_frames: int | None = None        # None means non-overlapping segments
    baselines: bool = True
    log_every: int = 25
    # Synchronise inside the training loop so compute_s is compute rather than
    # queue-submission time. Off by default: it serialises the pipeline, so it
    # measures the split truthfully and makes the run slower while it does.
    profile: bool = False
    # Continue from out_dir/last.pt instead of starting over. The schedule inputs
    # must match what the checkpoint was written under; see check_resumable.
    resume: bool = False
    out_dir: Path = field(default_factory=lambda: BUILD_ROOT / "runs" / "cfmamba")


# Evaluation loaders get a fraction of the worker budget, and never persist.
#
# This is not a tuning preference, it is a correctness fix. A run builds four
# loaders -- train, dev, test, and the limited per-epoch watch split -- and giving
# every one of them `num_workers=12, persistent_workers=True` leaves **48 worker
# processes** alive simultaneously, each holding an ffmpeg child, decode buffers and
# pinned host memory. Measured on a 16-thread machine: load average 23.5, GPU
# utilisation 1-7%, 95% of VRAM consumed, and batch 10 running *slower* (2.2 clip/s)
# than batch 12 (2.8 clip/s) because the contention outweighed the larger batch.
# Only the training loader needs to stay warm.
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
        # which means a second set of cuFFT plans and a second workspace on a card
        # that has already run out once -- cuFFT reports that as
        # CUFFT_INTERNAL_ERROR rather than as an allocation failure, so the cause is not
        # apparent from the message. Evaluation keeps every window, so it does not
        # drop.
        drop_last=train,
    )


def build_model(cfg: TrainConfig) -> CFMambaPhys:
    return CFMambaPhys(
        n_frames=cfg.n_frames, fps=TARGET_FPS, stem_variant=cfg.stem_variant,
        fuse_stem=cfg.fuse_stem, use_pga=cfg.use_pga, use_cam=cfg.use_cam,
        ffn=cfg.ffn, pts_mode=cfg.pts_mode, direction=cfg.direction,
        cam_pooling=cfg.cam_pooling, ffn_activation=cfg.ffn_activation,
    )


def _optimiser(
    model: CFMambaPhys, cfg: TrainConfig, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with two parameter groups, then linear warmup into cosine decay.

    **Two groups, because weight decay does not belong on everything.** Decaying a
    LayerNorm gain pulls it toward zero, which is a rescaling of the layer rather
    than a regularisation; decaying a bias just shifts it. The convention is to
    exempt every 1-D parameter, and it matters more than usual here for two
    specific tensors:

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

    **Warmup then cosine.** No paper in the lineage states a schedule. Cosine decay
    after a short linear warmup is the AdamW convention for vision models of this
    class, and is what this project's previous MambaVision-based configuration used
    minus the warmup. The warmup is the addition: at batch 4 the first steps carry
    both the largest gradients the model will ever see and the noisiest estimate of
    them, and a cosine schedule alone uses its highest learning rate exactly
    there.
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

    total = max(1, steps_per_epoch * cfg.epochs // max(cfg.accum_steps, 1))
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
# of length `steps_per_epoch * epochs // accum_steps`, and `steps_per_epoch` follows
# from the manifest, the batch size and the window length. A checkpoint stores the
# scheduler's step counter, so resuming with any of these changed would apply that
# counter to a differently-shaped schedule.
SCHEDULE_FIELDS = ("epochs", "batch_size", "n_frames", "accum_steps")


def save_checkpoint(
    path: Path, *, model, optimiser, scheduler, epoch: int,
    history: list[dict], config: dict, steps_per_epoch: int,
) -> None:
    """Everything needed to continue this run, not only to reuse its weights.

    Weights alone give a warm restart: AdamW's moments are rebuilt from scratch and
    the scheduler restarts its warmup, jumping the learning rate back to peak. For a
    30-epoch run that is a different experiment from the one that was interrupted.

    `epoch` is the **next** epoch to run, so a resume needs no arithmetic.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "epochs": epoch,          # kept for readers of the old two-key format
        "history": history,
        "config": config,
        "steps_per_epoch": steps_per_epoch,
        "torch_rng": torch.get_rng_state(),
    }, path)


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

    Silent mismatch is the failure worth preventing: the run would restore a step
    counter into a cosine of a different length and carry on, producing a learning
    rate curve that matches neither the interrupted run nor a fresh one, with
    nothing in the log to say so.
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

    Called after **every** epoch, not just the last. Both artefacts used to be
    written only once the final epoch returned, so an interrupted run left an empty
    output directory however far it had got -- twice on this project a run was
    terminated after 2.6 hours and its completed epochs existed only in a terminal
    log. An epoch here costs over an hour; a JSON dump costs nothing.

    `complete` distinguishes two finished epochs of six from a finished two-epoch
    run, which the history alone cannot express. The caller's `result` is not
    altered, so the final write can still build on it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = result | {"history": history, "complete": complete}
    (out_dir / "history.json").write_text(json.dumps(partial, indent=2))


def check_targets_are_supervised(
    dataset, sample: int = 64, tolerance: float = 0.2
) -> None:
    """Fail noisily if the contact PPG is not reaching the loss.

    A missing waveform does not raise anywhere: `load_ppg` returns None,
    `_waveform` returns zeros, and every tensor downstream keeps its shape. The
    run then trains against a flat target -- `neg_pearson` fixes at exactly 1.0 and
    the frequency term reduces onto the one constant label that argmax of a zero
    PSD produces, which reads in the log as fast progress rather than as failure.

    That is not theoretical. Repointing `video_path` to the remuxed containers
    severed MCD's labels from its clips, because the PPG is located relative to the
    video, and a 1.3-hour epoch trained on zeros before the dropped-window count
    gave it away.

    Only the PPG is modified, never the video, so this costs a few hundred
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

    Every split is a segment table, not a clip table. The 85/10/5 target is in
    segments because that is what the model sees, and clip length varies four-fold
    across this corpus -- splitting clips and expanding afterwards gives the right
    number of recordings per side and the wrong number of examples.
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

    if cfg.protocol == "manifest":
        # The split the manifest already carries, rather than one derived here.
        # `src.cli mrnirp` balances MR-NIRP in segments, stratifies it by corpus
        # and persists the result, precisely so the assignment cannot move when
        # the manifest grows -- re-deriving it here would discard all of that and
        # silently train on a different partition.
        if "split" not in manifest.columns:
            raise ValueError(
                f"--protocol manifest needs a `split` column; {manifest_path} has "
                f"none. Build one with `src.cli mrnirp`, or use --protocol random."
            )
        missing = [n for n in ("train", "dev", "test")
                   if manifest.filter(pl.col("split") == n).height == 0]
        if missing:
            raise ValueError(f"manifest split is missing {', '.join(missing)}")
        # Expanded after partitioning, as with `paper`: the persisted split was
        # already balanced in segments, so re-balancing would undo it.
        return {
            name: expand_to_segments(
                manifest.filter(pl.col("split") == name),
                cfg.n_frames, TARGET_FPS, cfg.stride_frames,
            )
            for name in ("train", "dev", "test")
        }
    if cfg.protocol != "random":
        raise ValueError(
            f"unknown protocol {cfg.protocol!r}, expected manifest or random"
        )
    return prepare_splits(
        manifest, n_frames=cfg.n_frames, fps=TARGET_FPS, seed=cfg.seed,
        stride_frames=cfg.stride_frames,
    )


def train(
    cfg: TrainConfig,
    manifest_path: Path = BUILD_ROOT / "clips_ubfc.parquet",
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
    # A cheap, fixed subsample of dev for the per-epoch curve. Deterministic, so
    # the trajectory is comparable across runs rather than a different sample each
    # time; subject coverage follows from sampling segments evenly.
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

    # Before anything expensive: confirm the labels actually reach the loss.
    check_targets_are_supervised(loaders["train"].dataset)

    model = build_model(cfg).to(device)
    print(f"\n{model.describe()}")

    result: dict = {"config": {k: str(v) if isinstance(v, Path) else v
                               for k, v in asdict(cfg).items()}}

    if cfg.baselines:
        print("\n--- classical baselines on the test windows (no training) ---")
        rows = run_baselines(loaders["test"], fps=TARGET_FPS)
        result["baselines"] = {}
        for name, method_rows in rows.items():
            metrics = summarise(method_rows)
            result["baselines"][name] = metrics
            print(format_metrics(name.upper(), metrics))

    total_steps = len(loaders["train"])
    optimiser, scheduler = _optimiser(model, cfg, total_steps)

    # --- resume -------------------------------------------------------------
    start_epoch = 0
    history: list[dict] = []
    checkpoint = cfg.out_dir / "last.pt"
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
        # Data-wait against compute. Without this the epoch time states the run is
        # slow but not which half is slow, which is the question that determines
        # whether to touch the loader or the model.
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
            (loss / cfg.accum_steps).backward()
            if (step + 1) % cfg.accum_steps == 0:
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                # Per step, not per epoch: an epoch here is 100-25000 steps
                # depending on the corpus, so an epoch-level schedule would mean a
                # completely different curve for each.
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

        if total_steps % cfg.accum_steps:
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()

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

        print(f"epoch {epoch:3d}  lr {record['lr']:.2e}  train {record['train_loss']:.4f} "
              f"(time {record['train_time']:.3f} freq {record['train_freq']:.3f}){gap}  "
              f"{format_metrics(watch, metrics)}  "
              f"({record['seconds']:.0f}s, fetch {fetch_s:.0f}s)",
              flush=True)

    torch.save({"model": model.state_dict(), "epochs": cfg.epochs,
                "config": result["config"]}, cfg.out_dir / "final.pt")

    print("\n--- final, last epoch ---")
    for name in ("dev", "test"):
        if name in loaders and name != "test":
            full = summarise(evaluate(model, loaders[name], device, TARGET_FPS,
                                      alpha=cfg.alpha, beta=cfg.beta))
            print(format_metrics(f"{name} (full)", full))
            result[f"{name}_full"] = full
    rows = evaluate(model, loaders["test"], device, TARGET_FPS,
                    alpha=cfg.alpha, beta=cfg.beta)
    final = summarise(rows)
    print(format_metrics("test", final))
    if cfg.baselines:
        for name, metrics in result.get("baselines", {}).items():
            verdict = "BEATEN" if final.get("mae", 1e9) < metrics.get("mae", 1e9) else "NOT beaten"
            print(f"  vs {name.upper():6} {metrics.get('mae', float('nan')):6.2f} bpm"
                  f"  -> {verdict}")
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
