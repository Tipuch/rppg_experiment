"""Command line interface for the rPPG pipeline.

    uv run python -m src.cli --help

Builds the clip manifest, renders the preprocessing for review, then trains and
scores CFMamba-Phys.
"""

from __future__ import annotations

from pathlib import Path

import click

from .aggregation.splits import DEFAULT_FRAMES, DEFAULT_SEED
from .paths import BUILD_ROOT

# What the consuming commands read: the pooled manifest `combine` writes, which
# is also what `train` defaults to. `clips` and `remux` produce CLIPS_MANIFEST
# instead; they are upstream of the pool, not readers of it.
DEFAULT_MANIFEST = BUILD_ROOT / "clips_all.parquet"
CLIPS_MANIFEST = BUILD_ROOT / "clips.parquet"
DEFAULT_RUN_DIR = BUILD_ROOT / "runs" / "cfmamba"
# Dataloader workers. `mrnirp` overrides this downward: each of its workers holds
# a ~1.2 GB SegFace copy on the GPU, so it trades against VRAM rather than CPU.
DEFAULT_WORKERS = 8


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Face-video pipeline for remote photoplethysmography.

    \b
    Order:
      1. clips     build the manifest (face boxes + skin masks, once per recording)
      2. mrnirp    ingest MR-NIRP from its zips; it ships stills, not video
      3. remux     rewrite MCD's container so seeking stops scanning from frame 0
      4. combine   pool the corpora into one manifest with one split
      5. cache     decode each face box once, so training reads 23 MB a window
      6. samples   render contact sheets for review of the preprocessing
      7. info      coverage, label spread and the split
      8. check     shapes, parameter budget and throughput on this GPU
      9. sanity    fit a synthetic pulse, as a control on the architecture
     10. baseline  POS and CHROM on the test windows, the classical floor
     11. train     fit the corpora for 50 epochs, then score
     12. predict   run one video through a trained model and plot the pulse
     13. readout   score every heart-rate readout against contact PPG

    Steps 1-5 are one-off. After them `train` takes no arguments: the pooled
    manifest, 90/3/7, 300-frame windows, 50 epochs. `--resume` continues an
    interrupted run.
    """


@cli.command()
@click.option("--out", type=click.Path(path_type=Path),
              default=BUILD_ROOT / "clips_all.parquet", show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--frames", type=int, default=DEFAULT_FRAMES, show_default=True,
              help="Window length the split is balanced in. Match --frames on train.")
def combine(out: Path, seed: int, frames: int) -> None:
    """Pool the corpora into one manifest with one split.

    Each corpus is built by its own command and lands in its own file. This joins
    them on a union schema, takes each corpus from one file, and assigns a single
    split over the pooled table, stratified by source, so each corpus reaches dev
    and test.

    MCD is 180 h against UBFC's 0.9 and MR-NIRP's 2.0, so it is ~98% of the
    segments under any split. Stratifying puts the other two in dev and test; it
    does not make them a large share. `train` prints the per-source breakdown and
    `--stride` there subsamples.
    """
    import polars as pl

    from .aggregation.combine import build
    from .aggregation.splits import summarise

    tagged = build(seed=seed, n_frames=frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    tagged.write_parquet(out)

    total = int(tagged["n_segments"].sum())
    click.echo(f"\n{tagged.height} clips, {tagged['subject_id'].n_unique()} subjects, "
               f"{tagged['duration_s'].sum() / 3600:.1f} h, {total} segments -> {out}")
    click.echo(summarise(tagged))
    click.echo("\nsegments per split and source:")
    click.echo(
        tagged.group_by("split", "source")
        .agg(pl.col("n_segments").sum().alias("segments"),
             pl.col("subject_id").n_unique().alias("subjects"))
        .with_columns((pl.col("segments") / total * 100).round(2).alias("pct_all"))
        .sort("split", "source")
    )


@cli.command()
@click.option("--downloads", type=click.Path(path_type=Path), default=None,
              help="Where the MR-NIRP zips are. Defaults to ~/Downloads.")
@click.option("--out", type=click.Path(path_type=Path),
              default=BUILD_ROOT / "clips_mrnirp.parquet", show_default=True,
              help="Where to write the manifest, split column included.")
@click.option("--cache-dir", type=click.Path(path_type=Path),
              default=BUILD_ROOT / "frames_cache", show_default=True,
              help="Frame cache. The same one the other corpora use, on purpose.")
@click.option("--limit", type=int, default=None,
              help="Prepare only the first N sessions.")
@click.option("--workers", type=int, default=2, show_default=True,
              help="Sessions prepared in parallel. Each holds its own ~1.2 GB "
                   "SegFace copy on the GPU, so this trades against VRAM, not CPU.")
@click.option("--force", is_flag=True, help="Rebuild sessions already cached.")
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
def mrnirp(downloads: Path | None, out: Path, cache_dir: Path, limit: int | None,
           workers: int, force: bool, seed: int) -> None:
    """Ingest MR-NIRP: nested zips of Bayer PGM stills -> the standard frame cache.

    MR-NIRP ships no video, so this does in one pass what `clips` and `cache` do
    for the other corpora, and writes the same artifacts. MR-NIRP then trains
    through the same path UBFC does.

    A session is usable only if it holds both an RGB stream and a pulse trace.
    Google Drive split the Car download by size and delivered the two parts in
    unrelated archives, so 29 Car sessions (10 subjects) and 15 Indoor sessions
    (8 subjects) come through: 44 in total.

    The split is subject-level and persisted. It packs largest subject first
    (`order="size"`), because 3% of 15 subjects is under one person and the
    shuffled fill sends its first subjects to the smallest bins, which on this
    corpus returns 64/14/21 rather than 90/3/7. The achieved ratios it prints are
    the ones to quote, not the requested ones.
    """
    import polars as pl

    from .aggregation.splits import RATIOS_POOLED, assign, segment_counts, summarise
    from .datasets import mrnirp as reader

    built = reader.build(downloads=downloads, cache_dir=cache_dir, limit=limit,
                         force=force, workers=workers)
    if not built.height:
        raise SystemExit("no sessions prepared")

    tagged = assign(
        built.with_columns(segment_counts().alias("n_segments")),
        ratios=RATIOS_POOLED, seed=seed, weight="n_segments", order="size",
        # Within each corpus, so Car and Indoor both reach dev and test. Without
        # it the fill gave dev 2 Car clips and test 3 Indoor ones, and the test
        # score then measured Indoor.
        stratify="corpus",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tagged.write_parquet(out)

    print(f"\n{tagged.height} clips, {tagged['subject_id'].n_unique()} subjects, "
          f"{tagged['duration_s'].sum() / 60:.1f} min -> {out}")
    print(tagged.group_by("corpus").agg(pl.len().alias("clips"),
                                        pl.col("subject_id").n_unique().alias("subjects")))
    print("\nsplit, by clip and subject:")
    print(summarise(tagged))
    # The ratios were balanced in segments, so that is the table to read them off.
    # `summarise`'s pct_rows above counts clips, a different number wherever clip
    # lengths differ; here they differ by 4x.
    print("achieved in segments, against requested "
          + ", ".join(f"{k} {100 * v:.0f}%" for k, v in RATIOS_POOLED.items()) + ":")
    print(tagged.group_by("split").agg(pl.col("n_segments").sum().alias("segments"))
          .with_columns((pl.col("segments") / tagged["n_segments"].sum() * 100)
                        .round(2).alias("pct_segments")).sort("split"))


@cli.command()
@click.option(
    "--limit", type=int, default=None,
    help="Cap clips read from each dataset. Omit for all of them.",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=CLIPS_MANIFEST,
    show_default=True, help="Where to write the clip manifest.",
)
@click.option("--force", is_flag=True,
              help="Rebuild from scratch instead of resuming from --out.")
def clips(limit: int | None, out: Path, force: bool) -> None:
    """Build the clip manifest: one row per recording.

    Runs YuNet once per clip for the face box and SegFace for a median skin mask,
    then records fps, duration and labels. SegFace is too slow to run per training
    batch, and its output is a few KB per clip, so the training loader is left with
    decoding and arithmetic.

    Resumable: rows already in --out are reused unless --force is given.
    """
    from .model import clips as clips_mod

    clips_mod.OUT_PARQUET = out
    raise SystemExit(clips_mod.main(limit, resume=not force))


@cli.command()
@click.option("--manifest", type=click.Path(exists=True, path_type=Path),
              default=CLIPS_MANIFEST, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Where to write the rewritten containers. Defaults to build/mcd_remux.")
@click.option("--corpus", default="mcd", show_default=True,
              help="Only remux clips from this source.")
@click.option("--force", is_flag=True, help="Rewrite clips that are already indexed.")
@click.option("--write-manifest", type=click.Path(path_type=Path), default=None,
              help="Also write a copy of the manifest with video_path repointed.")
def remux(manifest: Path, out_dir: Path | None, corpus: str, force: bool,
          write_manifest: Path | None) -> None:
    """Rewrite MCD's AVI container so ffmpeg can seek instead of scanning.

    MCD's AVIs carry no index -- ffprobe reports duration_ts=0, nb_frames=N/A and
    duration=N/A -- so `-ss` decodes forward from frame 0. Measured, that is 90% of
    the loader's per-window cost: a seek to 120 s costs up to 1,449 ms, against
    44 ms once indexed. Keyframes are dense here, at one every 0.4 s.

    `-c copy` copies every video packet unchanged, so no pixel changes and the face
    boxes and cached masks stay valid. Output stays AVI: MP4 and MKV drop three
    frames mid-stream on the 29.9167 fps clips, which shifts every frame after and
    desynchronises the contact-PPG target.

    Resumable and safe to interrupt: a clip is renamed into place only after it
    probes as indexed.
    """
    import polars as pl

    from .aggregation import remux as remux_mod

    rows = pl.read_parquet(manifest)
    subset = rows.filter(pl.col("source") == corpus)
    out_dir = out_dir or remux_mod.REMUX_DIR
    click.echo(f"{subset.height} {corpus} clips -> {out_dir}")

    counts = remux_mod.build(subset, out_dir, force=force, progress=True)
    click.echo(f"\nremuxed {counts['remuxed']}, already indexed {counts['skipped']}, "
               f"failed {counts['failed']}")

    if write_manifest is not None:
        updated = remux_mod.rewrite_manifest(rows, out_dir, source=corpus)
        moved = int((updated["video_path"] != rows["video_path"]).sum())
        updated.write_parquet(write_manifest)
        click.echo(f"wrote {write_manifest} with {moved} paths repointed")


@cli.command()
@click.option(
    "--manifest", type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_MANIFEST, show_default=True,
)
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), default=None,
    help="Where the frames go. Defaults to build/frames_cache.",
)
@click.option("--force", is_flag=True, help="Rebuild clips that are already cached.")
def cache(manifest: Path, cache_dir: Path | None, force: bool) -> None:
    """Decode each clip's face box once, so training stops re-reading the video.

    UBFC-rPPG is uncompressed rawvideo: one 640x480 frame is 921,600 bytes, so a
    160-frame window reads 147 MB to yield a 23 MB face box. Measured, one epoch
    moved 71 GB off disk to produce 3.7 GB of pixels, and the loader rather than
    the GPU set the pace.

    Resumable -- rerunning skips what is already built -- and safe to interrupt,
    because a clip's sidecar is written only after its frames are complete.

    Run it after `clips` and before `train`.
    """
    import polars as pl

    from .model import framecache

    rows = pl.read_parquet(manifest)
    out_dir = cache_dir or framecache.CACHE_DIR
    click.echo(f"{rows.height} clips -> {out_dir}")
    click.echo(f"  {framecache.total_bytes(rows) / 1e9:.1f} GB when complete\n")

    counts = framecache.build(rows, out_dir, force=force, progress=True)
    click.echo(f"\nbuilt {counts['built']}, already present {counts['skipped']}, "
               f"no face box {counts['no_box']}, failed {counts['failed']}")


@cli.command()
@click.option(
    "--per-source", type=int, default=4, show_default=True,
    help="Distinct subjects to render from each dataset.",
)
@click.option(
    "--seconds", type=float, default=5.0, show_default=True,
    help="Seconds of each clip to process.",
)
def samples(per_source: int, seconds: float) -> None:
    """Render contact sheets for visual review.

    Each sheet shows three rows -- original crop, skin mask, brightness-normalised
    output -- across four frames. The normalised faces are flat mid-grey by
    construction: each frame's mean skin luma is subtracted and re-centred to 128.

    Also writes the frames and the mean_Y trace as .npy, so the numbers can be read
    as well as the picture. Output lands in build/samples/.
    """
    from .aggregation import make_samples

    raise SystemExit(make_samples.main(per_source=per_source, seconds=seconds))


@cli.command()
@click.option(
    "--manifest", type=click.Path(exists=True, path_type=Path), default=DEFAULT_MANIFEST,
    show_default=True, help="Clip manifest to summarise.",
)
def info(manifest: Path) -> None:
    """Show coverage, label spread and the train/dev/test split.

    The split reported is the one the manifest carries, which is the partition
    `train` uses. A manifest with no `split` column gets one derived here, and the
    line says so.

    Either way the split is grouped by subject, never by row: windows from one
    recording are near-duplicates of each other, so a row-level split would score a
    model on people it trained on.
    """
    import polars as pl

    df = pl.read_parquet(manifest)
    hours = df["duration_s"].sum() / 3600

    click.echo(f"{df.height} clips, {df['subject_id'].n_unique()} subjects, {hours:.1f} h\n")
    click.echo(df.group_by("source").agg(
        pl.len().alias("clips"),
        pl.col("subject_id").n_unique().alias("subjects"),
        (pl.col("duration_s").sum() / 3600).round(2).alias("hours"),
        pl.col("fps").mean().round(1).alias("mean_fps"),
        (pl.col("skin_frac").mean() * 100).round(1).alias("skin_pct"),
    ).sort("source"))

    click.echo("\nlabels:")
    click.echo(df.select("hr_bpm").describe())

    if "split" in df.columns:
        click.echo(f"\nsplit, as {manifest} carries it:")
        parts = {name: df.filter(pl.col("split") == name)
                 for name in ("train", "dev", "test")}
    else:
        from .model.dataset import split_manifest

        click.echo(f"\nsplit derived here: {manifest} carries no split column")
        parts = split_manifest(df)

    for name, part in parts.items():
        pct = 100 * part.height / df.height if df.height else 0
        click.echo(f"  {name:5} {part.height:5} clips  "
                   f"{part['subject_id'].n_unique():4} subjects  {pct:5.1f}%")


@cli.command()
@click.option("--frames", type=int, default=DEFAULT_FRAMES, show_default=True)
@click.option("--batch", type=int, default=4, show_default=True)
@click.option("--steps", type=int, default=5, show_default=True,
              help="Forward/backward/optimiser steps to time.")
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--manifest", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_MANIFEST, show_default=True)
def check(frames: int, batch: int, steps: int, workers: int, manifest: Path) -> None:
    """Shapes, the published cost budget, and throughput on this card.

    Reports the parameter count and MACs per frame against CFMamba Table 4's 0.91M
    and 80.82M. A model outside those numbers is a different model from the paper's,
    whatever it scores.

    Requires CUDA: Mamba-3's scan kernel has no CPU path.
    """
    import time

    import torch
    from torch.utils.data import DataLoader

    from .model.cfmamba import CFMambaPhys
    from .model.dataset import (
        TARGET_FPS,
        WindowDataset,
        expand_to_segments,
        load_manifest,
    )
    from .model.losses import composite_loss

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA required: Mamba-3's scan kernel has no CPU path.")

    model = CFMambaPhys(n_frames=frames).cuda().train()
    click.echo(model.describe())
    published = 0.91e6
    error = (model.parameter_count() - published) / published
    click.echo(f"  vs published 0.91M: {100 * error:+.1f}%")

    clips_table = load_manifest(manifest)
    segments = expand_to_segments(clips_table, frames, TARGET_FPS)
    loader = DataLoader(
        WindowDataset(segments, n_frames=frames, train=True),
        batch_size=batch, shuffle=True, num_workers=workers,
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    click.echo(f"\n{segments.height} segments from {clips_table.height} clips")

    torch.cuda.reset_peak_memory_stats()
    started, done = time.time(), 0
    for item in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(item["frames"].cuda(non_blocking=True), item["skin"].cuda())
        loss, parts = composite_loss(predicted, item["wave"].cuda())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        done += 1
        click.echo(f"  step {done}  out {tuple(predicted.shape)}  "
                   f"loss {parts['loss']:.4f} (time {parts['time']:.3f} "
                   f"freq {parts['freq']:.3f})")
        if done >= steps:
            break
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated()
    click.echo(f"\n{done} steps in {elapsed:.1f}s -> {elapsed / max(done, 1):.2f}s/step, "
               f"{done * batch / elapsed:.1f} clip/s")
    click.echo(f"peak VRAM {peak / 1e9:.2f} GB  "
               f"({peak / 1e6 / (batch * frames):.2f} MB/frame vs Table 4's 2.85)")


@cli.command()
@click.option("--steps", type=int, default=300, show_default=True,
              help="Optimiser steps. This is a control, not a training run.")
@click.option("--frames", type=int, default=DEFAULT_FRAMES, show_default=True)
def sanity(steps: int, frames: int) -> None:
    """Fit a synthetic pulse, as a control on the architecture and the loss.

    Trains on synthetic clips: a uniform patch whose brightness is modulated at a
    known rate, with the rate varying between clips. No face, no motion, no
    compression. The signal is present by construction, so a model that cannot fit
    this has a fault in the architecture or the loss rather than in the data.
    """
    import math

    import torch

    from .model.cfmamba import CFMambaPhys
    from .model.dataset import TARGET_FPS
    from .model.losses import composite_loss
    from .model.postprocess import heart_rate
    from .model.train import TrainConfig, _optimiser

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA required: Mamba-3's scan kernel has no CPU path.")

    torch.manual_seed(0)
    model = CFMambaPhys(n_frames=frames).cuda().train()
    # The training optimiser and schedule, not a substitute: a control run under a
    # different configuration would pass with a broken learning rate or a
    # mis-scaled warmup.
    config = TrainConfig(epochs=1)
    optimiser, scheduler = _optimiser(model, config, steps_per_epoch=steps)
    generator = torch.Generator(device="cpu").manual_seed(1)

    def batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        """A uniform patch modulated at a random cardiac rate, strongest in green."""
        bpm = 50.0 + 100.0 * torch.rand(n, generator=generator)
        t = torch.arange(frames).float() / TARGET_FPS
        wave = torch.sin(2 * math.pi * (bpm[:, None] / 60.0) * t[None, :])
        frames_out = torch.empty(n, frames, 3, 64, 64)
        for channel, (base, gain) in enumerate(((0.55, 0.3), (0.45, 1.0), (0.40, 0.2))):
            frames_out[:, :, channel] = (
                base + 0.01 * gain * wave[:, :, None, None]
            )
        return frames_out.cuda(), wave.cuda()

    for step in range(steps):
        frames_in, target = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(frames_in, torch.ones(frames_in.shape[0], 64, 64).cuda())
        loss, parts = composite_loss(predicted, target)
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        scheduler.step()
        if (step + 1) % 25 == 0:
            click.echo(f"  step {step + 1:4d}  lr {optimiser.param_groups[0]['lr']:.2e}  "
                       f"loss {parts['loss']:.4f}  time {parts['time']:.3f}  "
                       f"freq {parts['freq']:.3f}")

    model.eval()
    errors = []
    with torch.no_grad():
        for _ in range(8):
            frames_in, target = batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(frames_in, torch.ones(frames_in.shape[0], 64, 64).cuda())
            for i in range(len(predicted)):
                got = heart_rate(predicted[i].float().cpu().numpy(), TARGET_FPS)
                want = heart_rate(target[i].cpu().numpy(), TARGET_FPS)
                errors.append(abs(got - want))
    mae = sum(errors) / len(errors)
    click.echo(f"\nsynthetic MAE {mae:.2f} bpm over {len(errors)} clips")
    click.echo("above the ~2 bpm this control is expected to reach" if mae > 2.0
               else "within the ~2 bpm this control is expected to reach")


@cli.command()
@click.option("--manifest", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_MANIFEST, show_default=True)
@click.option("--frames", type=int, default=DEFAULT_FRAMES, show_default=True)
@click.option("--workers", type=int, default=DEFAULT_WORKERS, show_default=True)
@click.option("--methods", default="pos,chrom", show_default=True)
@click.option("--split", type=click.Choice(["train", "dev", "test", "all"]),
              default="test", show_default=True)
def baseline(manifest: Path, frames: int, workers: int, methods: str, split: str) -> None:
    """POS and CHROM on the same windows the model is scored on.

    Both publish ~4.06-4.08 bpm MAE on UBFC-rPPG in both source papers' tables, and
    neither needs training, so they are the classical floor a learned model is read
    against.

    Requires the vendored tools/rPPG-Toolbox, which supplies both implementations.
    """
    import polars as pl
    from torch.utils.data import DataLoader

    from .model.baselines import run as run_baselines
    from .model.dataset import (
        TARGET_FPS,
        WindowDataset,
        expand_to_segments,
        load_manifest,
    )
    from .model.evaluate import format_metrics, summarise

    full = load_manifest(manifest)
    if split != "all" and "split" not in full.columns:
        raise SystemExit(
            f"{manifest} has no split column, so --split {split} has nothing to "
            "select. Build one with `src.cli combine`, or pass --split all."
        )
    part = full if split == "all" else full.filter(pl.col("split") == split)
    segments = expand_to_segments(part, frames, TARGET_FPS)
    click.echo(f"{part.height} clips, {segments.height} segments of {frames} frames")
    loader = DataLoader(
        WindowDataset(segments, n_frames=frames, train=False),
        batch_size=1, num_workers=workers, prefetch_factor=2 if workers else None,
    )
    chosen = tuple(m.strip() for m in methods.split(",") if m.strip())
    for name, rows in run_baselines(loader, methods=chosen, fps=TARGET_FPS).items():
        click.echo(format_metrics(name.upper(), summarise(rows)))


@cli.command()
@click.option("--epochs", type=int, default=50, show_default=True)
@click.option("--batch", type=int, default=4, show_default=True)
@click.option("--frames", type=int, default=DEFAULT_FRAMES, show_default=True)
@click.option("--workers", type=int, default=DEFAULT_WORKERS, show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True,
              help="AdamW's own default. Warmup then cosine decay to 1% of it.")
@click.option("--weight-decay", type=float, default=0.05, show_default=True,
              help="Applied to weight matrices only. Norms, biases, Mamba's "
                   "A_log/D and the Gaussian band's f_c/b_w are exempt.")
@click.option("--warmup-frac", type=float, default=0.05, show_default=True,
              help="Fraction of total steps spent warming lr up from zero.")
@click.option("--alpha", type=float, default=0.8, show_default=True,
              help="Weight on the temporal (negative Pearson) term, Eq. 19.")
@click.option("--beta", type=float, default=1.0, show_default=True,
              help="Weight on the frequency cross-entropy term, Eq. 19.")
@click.option("--sources", default=None,
              help="Comma-separated corpora to use, e.g. 'ubfc'. Default: every "
                   "source in the manifest.")
@click.option("--stride", type=int, default=None,
              help="Segment stride in frames. Default: non-overlapping. Raise it "
                   "to subsample a large corpus.")
@click.option("--log-every", type=int, default=25, show_default=True,
              help="Steps between progress lines.")
@click.option("--resume", is_flag=True, default=False,
              help="Continue from <run-dir>/last.pt: weights, optimiser moments, "
                   "and the learning-rate schedule's position. The schedule inputs "
                   "(epochs, batch, frames) must match the checkpoint.")
@click.option("--profile", is_flag=True, default=False,
              help="Synchronise inside the training loop so the reported compute "
                   "time is compute rather than queue submission. Slows the run.")
@click.option("--manifest", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_MANIFEST, show_default=True)
@click.option("--run-dir", type=click.Path(path_type=Path),
              default=DEFAULT_RUN_DIR, show_default=True,
              help="Where last.pt, best.pt, final.pt and history.json go.")
def train(epochs: int, batch: int, frames: int, workers: int, seed: int, lr: float,
          weight_decay: float, warmup_frac: float, alpha: float, beta: float,
          sources: str | None, stride: int | None, log_every: int, resume: bool,
          profile: bool, manifest: Path, run_dir: Path) -> None:
    """Fit CFMamba-Phys and score the last epoch.

    Takes no arguments for the standard run: the pooled manifest, 300-frame
    windows, batch 4, 50 epochs.

    The split is the one the manifest carries, which `combine` writes at 90/3/7
    grouped by subject and stratified by source. A manifest with no split column
    gets a subject-grouped 85/10/5 derived at load time, and the run says which it
    used. Ratios are in segments rather than clips because clip length varies
    four-fold here and segments are what the model sees.

    Dev is scored each epoch for the trajectory, on a fixed subsample; test is
    scored once at the end. The reported result is the last epoch, as both source
    papers report, so the epoch budget is chosen in advance rather than tuned on
    the result. `<run-dir>/best.pt` is written whenever an epoch sets a new lowest
    dev loss, because last.pt cannot be rolled back to an earlier epoch. It is a
    second artefact, scored only by pointing --model at it.

    Results are broken out per source. MCD-rPPG is ~98% of the test segments, so an
    aggregate over the pooled split is largely a measurement of MCD.

    Ablations (stem variant, PGA, CAM, FFN kind, PTS mode, scan direction, frame
    normalisation, augmentation) are fields on `TrainConfig` rather than flags
    here. Set them from Python.
    """
    from .model.train import TrainConfig
    from .model.train import train as run_training

    config = TrainConfig(
        epochs=epochs, batch_size=batch, n_frames=frames, workers=workers,
        seed=seed, lr=lr, weight_decay=weight_decay, warmup_frac=warmup_frac,
        alpha=alpha, beta=beta, log_every=log_every,
        sources=tuple(x.strip() for x in (sources or "").split(",") if x.strip()),
        stride_frames=stride,
        profile=profile, resume=resume, out_dir=run_dir,
    )
    run_training(config, manifest)



@cli.command()
@click.option("--video", required=True, type=click.Path(exists=True, path_type=Path),
              help="Any face video. Needs no manifest entry -- the face box and "
                   "skin mask are built inline.")
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="Checkpoint to run. Defaults to the most recently written "
                   "build/runs/*/final.pt or last.pt.")
@click.option("--start", type=float, default=0.0, show_default=True,
              help="Seconds into the recording to begin.")
@click.option("--seconds", type=float, default=30.0, show_default=True,
              help="Seconds of recording to run. 0 means to the end.")
@click.option("--batch", type=int, default=1, show_default=True,
              help="Windows per forward pass. 1 by default: a 300-frame window is "
                   "the same activation footprint a training step has, and a short "
                   "clip is a handful of windows, so there is nothing to gain and a "
                   "shared card to run out of.")
@click.option("--workers", type=int, default=DEFAULT_WORKERS, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="PNG to write. Defaults to build/predict/<directory>__<stem>.png.")
def predict(video: Path, model_path: Path | None, start: float, seconds: float,
            batch: int, workers: int, out: Path | None) -> None:
    """Run one video through a trained model and plot the pulse it predicts.

    Window length is read from the checkpoint rather than passed: the model's
    temporal axis is fixed at the length it was built with. A 300-frame checkpoint
    reads 10 s at a time, and a longer clip becomes consecutive non-overlapping
    windows laid end to end.

    The y axis has no units. Eq. 19's temporal term is negative Pearson, which is
    invariant to a positive scale factor, so each window is z-scored before the
    stitch and only the timing carries meaning.

    Two rates are reported: the median inter-beat interval of the marked peaks,
    which is what this project's tables quote, and the dominant cardiac-band
    spectral peak as a cross-check. The two disagree when the window holds more
    than one rhythm. `src.cli readout` scores both against contact PPG.
    """
    import numpy as np
    import torch

    from .model import predict as predict_mod
    from .model import predict_plot
    from .model.dataset import TARGET_FPS

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA required: Mamba-3's scan kernel has no CPU path.")

    checkpoint = model_path or predict_mod.latest_checkpoint()
    model, config, state = predict_mod.load_model(checkpoint)
    span = config.n_frames / TARGET_FPS
    click.echo(f"{checkpoint}  epoch {state.get('epoch', '?')}  "
               f"{config.n_frames} frames per window ({span:.1f}s at {TARGET_FPS:.0f} fps)")

    try:
        dataset, segments, row = predict_mod.prepare(
            video, config, start_s=start, seconds=seconds or None
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"{video.name}: {row['fps']:.2f} fps, {row['duration_s']:.1f}s, "
               f"skin {row['skin_frac'] * 100:.0f}% -> {segments.height} windows")

    try:
        windows, truth = predict_mod.run_windows(
            model, dataset, batch_size=batch, workers=workers
        )
    except torch.OutOfMemoryError as error:
        # --seconds is not the handle here. Windows run one batch at a time, so peak
        # memory is set by the checkpoint's window length and --batch, not by how
        # much of the recording is being read.
        free, total = torch.cuda.mem_get_info()
        raise click.ClickException(
            f"out of GPU memory at batch {batch} on a {config.n_frames}-frame "
            f"window: {free / 1e9:.1f} of {total / 1e9:.1f} GB free. Lower --batch, "
            f"or free the card -- a shorter --seconds will not help, because peak "
            f"memory is per window.\n  {error}"
        ) from error
    result = predict_mod.analyse(windows, TARGET_FPS, truth=truth)

    per_window = result["per_window_bpm"]
    finite = per_window[np.isfinite(per_window)]
    click.echo(f"\n  {result['bpm_beats']:.1f} bpm  median inter-beat interval "
               f"over {result['seconds']:.1f}s, {len(result['peaks'])} beats")
    click.echo(f"  {result['bpm_fft']:.1f} bpm  dominant cardiac-band peak, "
               f"the spectral cross-check")
    if len(finite):
        click.echo(f"  per window: {finite.min():.1f}-{finite.max():.1f} bpm "
                   f"over {len(finite)} windows")
    if "bpm_true" in result:
        click.echo(f"\n  {result['bpm_true']:.1f} bpm  contact PPG over the same "
                   f"windows  -> {result['bpm_beats'] - result['bpm_true']:+.1f} bpm, "
                   f"MACC {result['macc']:.3f}")

    out = out or BUILD_ROOT / "predict" / f"{predict_mod.clip_name(video)}.png"
    predict_plot.plot(
        result["trace"], result["peaks"], fps=TARGET_FPS, start_s=start,
        bpm_fft=result["bpm_fft"], bpm_beats=result["bpm_beats"],
        seams=result["seams"], title=video.name,
        subtitle=f"{checkpoint}  ·  {config.n_frames}-frame windows",
        out=out, truth=result.get("truth"), bpm_true=result.get("bpm_true"),
    )
    # The trace as well as the picture, so the numbers can be read as well as the
    # plot. Same reason `samples` writes its .npy.
    wave_path = out.with_name(f"{out.stem}_wave.npy")
    np.save(wave_path, result["trace"])
    click.echo(f"\nwrote {out} and {wave_path}")


@cli.command()
@click.option("--manifest", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_MANIFEST, show_default=True)
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="Checkpoint to run. Defaults to the most recently written "
                   "build/runs/*/final.pt or last.pt.")
@click.option("--split", type=click.Choice(["train", "dev", "test"]),
              default="test", show_default=True)
@click.option("--stride", type=int, default=None,
              help="Segment stride in frames. Default: non-overlapping. Raise it "
                   "to subsample; MCD is ~98% of the split either way.")
@click.option("--limit", type=int, default=None,
              help="Score only the first N segments. The sweep compares readouts "
                   "against each other, so it does not need all of them.")
@click.option("--batch", type=int, default=4, show_default=True)
@click.option("--workers", type=int, default=DEFAULT_WORKERS, show_default=True)
@click.option("--dump", type=click.Path(path_type=Path), default=None,
              help="Where the forward pass is cached. "
                   "Defaults to build/readout_<split>.npz.")
@click.option("--force", is_flag=True, default=False,
              help="Re-run the forward pass even though the dump exists.")
def readout(manifest: Path, model_path: Path | None, split: str, stride: int | None,
            limit: int | None, batch: int, workers: int, dump: Path | None,
            force: bool) -> None:
    """Score every heart-rate readout against contact PPG, on one split.

    The readouts disagree by up to 15 bpm on the clips inspected by hand, in both
    directions. This scores them over a labelled split instead.

    The forward pass is cached, so adding a variant and re-running costs no GPU
    time. The truth rate is read with the same variant as the prediction; reading
    them differently would report the difference between two methods as model
    error.
    """
    import numpy as np
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    from .model import readout as readout_mod
    from .model.dataset import TARGET_FPS, WindowDataset
    from .model.predict import load_model
    from .model.train import build_splits

    dump = dump or BUILD_ROOT / f"readout_{split}.npz"

    if force or not dump.exists():
        if not torch.cuda.is_available():
            raise click.ClickException(
                "CUDA required for the forward pass: Mamba-3's scan kernel has no "
                f"CPU path. An existing {dump} would be scored without one."
            )
        from .model.predict import latest_checkpoint

        checkpoint = model_path or latest_checkpoint()
        model, config, state = load_model(checkpoint)
        if stride is not None:
            config.stride_frames = stride
        click.echo(f"{checkpoint}  epoch {state.get('epoch', '?')}  "
                   f"{config.n_frames} frames per window")

        segments = build_splits(config, manifest)[split]
        if limit:
            segments = segments.head(limit)
        click.echo(f"{split}: {segments.height} segments")

        loader = DataLoader(
            WindowDataset(segments, n_frames=config.n_frames, train=False,
                          resolution=config.resolution,
                          frame_norm=config.frame_norm,
                          apply_skin_mask=config.apply_skin_mask,
                          return_waveform=True),
            batch_size=batch, shuffle=False, num_workers=workers,
            prefetch_factor=2 if workers else None,
        )
        predicted_windows: list[np.ndarray] = []
        truth_windows: list[np.ndarray] = []
        sources: list[str] = []
        with torch.no_grad():
            for step, item in enumerate(loader, start=1):
                # train=False fixes the resampling factor at 1.0. A stretched
                # window would make every bpm below wrong by that factor, and the
                # sweep would rank the readouts on an artefact of the loader.
                scale = item["fps_scale"].numpy()
                if not np.allclose(scale, 1.0):
                    raise click.ClickException(
                        f"expected fps_scale 1.0 from an evaluation loader, got {scale}"
                    )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(item["frames"].cuda(non_blocking=True),
                                item["skin"].cuda(non_blocking=True))
                predicted_windows.append(out.float().cpu().numpy())
                truth_windows.append(item["wave"].numpy())
                sources.extend(c.split("/")[0] for c in item["clip_id"])
                if step % 25 == 0:
                    click.echo(f"  {len(sources)} windows")
        readout_mod.save_dump(
            dump, np.concatenate(predicted_windows), np.concatenate(truth_windows),
            sources,
        )
        click.echo(f"wrote {dump}")

    predicted, truth, sources = readout_mod.load_dump(dump)
    click.echo(f"\n{len(predicted)} windows from {dump}")
    table = readout_mod.score(predicted, truth, sources, TARGET_FPS)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        click.echo(table)


if __name__ == "__main__":
    cli()
