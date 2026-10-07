"""Run AH's RR network (src/vitals/rr_mamba.py): the frozen design, its training
rules, and a regression on AH's own epoch-28 checkpoint.

Everything runs on the CPU with the "export" scan, except the kernel parity test
(CUDA only) and the AH regression (skipped when the checkpoint or table is absent).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from src.paths import BUILD_ROOT
from src.vitals import rr_mamba as rm
from src.vitals import rr_table

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Mamba-3 kernel is CUDA-only"
)
AH = BUILD_ROOT / "rr_saved" / "rr_mamba_run_ah_epoch028.pt"
AC = BUILD_ROOT / "rr_saved" / "rr_mamba_run_ac_epoch015.pt"


def _model(**kwargs) -> rm.RRMamba:
    torch.manual_seed(0)
    return rm.RRMamba(16.0, scan="export", **kwargs)


def _x(batch: int = 3) -> torch.Tensor:
    return torch.randn(batch, 2, 900, generator=torch.Generator().manual_seed(1))


def test_shapes_and_the_bias_starting_at_the_train_mean():
    model, x = _model().eval(), _x()
    assert model.stem(x).shape == (3, 225, rm.DIM)
    assert model.pooled(x).shape == (3, 2 * rm.DIM)
    assert model(x).shape == (3,)
    assert model.head[-1].bias.item() == 16.0
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in _model().parameters()
    )


def test_pooling_concatenates_the_stem_and_the_scan_of_its_layer_norm():
    model, x = _model().eval(), _x()
    seen = []
    model.mamba.register_forward_hook(lambda module, args, out: seen.append(args[0]))
    with torch.no_grad():
        s, pooled = model.stem(x), model.pooled(x)
    torch.testing.assert_close(
        seen[0],
        nn.functional.layer_norm(
            s, (rm.DIM,), model.scan_norm.weight, model.scan_norm.bias
        ),
    )
    torch.testing.assert_close(pooled[:, : rm.DIM], s.mean(1))
    # The skip: with the scan's output layer zeroed the stem half is unchanged and
    # the scan half is 0.
    with torch.no_grad():
        model.mamba.out_proj.weight.zero_()
        torch.testing.assert_close(
            model.pooled(x), torch.cat([s.mean(1), torch.zeros(3, rm.DIM)], -1)
        )


def test_run_ac_pooling_is_the_mean_of_the_sum():
    model, x = _model(pool_concat=False).eval(), _x()
    with torch.no_grad():
        s = model.stem(x)
        m = model.mamba(model.scan_norm(s))
        torch.testing.assert_close(model.pooled(x), (s + m).mean(1))
    assert model.head[-1].in_features == rm.DIM


def test_dropout_is_in_the_head_only():
    model, x = _model(), _x()
    assert [m for m in model.modules() if isinstance(m, nn.Dropout)] == [
        model.head_dropout
    ]
    assert model.head_dropout.p == rm.HEAD_DROPOUT == 0.5
    model.train()
    with torch.no_grad():
        torch.testing.assert_close(model.pooled(x), model.pooled(x))
        assert not torch.equal(model(x), model(x))
        model.eval()
        torch.testing.assert_close(model(x), model(x))


def test_huber_loss_with_delta_3():
    assert (
        isinstance(rm.loss_fn, nn.HuberLoss)
        and rm.loss_fn.delta == rm.HUBER_DELTA == 3.0
    )
    errors = torch.tensor([1.0, 5.0])
    assert rm.loss_fn(errors, torch.zeros(2)).item() == pytest.approx(
        (0.5 + 3 * (5 - 1.5)) / 2
    )


def test_optimiser_groups_and_schedule():
    model = _model()
    adamw, scheduler = rm.optimiser(model, steps_per_epoch=10, epochs=10, lr=1.0)
    decayed, exempt = (group["params"] for group in adamw.param_groups)
    assert (
        adamw.param_groups[0]["weight_decay"] == rm.WEIGHT_DECAY
        and adamw.param_groups[1]["weight_decay"] == 0
    )
    assert all(p.ndim >= 2 for p in decayed)
    assert all(
        any(p is q for q in exempt)
        for p in (model.mamba.B_bias, model.mamba.C_bias, model.head[-1].bias)
    )
    assert len(decayed) + len(exempt) == len(list(model.parameters()))
    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        adamw.step()
        scheduler.step()
    warmup = int(100 * rm.WARMUP_FRAC)
    assert rates[0] == pytest.approx(1 / warmup) and rates[warmup - 1] == pytest.approx(
        1.0
    )
    assert rates == sorted(rates[:warmup]) + sorted(rates[warmup:], reverse=True)
    assert rates[-1] == pytest.approx(rm.LR_FLOOR, abs=1e-3)


def _table(path, rows_per_split=(24, 8, 8), seed: int = 0) -> pl.DataFrame:
    """A tiny table in `rr_table`'s layout, and its waves_pp, written to `path`."""
    rng = np.random.default_rng(seed)
    splits = [
        s
        for s, n in zip(("train", "dev", "test"), rows_per_split, strict=True)
        for _ in range(n)
    ]
    n = len(splits)
    wave = rng.normal(size=(n, 900)).astype(np.float32)
    table = pl.DataFrame({
        "clip_id": [f"c{i}" for i in range(n)], "source": ["ab"[i % 2] for i in range(n)],
        "subject_id": [str(i) for i in range(n)], "split": splits, "seen_by_ppg_model": False,
        "segment_start_s": 0.0, "rr_bpm": rng.uniform(6, 40, n),
        "augment": ["shifted" if i % 4 == 3 and s == "train" else "natural" for i, s in enumerate(splits)],
        "repeat": pl.Series([0] * n, dtype=pl.UInt32), "skewness": rr_table.skewness(wave),
    })  # fmt: skip
    table = table.with_columns(
        pl.lit(table["rr_bpm"][:8].mean()).alias(rr_table.NATURAL_MEAN)
    )
    pp = table.select(rm.KEYS).with_columns(
        wave=pl.Series(wave), tach=pl.Series(rng.normal(size=(n, 900)).astype(np.float32)),
        ls=pl.Series(rng.normal(size=(n, 900)).astype(np.float32)),
    )  # fmt: skip
    path.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path / "features.parquet")
    pp.write_parquet(path / rm.PP_FILE)
    return table


def test_join_keeps_copies_and_drops_rows_without_label_or_wave(tmp_path):
    table = _table(tmp_path)
    pp = pl.read_parquet(tmp_path / rm.PP_FILE)
    copied = pl.concat([table, table.head(1).with_columns(repeat=pl.lit(1, pl.UInt32))])
    copied = copied.with_columns(
        rr_bpm=pl.when(pl.col("clip_id") == "c5").then(None).otherwise("rr_bpm")
    )
    rows, report = rm.join(copied, pp.filter(pl.col("clip_id") != "c7"))
    assert rows.height == table.height + 1 - 2
    assert rows.filter(pl.col("clip_id") == "c0").height == 2
    assert report["no_label"].sum() == 1 and report["no_wave"].sum() == 1
    rm.check_waves(rows)
    with pytest.raises(ValueError, match="skewness"):
        rm.check_waves(rows.with_columns(wave=pl.Series(-rows["wave"].to_numpy())))


def test_fit_optimises_the_huber_loss_and_keeps_the_highest_dev_slope(
    tmp_path, monkeypatch
):
    rows, _ = rm.join(_table(tmp_path), pl.read_parquet(tmp_path / rm.PP_FILE))
    calls = []
    huber = rm.loss_fn
    monkeypatch.setattr(rm, "loss_fn", lambda e, y: calls.append(1) or huber(e, y))
    best, history = rm.fit(_model(), rows.filter(pl.col("split") == "train"), rows.filter(pl.col("split") == "dev"),
                           epochs=3, batch=8, lr=1e-3)  # fmt: skip
    assert len(calls) == 3 * 3
    assert best["epoch"] == history["epoch"][int(history["dev_slope"].arg_max())]
    assert best["dev_slope"] == history["dev_slope"].max()


def test_main_writes_its_files_and_the_checkpoint_loads_back(tmp_path):
    _table(tmp_path / "t")
    out = tmp_path / "out"
    scores = rm.main(
        tmp_path / "t" / "features.parquet", out, epochs=2, batch=8, device="cpu"
    )
    names = {p.name for p in out.iterdir()} | {
        p.name for p in (out / "checkpoints").iterdir()
    }
    assert names >= {
        "rr_mamba_rr_bpm.pt", "rr_mamba_rr_bpm_history.parquet", "rr_mamba_metrics.parquet",
        "rr_mamba_test_predictions.parquet", "rr_mamba_rr_bpm_loss.png", "epoch_001.pt", "epoch_002.pt",
    }  # fmt: skip
    assert scores.filter(pl.col("source") == "all")["split"].to_list() == [
        "train",
        "dev",
        "test",
    ]
    model, checkpoint = rm.load(out / "rr_mamba_rr_bpm.pt")
    assert (
        checkpoint["config"]["model"]["scan"] == "export"
        and checkpoint["config"]["train"]["lr"] == rm.LR
    )
    test = pl.read_parquet(out / "rr_mamba_test_predictions.parquet")
    pp = pl.read_parquet(tmp_path / "t" / rm.PP_FILE)
    x = rm._tensors(test.join(pp, on=rm.KEYS, maintain_order="left"), "cpu")[0]
    np.testing.assert_allclose(
        rm.predict(model, x).numpy(), test["predicted"].to_numpy(), atol=1e-5
    )
    with pytest.raises(FileExistsError):
        rm.main(tmp_path / "t" / "features.parquet", out, epochs=1, device="cpu")


def test_load_refuses_another_design(tmp_path):
    model = _model()
    path = tmp_path / "other.pt"
    torch.save(
        {
            "config": {"model": model.config | {"dim": 32}},
            "state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="another design"):
        rm.load(path)


@pytest.mark.skipif(
    not (AH.exists() and AC.exists()), reason="AH and AC checkpoints not on disk"
)
def test_ah_and_ac_checkpoints_load():
    assert rm.load(AH, scan="export")[0].pool_concat
    assert not rm.load(AC, scan="export")[0].pool_concat


@cuda
def test_kernel_and_export_scans_agree():
    torch.manual_seed(0)
    kernel = rm.RRMamba(16.0, scan="kernel").cuda().eval()
    export = rm.RRMamba(16.0, scan="export").cuda().eval()
    export.load_state_dict(kernel.state_dict())
    x = _x(4).cuda()
    with torch.no_grad():
        # The kernel's fp32 tiles round apart from the sequential scan, about 1e-3.
        torch.testing.assert_close(
            kernel.pooled(x), export.pooled(x), rtol=1e-2, atol=2e-3
        )
        torch.testing.assert_close(kernel(x), export(x), rtol=0, atol=1e-2)


def test_cli_defaults_equal_the_module_constants():
    from src.cli import vitals_rr_mamba

    defaults = {p.name: p.default for p in vitals_rr_mamba.params}
    assert defaults == {"table": rm.TABLE, "out": None, "epochs": rm.EPOCHS, "batch": rm.BATCH,
                        "lr": rm.LR, "device": None}  # fmt: skip
    assert (rm.EPOCHS, rm.BATCH, rm.LR) == (50, 32, 2e-5)


# Run AH's epoch-28 estimates (rounded to 0.01, as the run report holds them) for
# the extremes of each split and source: (clip_id, segment_start_s, rr_bpm, p28).
AH_POINTS = [
    ("bidmc_42", 120.0, 16.632346370128097, 19.04),
    ("bidmc_53", 0.0, 27.027027027027028, 15.93),
    ("capnobase_0029_8min", 30.0, 9.605122732123794, 13.40),
    ("capnobase_0038_8min", 360.0, 23.593898951382247, 32.11),
    ("mcd/3066_FullHDwebcam_before", 0.0, 15.0, 17.40),
    ("mcd/7437_USBVideo_before", 120.0, 20.0, 16.11),
    ("bidmc_05", 360.0, 5.196623751838207, 17.31),
    ("bidmc_28", 210.0, 20.731007117759148, 16.14),
    ("capnobase_0028_8min", 180.0, 11.153798488040705, 10.96),
    ("capnobase_0009_8min", 210.0, 22.15384615384603, 30.25),
    ("mcd/1672_FullHDwebcam_before", 0.0, 15.0, 15.88),
    ("mcd/3625_USBVideo_before", 120.0, 20.0, 15.38),
]


@pytest.mark.skipif(not (AH.exists() and (rm.TABLE.parent / rm.PP_FILE).exists()),
                    reason="AH checkpoint or the rr_shuffled table not on disk")  # fmt: skip
def test_ah_epoch_28_reproduces_its_reported_estimates():
    points = pl.DataFrame(AH_POINTS, schema=[*rm.KEYS, "rr_bpm", "p28"], orient="row")
    labels = (
        pl.read_parquet(rm.TABLE)
        .filter(pl.col("repeat") == 0)
        .join(points.select(rm.KEYS), on=rm.KEYS)
    )
    np.testing.assert_allclose(
        points.join(labels, on=rm.KEYS, maintain_order="left")["rr_bpm_right"],
        points["rr_bpm"],
    )
    pp = (
        pl.scan_parquet(rm.TABLE.parent / rm.PP_FILE)
        .join(points.lazy(), on=rm.KEYS)
        .collect()
    )
    pp = points.join(pp.select(*rm.KEYS, *rm.INPUTS), on=rm.KEYS, maintain_order="left")
    # 0.03 /min: the reported values are rounded to 0.01, the export scan differs
    # from the kernel by up to 0.02, and so does the kernel itself from one batch
    # shape to another (12 rows here, 1,024 in the report).
    scan = "kernel" if torch.cuda.is_available() else "export"
    model = rm.load(AH, scan=scan)[0].to("cuda" if scan == "kernel" else "cpu")
    estimate = rm.predict(model, rm._tensors(pp, "cpu")[0]).numpy()
    np.testing.assert_allclose(estimate, points["p28"], atol=0.03)
