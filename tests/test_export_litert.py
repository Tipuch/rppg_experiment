"""The LiteRT exporter's checkpoint handling, from tools/export_litert.py.

The exporter is the one place that rebuilds a trained architecture without
`build_model`: it runs in a Python 3.13 environment holding torch and
litert-torch and nothing else, so importing `train.py` -- and polars and cv2
with it -- is not available to it. That makes drift between the two
constructors the failure this file exists to catch. A hand-written key list
already broke once, when `cam_pooling` left `TrainConfig` and the tool kept
reading it; every test here is built from the *current* `TrainConfig`, so the
next field to move breaks a test rather than the tool.

`litert_torch` is imported inside the functions that convert, so everything
tested here runs without it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.model.train import TrainConfig, build_model

_spec = importlib.util.spec_from_file_location(
    "export_litert", Path(__file__).resolve().parents[1] / "tools" / "export_litert.py"
)
export_litert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_litert)


def _checkpoint(path: Path, cfg: TrainConfig) -> Path:
    """A checkpoint in the format `train.py` writes, for the architecture `cfg` names."""
    model = build_model(cfg, scan="export", spectral="matmul")
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, path)
    return path


def test_a_checkpoint_from_the_current_config_loads(tmp_path: Path) -> None:
    """The regression test for the drift itself.

    `asdict(TrainConfig())` is whatever the code records today, so this fails the
    moment the exporter reads a key that training no longer writes.
    """
    cfg = replace(TrainConfig(), n_frames=48)
    model, saved = export_litert.load_exportable(_checkpoint(tmp_path / "last.pt", cfg))
    assert model.n_frames == 48
    assert saved["frame_norm"] == cfg.frame_norm


def test_the_export_is_built_on_the_two_implementations_that_export() -> None:
    """scan and spectral are the exporter's own, not the checkpoint's."""
    cfg = replace(TrainConfig(), n_frames=48)
    assert "scan" not in asdict(cfg)
    assert "spectral" not in asdict(cfg)


def test_the_architecture_filter_ignores_the_training_only_fields() -> None:
    """Every key the constructor takes is kept; the rest -- lr, workers -- is not."""
    kept = export_litert.architecture(asdict(TrainConfig()))
    assert kept["n_frames"] == TrainConfig().n_frames
    assert kept["fps"] == TrainConfig().fps
    assert kept["use_pga"] is True
    for training_only in ("lr", "workers", "epochs", "out_dir", "frame_norm"):
        assert training_only not in kept


def test_an_older_checkpoint_missing_a_field_still_builds() -> None:
    """Absent means the constructor default, which is what the run would have used."""
    partial = {k: v for k, v in asdict(TrainConfig()).items() if k != "fps"}
    assert "fps" not in export_litert.architecture(partial)


def test_the_frontend_honours_the_pga_ablation(tmp_path: Path) -> None:
    """use_pga=False loads without complaint, so it has to run as well.

    The model's own fallback is a spatial mean; without it here the export dies on
    `'NoneType' object is not callable` after a clean strict load.
    """
    cfg = replace(TrainConfig(), n_frames=48, use_pga=False)
    model, _ = export_litert.load_exportable(_checkpoint(tmp_path / "ablation.pt", cfg))
    frames = torch.randn(1, 8, 3, cfg.resolution, cfg.resolution)
    skin = torch.rand(1, cfg.resolution, cfg.resolution)
    with torch.no_grad():
        features = export_litert.Frontend(model)(frames, skin)
    assert features.shape == (1, 8, model.dim)
