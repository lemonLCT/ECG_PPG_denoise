from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("utils", None)

from config import load_config
import torch
import train
from models import DDPM
from trainers import ExperimentRunner
from trainers.runner import TrainEngine
from utils.logging import build_logger


def test_smoke_runner_executes() -> None:
    cfg = deepcopy(load_config())
    cfg["data"]["window_length"] = 64
    cfg["model"]["main_model"]["signal_length"] = 64
    cfg["model"]["diffusion"]["num_steps"] = 8
    cfg["model"]["conditional_model"]["base_channels"] = 32
    cfg["model"]["conditional_model"]["joint_channels"] = 64

    runner = ExperimentRunner(config=cfg, logger=build_logger("test_logger"))
    runner.run_smoke()


def test_train_engine_uses_fixed_stage_masks() -> None:
    cfg = deepcopy(load_config())
    model = DDPM(base_model=None, config=cfg, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ecg_engine = TrainEngine(
        model=model,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        stage="ecg_pretrain",
        use_amp=False,
        grad_clip=1.0,
    )
    ppg_engine = TrainEngine(
        model=model,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        stage="ppg_pretrain",
        use_amp=False,
        grad_clip=1.0,
    )
    joint_engine = TrainEngine(
        model=model,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        stage="joint",
        use_amp=False,
        grad_clip=1.0,
    )

    assert torch.equal(ecg_engine._resolve_stage_mask(), torch.tensor([1.0, 0.0]))
    assert torch.equal(ppg_engine._resolve_stage_mask(), torch.tensor([0.0, 1.0]))
    assert joint_engine._resolve_stage_mask() is None


def test_build_datasets_can_switch_to_bidmc(monkeypatch) -> None:
    cfg = deepcopy(load_config())
    calls: list[tuple[str, int]] = []

    def fake_build_bidmc_train_val_datasets(*, config, split_seed):
        calls.append(("bidmc", split_seed))
        assert config is cfg
        return ["train"], ["val"]

    monkeypatch.setattr(train, "build_bidmc_train_val_datasets", fake_build_bidmc_train_val_datasets)

    args = train.argparse.Namespace(
        use_bidmc_dataset=True,
        use_qt_dataset=False,
        qt_noise_version=1,
    )

    train_ds, val_ds = train.build_datasets(cfg, args)

    assert calls == [("bidmc", int(cfg["bidmc"]["data"]["split_seed"]))]
    assert train_ds == ["train"]
    assert val_ds == ["val"]
