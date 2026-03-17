from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("utils", None)

from config import load_config
import torch
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
