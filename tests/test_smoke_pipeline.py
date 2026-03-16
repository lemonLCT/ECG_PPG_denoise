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
from trainers import ExperimentRunner
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
