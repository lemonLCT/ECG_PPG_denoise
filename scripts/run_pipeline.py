"""最小可运行入口：仅验证工程流程连通，不执行实际训练。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将 src 加入路径，原理是允许在未安装 editable 包时直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg_denoise.config.loader import load_experiment_config
from ecg_ppg_denoise.trainers.runner import ExperimentRunner
from ecg_ppg_denoise.utils.logging import build_logger


def main() -> int:
    """命令行入口函数。"""
    parser = argparse.ArgumentParser(description="运行 ECG+PPG 去噪骨架流程")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_experiment_config(config_path=config_path)
    logger = build_logger()
    runner = ExperimentRunner(config=config, logger=logger)
    runner.run_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

