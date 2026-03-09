"""最小流程测试。

原理：调用运行器的 smoke 方法，验证“配置-日志-运行器”链路可执行。
"""

from ecg_ppg_denoise.config.loader import load_experiment_config
from ecg_ppg_denoise.trainers.runner import ExperimentRunner
from ecg_ppg_denoise.utils.logging import build_logger


def test_smoke_runner_executes() -> None:
    """最小流程应不抛异常。"""
    cfg = load_experiment_config()
    runner = ExperimentRunner(config=cfg, logger=build_logger("test_logger"))
    runner.run_smoke()

