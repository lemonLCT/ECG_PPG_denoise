"""实验运行器。

该模块提供最小可运行入口，原理是先打通“配置-日志-运行器”链路，
确认工程架构可启动，再逐步填充训练细节。
"""

import logging

from ..config.schema import ExperimentConfig
from ..utils.seed import set_global_seed


class ExperimentRunner:
    """组装组件并执行流程连通性检查。"""

    def __init__(self, config: ExperimentConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run_smoke(self) -> None:
        """仅做流程连通检查，不执行训练。"""
        # 设置全局随机种子，保证最小流程具备可复现性。
        set_global_seed(self.config.train.seed)
        self.logger.info("实验名: %s", self.config.runtime.experiment_name)
        self.logger.info("流程连通性检查通过（未执行训练/评估）。")

