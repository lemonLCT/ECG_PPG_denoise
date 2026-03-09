"""训练入口骨架：仅保留流程接口，不实现具体训练逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 src 加入路径，原理是允许在未安装 editable 包时直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    """训练入口。"""
    # TODO: 组装 datamodule/model/loss/trainer 并启动训练
    raise NotImplementedError("TODO: 实现训练入口逻辑")


if __name__ == "__main__":
    raise SystemExit(main())

