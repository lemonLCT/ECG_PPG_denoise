"""评估入口骨架：仅保留流程接口，不实现具体评估逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 src 加入路径，原理是允许在未安装 editable 包时直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    """评估入口。"""
    # TODO: 加载权重并执行离线评估
    raise NotImplementedError("TODO: 实现评估入口逻辑")


if __name__ == "__main__":
    raise SystemExit(main())

