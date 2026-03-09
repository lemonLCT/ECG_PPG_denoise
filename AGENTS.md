# Repository Guidelines

## 项目结构与模块组织
本仓库采用 `src` 布局，核心代码位于 `src/ecg_ppg_denoise/`，按职责拆分为 `config/`、`data/`、`preprocess/`、`models/`、`losses/`、`trainers/`、`evaluators/`、`metrics/`、`utils/`。  
命令行入口在 `scripts/`：
- `run_pipeline.py`：最小流程连通性验证
- `train.py`、`evaluate.py`：训练/评估扩展骨架  
测试代码在 `tests/`，当前覆盖导入、配置结构和烟雾流程。  
`model/Score-based-ECG-Denoising/` 为外部参考实现与权重资源，不作为主包源码目录。

## 构建、测试与开发命令
以下命令均在仓库根目录执行：

- `pip install -e ".[dev]"`：以可编辑模式安装项目及开发依赖。
- `python scripts/run_pipeline.py`：运行最小端到端 smoke 流程。
- `pytest -q`：执行全部测试（配置见 `pyproject.toml`）。
- `python scripts/train.py` / `python scripts/evaluate.py`：当前为骨架入口，未实现完整逻辑。

## 代码风格与命名规范
Python 版本要求 `>=3.10`，统一 4 空格缩进。公开接口应保留类型注解。  
命名约定如下：
- 模块、函数、变量：`snake_case`
- 类、数据类：`PascalCase`
- 常量：`UPPER_SNAKE_CASE`  
优先编写单一职责、可读性高的函数；通过显式配置对象传递依赖；异常处理应提供上下文，禁止静默失败。

## 测试规范
测试框架为 `pytest`，`pyproject.toml` 已设置 `testpaths = ["tests"]` 与 `pythonpath = ["src"]`。  
测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。  
新增功能必须补充至少一个单元测试；涉及流程编排变更时，需同步补充或更新类似 `tests/test_smoke_pipeline.py` 的烟雾测试。

## 提交与合并请求规范
当前目录快照不含 `.git`，无法读取历史提交格式。建议统一采用 Conventional Commits，例如：`feat(config): 增加严格配置校验`。  
PR 至少包含：
- 变更目的与范围
- 关联任务或问题链接
- 测试证据（执行命令与结果）
- 配置/数据假设及潜在破坏性变更说明
