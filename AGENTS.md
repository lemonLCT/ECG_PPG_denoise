# AGENTS.md

## 1) 全局强约束（必须遵守）
- 全程使用简体中文进行沟通、注释与文档输出。
- 只要任务涉及第三方依赖库或包（新增、升级、API 用法、兼容性判断），必须先查询 Context7 MCP：先 `resolve-library-id`，再 `query-docs`，并基于官方文档给出实现或结论。
- 禁止凭记忆臆断第三方库 API；若 Context7 无结果，需明确说明并给出保守替代方案。

## 2) 项目目标与边界
- 项目目标：构建 ECG/PPG 去噪实验工程，支持数据准备、训练、评估、复现实验与结果沉淀。
- 当前仓库以工程骨架为主，`scripts/train.py` 与 `scripts/evaluate.py` 仍是入口骨架。
- `model/Score-based-ECG-Denoising/` 为外部参考实现与资源区，不作为主工程业务源码目录。

## 3) 当前目录职责（按现状）
- `scripts/`：命令行入口（`run_pipeline.py`、`train.py`、`evaluate.py`）。
- `tests/`：基础测试（导入、配置、smoke）。
- `Data_Preparation/`：数据预处理脚本（QTDB/NSTDB）。
- `model/Score-based-ECG-Denoising/`：外部参考代码、数据与权重（默认只读使用）。
- `requirements.in` / `requirements-dev.in`：依赖输入清单（建议用 pip-tools 锁定）。
- `pyproject.toml`：项目元信息与 pytest 配置。

## 4) 目标工程结构（面向大型复杂项目演进）
- 推荐逐步落地 `src/ecg_ppg_denoise/`，并按职责拆分：
  - `config/`：配置定义、加载与校验。
  - `data/`：数据集、采样器、dataloader。
  - `preprocess/`：信号预处理与增强。
  - `models/`：模型结构与组件。
  - `losses/`：损失函数。
  - `trainers/`：训练流程与调度。
  - `evaluators/`：评估流程。
  - `metrics/`：评价指标。
  - `utils/`：日志、随机种子、IO、通用工具。
- 新增代码优先放入 `src/`，避免继续扩散到根目录脚本式实现。

## 5) 环境与依赖管理
- Python 版本：`>=3.10`。
- 开发安装：`pip install -e ".[dev]"`。
- 依赖管理原则：
  - 运行依赖维护在 `requirements.in`。
  - 开发/测试依赖维护在 `requirements-dev.in`。
  - 使用 `pip-compile` 生成锁定文件（如 `requirements.txt`、`requirements-dev.txt`）。
- 对 CUDA/PyTorch 等硬件相关依赖，必须在文档中写明版本矩阵（Python/CUDA/cuDNN/PyTorch）。

## 6) 数据治理规范
- 数据分层建议：`data/raw`、`data/processed`、`data/cache`、`artifacts/`。
- 原始数据不可在代码中硬编码绝对路径，统一走配置。
- 任何训练/评估数据都应记录：来源、版本、采样率、切分策略、预处理参数。
- 大体积数据与权重默认不纳入 Git（除非明确要求）；通过校验和（SHA256）保证可追溯。

## 7) 配置系统规范
- 采用“默认配置 + 环境覆盖 + CLI 覆盖”的分层策略。
- 配置对象必须可校验（类型、范围、必填项）；校验失败立即报错，禁止静默回退。
- 关键配置（随机种子、设备、batch size、学习率、数据路径、输出路径）必须显式声明。

## 8) 训练/评估/推理入口约定
- 统一入口参数风格（至少包含）：`--config`、`--seed`、`--device`、`--output-dir`。
- 训练输出至少包含：checkpoint、训练日志、验证指标、最终配置快照。
- 评估输出至少包含：样本级结果、聚合指标、可复现实验信息（模型版本 + 数据版本）。
- `scripts/run_pipeline.py` 用于最小连通性验证，禁止在该入口堆叠复杂训练逻辑。

## 9) 分布式、性能与可复现
- 固定随机种子并记录到实验元数据（Python/NumPy/框架级种子）。
- 明确混合精度策略（FP32/AMP/bfloat16）与梯度缩放策略。
- 大规模训练需支持断点续训（optimizer/scheduler/scaler 状态完整保存）。
- 需要多卡训练时，约定 DDP/FSDP 启动方式和日志聚合规则。

## 10) 代码风格与架构约束
- 4 空格缩进，公共接口保留类型注解。
- 命名规范：`snake_case`（模块/函数/变量）、`PascalCase`（类）、`UPPER_SNAKE_CASE`（常量）。
- 函数保持单一职责，避免超长函数与隐式全局状态。
- 异常处理必须带上下文信息；禁止空 `except` 或吞错。
- 日志需结构化且可检索，禁止仅依赖 `print` 进行流程追踪。

## 11) 测试策略与质量门禁
- 测试框架：`pytest`，测试文件命名 `test_*.py`。
- 每个新增功能至少补充一个单元测试；流程变更需补充/更新 smoke 测试。
- 建议分层：
  - 单元测试：纯函数/模块行为。
  - 集成测试：数据-模型-训练器协同。
  - 烟雾测试：端到端最小流程可运行。
- 提交前最低要求：`pytest -q` 通过。

## 12) 实验追踪与产物管理
- 每次实验必须记录：实验 ID、时间、代码版本（Git SHA）、配置快照、数据版本、关键指标。
- checkpoint 命名建议：`{model}_{dataset}_{noise}_{epoch}_{metric}.ckpt`。
- 区分 `latest` 与 `best`，并定义 best 的判定指标与方向（max/min）。
- 推理结果、图表和报告应放在统一产物目录，避免散落在源码目录。

## 13) 提交与评审规范
- 提交信息建议采用 Conventional Commits（如 `feat: ...`、`fix: ...`）。
- PR 必填项：
  - 变更目的与范围。
  - 关键设计与权衡。
  - 测试命令与结果。
  - 性能影响与潜在破坏性变更。
  - 数据/配置假设。
- 严禁将无关重构与功能改动混在同一 PR。

## 14) Agent 执行规则（面向自动化协作）
- 开始任务先阅读：`AGENTS.md`、`pyproject.toml`、相关 `scripts/` 与 `tests/`。
- 涉及第三方库时先查 Context7，再编码。
- 优先做最小可验证改动，并同步更新必要测试与文档。
- 不允许执行破坏性操作（如删除大批数据、覆盖外部参考资源）除非用户明确授权。
- 完成后最少回归：
  - 受影响测试。
  - 至少一次入口脚本或 smoke 验证。
  - 变更摘要（改了什么、为什么、如何验证）。
