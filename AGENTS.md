# AGENTS.md

## 1) 全局强约束（必须遵守）
- 全程使用简体中文进行沟通、注释与文档输出。
- 只要任务涉及第三方依赖库或包（新增、升级、API 用法、兼容性判断），必须先查询 Context7 MCP：先 `resolve-library-id`，再 `query-docs`，并基于官方文档给出实现或结论。
- 禁止凭记忆臆断第三方库 API；若 Context7 无结果，需明确说明并给出保守替代方案。
- 新增代码优先放入 `src/`，避免继续扩散到根目录脚本式实现。

## 2) 项目目标与边界
- 项目目标：构建 ECG/PPG 去噪实验工程，支持数据准备、训练、评估、复现实验与结果沉淀。
- 当前仓库主入口为根目录 `train.py`、`infer.py`、`evaluate.py`。
- `model/Score-based-ECG-Denoising/` 为外部参考实现与资源区，不作为主工程业务源码目录。

## 3) 当前目录职责与功能说明
- 推荐逐步落地并按职责拆分，同时如果有增加文件，请详细在本部分增加对应文件的功能说明和文件夹的职能。
- `tests/`：基础测试，所有的pytest有关的测试脚本，或者以test_开头的文件都存在这里。
- `data/`：项目级数据脚本与数据查看工具目录，可放置独立运行的数据检查脚本。
- `model/`：外部参考代码、数据与权重（默认只读使用）。
- `requirements.in` / `requirements-dev.in`：依赖输入清单（建议用 pip-tools 锁定）。
- `pyproject.toml`：项目元信息与 pytest 配置。
- `checkpoint`:用于存储训练模型文件，包含.pth等，禁止被git追踪并上传。
- `src/`:
  - `config/`：配置定义、加载与校验。
  - `data/`：数据集、采样器、dataloader。
  - `Data_Preparation/`：QT数据集的预处理脚本（QTDB/NSTDB），不准修改。
  - `models/`：当前默认模型结构与组件目录，训练/推理/评估统一依赖这里的模型文件。
  - `losses/`：损失函数。
  - `trainers/`：模型训练流程与调度。
  - `metrics/`：计算评价指标的函数和类的存储目录。
  - `utils/`：日志、随机种子、IO、通用工具。
- `train.py`：模型训练入口，用于命令行开始训练
- `infer.py`：模型推理入口，用于噪声去噪推理
- `evaluate.py`：模型评估入口，用于预训练模型评估

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
- 最小连通性验证走 `tests/` 中的 smoke 测试，避免新增重复脚本入口。

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
- 开始任务先阅读：`AGENTS.md`、`pyproject.toml`、相关入口文件与 `tests/`。
- 涉及第三方库时先查 Context7，再编码。
- 优先做最小可验证改动，并同步更新必要测试与文档。
- 不允许执行破坏性操作（如删除大批数据、覆盖外部参考资源）除非用户明确授权。
- 完成后最少回归：
  - 受影响测试。
  - 至少一次入口脚本或 smoke 验证。
  - 变更摘要（改了什么、为什么、如何验证）。

## 15) 当前统一扩散工程文件职责（本次实现）
- `train.py`：三阶段训练入口，支持 `ecg_pretrain/ppg_pretrain/joint`、AMP、梯度裁剪、teacher prior、checkpoint 保存与恢复。
- `infer.py`：推理入口，支持 ECG-only/PPG-only/joint 三种模式，支持 checkpoint 加载与结果落盘。
- `utils.py`：根目录辅助函数，提供最小演示信号生成。
- `src/ecg_ppg_denoise/config/config.py`：实验总配置定义、校验、JSON 读写。
- `src/ecg_ppg_denoise/config/loader.py`：统一配置加载入口。
- `src/ecg_ppg_denoise/config/schema.py`：配置 schema 兼容导出（供测试与旧调用路径）。
- `src/ecg_ppg_denoise/data/multimodal_dataset.py`：`.npz/.pt/.npy` 通用数据集、滑窗切分、train/val 划分、合成数据回退。
- `src/dataset/QTdataset.py`：QT 专用数据集入口，负责调用 `data/Data_Preparation`、整理 ECG-only 训练/验证数组，并适配为当前多模态训练接口。
- `src/ecg_ppg_denoise/models/diffusion_schedule.py`：DDPM 1D 调度（线性 beta、q_sample、predict_x0_from_eps、p_sample）与 DDIM 预留接口。
- `src/ecg_ppg_denoise/models/blocks.py`：基础网络块与共享主干 `UnifiedNoisePredictor1D`（1D U-Net + FiLM/AdaGN）。
- `src/ecg_ppg_denoise/models/encoders.py`：ECG/PPG 条件编码器，当前采用 UniCardio 风格的多卷积核并行 `singleEncoder` 结构。
- `src/ecg_ppg_denoise/models/quality_assessor.py`：质量评估器 `QualityAssessor1D`，负责从条件特征生成局部质量图 `q_map`。
- `src/ecg_ppg_denoise/models/fusion.py`：模态存在性 + 质量双重门控融合，含 missing token 与 joint 条件编码。
- `src/ecg_ppg_denoise/models/unified_diffusion_model.py`：统一模型 `ModalityFlexibleConditionalDiffusion`，仅负责 forward、条件噪声预测与 `denoise_signal` 采样。
- src/ecg_ppg_denoise/losses/masked_losses.py：支持 modality_mask 的 MSE/L1/导数损失。
- src/ecg_ppg_denoise/losses/diffusion_losses.py：统一扩散训练损失计算器，负责随机时间步采样、q_sample、四类损失聚合与 teacher prior。
- `src/models/single_encoder.py`：ECG/PPG 单模态编码器，采用 UniCardio 风格多分支卷积结构，并支持缺失模态直接输出零特征。
- `src/models/conditional_model.py`：底层噪声预测网络 `ConditionalNoiseModel1D`，同时包含时间嵌入、FiLM 残差块、上下采样和共享 U-Net 主干 `SharedConditionalUNet1D`，负责条件编码、联合上下文构建和噪声预测。
- `src/models/diffusion.py`：扩散调度器 `Diffusion1D`，负责 `q_sample`、`predict_x0_from_eps`、`p_sample` 与 DDIM 预留接口。
- `src/models/main_model.py`：主模型接口 `DDPM`，负责训练/推理分发、缺失模态处理、噪声预测调用与反向采样。
- `src/losses/diffusion_losses.py`：扩散训练损失，负责随机时间步采样、前向加噪、噪声监督和导数损失聚合。
- `src/trainers/runner.py`：训练引擎、阶段 mask 采样和 smoke 运行器 `ExperimentRunner`。
- `src/utils/common.py`：随机种子、设备解析、配置快照、checkpoint IO 和演示信号生成。
- `src/utils/logging.py`：结构化日志构建。
- `src/config/base.yaml`：默认模型与训练入口统一使用的基础 YAML 配置，顶层包含 `runtime/data/path/loss/model/train`。
- `src/config/loader.py`：默认配置加载与基础校验入口，返回嵌套 dict。
- `tests/test_multimodal_diffusion_smoke.py`：主模型前向输出、单步训练链路和缺失模态推理的 smoke 测试。
- `tests/test_smoke_pipeline.py`：最小训练链路 smoke 测试，验证 `ExperimentRunner` 可直接执行。
- `data/inspect_pickle.py`：pickle 查看脚本，默认读取 `D:\Code\data\PPG_FieldStudy\S1\S1.pkl` 并输出内容，兼容旧 pickle 编码回退。
- `data/resample.py`：PPG/ECG 重采样工具，提供 `ppg_resample` 与 `ecg_resample` 两个对外函数，显式接收 `source_hz` 和 `target_hz`，内部使用 `soxr` 开源库进行高质量重采样。
- `data/visualize_signal.py`：信号网页可视化脚本，读取 `S*.pkl` 中的 ECG 与 PPG/BVP 信号，生成带概览图和按 chunk 动态加载明细图的静态网页查看器。
- `tests/test_inspect_pickle_script.py`：验证 pickle 查看脚本对默认读取和 `latin1` 编码回退都可用。
- `tests/test_resample.py`：验证 PPG 上采样与 ECG 下采样后的长度和波形基本正确。
- `tests/test_visualize_signal.py`：验证网页波形查看器能够生成网页与 chunk 资源，并正常读取 pickle。
- `README.md`：项目说明、环境矩阵、训练/推理/测试命令、数据接口说明。
- `requirements.in`：运行依赖输入清单。
- `requirements-dev.in`：开发依赖输入清单。

