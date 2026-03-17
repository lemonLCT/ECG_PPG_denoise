# ECG/PPG 去噪实验工程

本项目实现统一的多模态条件扩散模型 `ModalityFlexibleConditionalDiffusion`，支持以下三种输入模式：

1. `ECG-only`，对应 `modality_mask=[1, 0]`
2. `PPG-only`，对应 `modality_mask=[0, 1]`
3. `ECG+PPG joint`，对应 `modality_mask=[1, 1]`

当前实现的关键约束如下：

- 缺失模态会在条件编码和噪声预测主干之前被显式屏蔽，不允许信息泄漏到主干输入。
- 默认训练目标是标准扩散噪声预测损失。
- `reconstruction_weight` 和 `derivative_weight` 是可选辅助项，默认关闭。
- `teacher prior` 已移除，当前版本不再支持 teacher checkpoint。

## 目录

```text
.
├─ configs/
├─ src/ecg_ppg_denoise/
│  ├─ config/
│  ├─ data/
│  ├─ losses/
│  ├─ models/
│  ├─ trainers/
│  └─ utils/
├─ tests/
├─ train.py
├─ infer.py
└─ evaluate.py
```

## 环境

- Python: `>=3.10`
- 推荐 PyTorch: `>=2.2`
- 主要依赖：`torch`、`numpy`、`PyYAML`、`tqdm`、`scikit-learn`

安装：

```bash
pip install -e ".[dev]"
```

如果使用 `pip-tools`：

```bash
pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
pip install -r requirements-dev.txt
```

## 配置

默认配置位于 [base.yaml](/D:/Code/OriginCode/PyCharm_Project/ECG_PPG_denoise/src/config/base.yaml)。

其中与当前训练逻辑最相关的字段：

- `loss.diffusion_weight`：标准噪声预测损失权重，默认 `1.0`
- `loss.reconstruction_weight`：`x0` 重建辅助损失，默认 `0.0`
- `loss.derivative_weight`：导数域辅助损失，默认 `0.0`
- `training.stages.*.modality_dropout`：联合训练时的模态 dropout

如果你的目标是严格复现实验中的标准扩散训练，不要开启两个辅助损失。

## 训练

三阶段训练示例：

```bash
python train.py --config src/config/base.yaml --stage ecg_pretrain
python train.py --config src/config/base.yaml --stage ppg_pretrain
python train.py --config src/config/base.yaml --stage joint
```

常用参数：

- `--config <yaml_path>`：配置文件路径
- `--model-name <model_key>`：选择 `models.variants` 中的模型配置
- `--device cpu|cuda`
- `--output-dir <path>`
- `--resume <checkpoint>`
- `--epochs <int>`
- `--batch-size <int>`
- `--lr <float>`
- `--max-steps-per-epoch <int>`

训练输出默认包含：

- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `config_snapshot.yaml`

## 推理

基础用法：

```bash
python infer.py --config src/config/base.yaml --checkpoint artifacts/runs/default/checkpoints/best.pt
```

单模态与联合模式：

```bash
python infer.py --mode ecg
python infer.py --mode ppg
python infer.py --mode joint
```

推理输出默认保存到 `artifacts/infer/denoised_result.npz`。

输出字段语义：

- 观测到的模态输出为 `denoised_ecg` / `denoised_ppg`
- 未观测模态如果仍被模型生成，会保存为 `generated_ecg` / `generated_ppg`
- `generated_*` 不能和有监督去噪结果直接等价比较

## 评估

```bash
python evaluate.py --config src/config/base.yaml --checkpoint artifacts/runs/default/checkpoints/best.pt --input-path your_data.npz
```

当前评估脚本聚焦 ECG 指标，支持：

- `SSD`
- `MAD`
- `PRD`
- `COS_SIM`
- `SNR`
- `SNR_improvement`

## 数据接口

支持 `.npz`、`.pt`、`.npy` 输入，其中键至少包含：

- `clean_ecg`
- `noisy_ecg`
- `clean_ppg`
- `noisy_ppg`

数组形状支持 `[N, T]` 或 `[N, 1, T]`，内部会统一到 `[N, 1, T]`。

## 数据脚本

`data/ppg_noise_generate.py` 已用纯 Python 复现 `gen_PPG_artifacts.m` 的核心生成逻辑，不再依赖 MATLAB 或 Octave。脚本会生成 PPG 伪影，并同时导出 CSV、SVG 波形图和 JSON 元数据：

```bash
python data/ppg_noise_generate.py --duration-samples 4096 --sampling-rate-hz 64 --preset demo
```

如果要显式指定原函数的全部参数，可以这样调用：

```bash
python data/ppg_noise_generate.py \
  --duration-samples 4096 \
  --sampling-rate-hz 64 \
  --prob 0.25 0.25 0.25 0.25 \
  --dur-mu 12 4 4 4 4 \
  --rms-shape 2 2 2 2 \
  --rms-scale 0.35 0.45 0.55 0.75 \
  --slope -6 -8 -10 -12
```

常用参数说明：
- `--seed <int>`：固定随机种子，便于复现实验
- `--output-dir <path>` 与 `--output-stem <name>`：控制输出目录和文件名前缀
- `--save-states`：额外导出状态序列，0 表示无伪影，1-4 表示 4 类伪影

脚本默认输出到 `artifacts/ppg_noise/`，生成：
- `*.csv`：逐采样点的噪声序列
- `*.svg`：可直接查看的波形图
- `*.json`：运行时参数与元数据快照
- `*_states.csv`：可选的状态序列输出

## 测试

```bash
pytest -q
```

已补充的关键回归点：

- 缺失模态不会泄漏到噪声预测主干
- 默认 `total_loss` 等于 `diffusion_loss`
- 单模态/双模态前向均可运行
