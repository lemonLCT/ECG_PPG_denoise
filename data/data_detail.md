# 数据集说明

当前使用的数据位于 `D:\Code\data\PPG_FieldStudy\`。根据 `PPG_FieldStudy_readme.pdf` 的描述，这是一套面向 PPG 心率估计的日常活动场景数据；仓库当前上下文中通常将其称为 `PPG-DaLiA` 数据。

## 数据集概览

- 共 15 名受试者，7 男 8 女。
- 年龄为 `30.60 ± 9.59` 岁。
- 每名受试者都完成一套接近日常生活的采集流程，总时长约 `2.5` 小时。
- `S6` 在采集时存在硬件问题，因此 `S6.pkl` 只保留了前约 `1.5` 小时的有效数据。

## 受试者目录结构

每个受试者对应一个目录 `SX`，其中 `X` 为受试者编号，例如 `S1/`、`S2/`。

每个目录下包含以下文件：

- `SX_quest.csv`：受试者基本信息，包括年龄、性别、身高、体重、皮肤类型、运动频率。
- `SX_activity.csv`：活动标签，时间以秒为单位，表示对应活动的起始时刻。
- `SX_RespiBAN.h5`：胸带设备 RespiBAN 的原始数据。
- `SX_E4.zip`：腕带设备 Empatica E4 的原始数据压缩包。
- `SX.pkl`：已经同步完成的多源信号与标签，后续训练和分析通常优先使用这个文件。

## 采集设备与传感器

### RespiBAN 胸带设备

PDF 中说明 RespiBAN 为胸带式设备，采集信号至少包括：

- ECG
- Respiration
- 3 轴加速度

README 中给出的 RespiBAN 原始采样率为：

- 所有 RespiBAN 信号统一为 `700 Hz`

结合 `S1.pkl` 的实际内容，`signal["chest"]` 中还能看到以下胸带侧字段：

- `ACC`
- `ECG`
- `EMG`
- `EDA`
- `Temp`
- `Resp`

### Empatica E4 腕带设备

Empatica E4 佩戴在非惯用手腕，采样率按传感器不同而不同。根据 PDF 与 `S1_E4.zip` 中的文件头信息，可确认：

- `ACC.csv`：`32 Hz`，3 轴加速度，单位为 `1/64 g`
- `BVP.csv`：`64 Hz`，PPG/BVP 信号
- `EDA.csv`：`4 Hz`
- `TEMP.csv`：`4 Hz`，温度，单位为 `°C`

PDF 特别说明以下文件属于派生信息，通常应忽略：

- `HR.csv`
- `IBI.csv`
- `tags.csv`

## 数据同步与标签生成

### 同步方式

- 活动标签与 RespiBAN 原始数据天然同起点同步。
- RespiBAN 与 Empatica E4 之间需要手动同步。
- README 说明受试者会在采集开始和结束时，用佩戴 E4 的非惯用手拍击胸口两次，通过这个明显的模式对齐两台设备。
- 结束时的双击模式还用于校正设备时钟漂移。
- 漂移修正幅度整体较小；README 中提到最严重的是 `S9` 和 `S10`，最多只需删除约 `1.2` 秒数据。

### 心率标签生成

- 数据集目标是做基于 PPG 的心率估计，因此地面真值来自 ECG。
- README 说明先进行 R-peak 检测，再人工检查并修正异常峰值。
- 基于修正后的 R-peak 计算瞬时心率。
- 最终采用滑动窗口生成标签：
  - 窗长 `8` 秒
  - 窗移 `2` 秒
- 每个窗口对应的标签是该窗口内 ECG 瞬时心率的均值。

### 活动标签

README 中给出的活动 ID 如下：

- `0`：活动切换时的过渡阶段
- `1`：Sitting
- `2`：Ascending and descending stairs
- `3`：Table soccer
- `4`：Cycling
- `5`：Driving a car
- `6`：Lunch break
- `7`：Walking
- `8`：Working

另外，`SX.pkl` 中为了便于处理，还构造了一个 `4 Hz` 的活动信号，这也是全部原始模态中的最低采样率。

## `SX.pkl` 文件内容说明

以 `D:\Code\data\PPG_FieldStudy\S1\S1.pkl` 为例，顶层是一个 `dict`，包含以下键：

- `signal`
- `label`
- `activity`
- `questionnaire`
- `rpeaks`
- `subject`

各字段含义如下。

### `signal`

同步后的原始传感器信号，分为两个子域：

- `signal["chest"]`
  - `ACC`
  - `ECG`
  - `EMG`
  - `EDA`
  - `Temp`
  - `Resp`
- `signal["wrist"]`
  - `ACC`
  - `BVP`
  - `EDA`
  - `TEMP`

### `label`

ECG 推导出的心率标签，对应 `8` 秒窗、`2` 秒滑动生成的窗口级心率均值。

### `activity`

由 `SX_activity.csv` 转换而来的活动序列，在 `SX.pkl` 中按 `4 Hz` 对齐保存，便于与其它模态统一处理。

### `questionnaire`

受试者静态属性，来自 `SX_quest.csv`。在 `S1.pkl` 中可见字段包括：

- `WEIGHT`
- `Gender`
- `AGE`
- `HEIGHT`
- `SKIN`
- `SPORT`

### `rpeaks`

ECG 上检测并人工修正后的 R 峰位置，是生成心率真值的基础。

### `subject`

受试者编号，例如 `S1`。

## 预处理时可直接使用的要点

- 如果任务是 PPG 心率估计，核心输入通常是 `signal["wrist"]["BVP"]`，监督信号是 `label`。
- 如果需要运动伪影辅助信息，可以同时使用 `signal["wrist"]["ACC"]`。
- 如果要做多模态对齐建模，可以进一步利用 `signal["chest"]` 中的 ECG、Resp、ACC 等胸带信号。
- 若严格复现 README 中的标注方式，建议统一采用 `8` 秒窗口、`2` 秒步长。
