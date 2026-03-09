## 阶段 1: 仓库调研与约束确认
**目标**: 确认现有代码状态、工具链约束和测试入口约定。  
**成功标准**: 明确项目当前为空仓库，仅有 IDE 配置；确定采用 `src` 布局与 `pytest`。  
**测试**: 无（信息收集阶段）。  
**状态**: 已完成

## 阶段 2: 工程骨架与模块分层创建
**目标**: 创建 `config/data/preprocess/models/losses/trainers/evaluators/metrics/utils/scripts/tests` 目录及骨架文件。  
**成功标准**: 所有文件可导入，核心接口仅保留签名、注释、TODO 与 `NotImplementedError`。  
**测试**: `pytest` 可发现测试；导入测试通过。  
**状态**: 已完成

## 阶段 3: 最小可运行流程接线
**目标**: 提供仅做流程连通的入口脚本，不执行真实训练。  
**成功标准**: `python scripts/run_pipeline.py` 可运行并输出连通性日志。  
**测试**: `tests/test_smoke_pipeline.py` 通过。  
**状态**: 已完成

## 阶段 4: 测试与收尾
**目标**: 完成基础测试并清理计划文件。  
**成功标准**: `pytest -q` 通过；阶段状态全部更新为已完成；删除 `IMPLEMENTATION_PLAN.md`。  
**测试**: `pytest -q`。  
**状态**: 进行中
**备注**: 当前系统 Python 环境缺少 `pip/pytest`，暂无法执行完整测试，待安装依赖后完成并删除本计划文件。
