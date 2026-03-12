from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Type, TypeVar


T = TypeVar("T", bound="ExperimentConfig")


@dataclass
class RuntimeConfig:
    experiment_name: str = "modality_flexible_diffusion"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "artifacts/runs/normal_train"
    use_amp: bool = True


@dataclass
class DataConfig:
    data_path: str = ""
    batch_size: int = 32
    val_batch_size: int = 32
    num_workers: int = 0
    val_ratio: float = 0.1
    window_length: int = 512
    window_stride: int = 256
    synthetic_num_samples: int = 1024


@dataclass
class ModelConfig:
    name: str = "modality_flexible_conditional_diffusion"
    signal_length: int = 512
    cond_channels: int = 128
    joint_channels: int = 256
    base_channels: int = 64
    gn_groups: int = 8
    diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    reconstruction_weight: float = 0.0
    derivative_weight: float = 0.0


@dataclass
class TrainConfig:
    stage_name: str = "ecg_pretrain"
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    save_every_epochs: int = 1
    max_steps_per_epoch: int = 2000
    modality_dropout: float = 0.1


@dataclass
class ExperimentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if not self.runtime.experiment_name:
            raise ValueError("runtime.experiment_name 不能为空")
        if self.runtime.seed < 0:
            raise ValueError("runtime.seed 必须 >= 0")
        if self.data.batch_size <= 0 or self.data.val_batch_size <= 0:
            raise ValueError("batch_size 和 val_batch_size 必须 > 0")
        if not (0.0 <= self.data.val_ratio < 1.0):
            raise ValueError("val_ratio 必须在 [0, 1) 区间")
        if self.data.window_length <= 0 or self.data.window_stride <= 0:
            raise ValueError("window_length 和 window_stride 必须 > 0")
        if not self.model.name:
            raise ValueError("model.name 不能为空")
        if self.model.signal_length <= 0:
            raise ValueError("model.signal_length 必须 > 0")
        if self.model.cond_channels <= 0 or self.model.base_channels <= 0:
            raise ValueError("模型通道数必须 > 0")
        if self.model.diffusion_steps <= 1:
            raise ValueError("diffusion_steps 必须 > 1")
        if not (0.0 < self.model.beta_start < self.model.beta_end < 1.0):
            raise ValueError("beta_start/beta_end 必须满足 0 < beta_start < beta_end < 1")
        if not self.train.stage_name:
            raise ValueError("train.stage_name 不能为空")
        if self.train.epochs <= 0:
            raise ValueError("epochs 必须 > 0")
        if self.train.lr <= 0.0:
            raise ValueError("lr 必须 > 0")
        if self.train.max_steps_per_epoch <= 0:
            raise ValueError("max_steps_per_epoch 必须 > 0")
        if not (0.0 <= self.train.modality_dropout < 1.0):
            raise ValueError("modality_dropout 必须在 [0, 1) 区间")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_flat_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
        runtime = RuntimeConfig(**payload.get("runtime", {}))
        data = DataConfig(**payload.get("data", {}))
        model = ModelConfig(**payload.get("model", {}))
        loss = LossConfig(**payload.get("loss", {}))
        train = TrainConfig(**payload.get("train", {}))
        cfg = cls(runtime=runtime, data=data, model=model, loss=loss, train=train)
        cfg.validate()
        return cfg
