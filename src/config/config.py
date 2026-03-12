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
    num_workers: int = 0
    val_ratio: float = 0.1
    window_length: int = 512
    window_stride: int = 256
    synthetic_num_samples: int = 1024


@dataclass
class PathConfig:
    dataset_path: str = ""
    checkpoint_path: str = ""
    infer_input_path: str = ""
    eval_input_path: str = ""
    train_output_dir: str = "artifacts/runs/normal_train"
    infer_output_path: str = "artifacts/infer/denoised_result.npz"
    eval_output_dir: str = "artifacts/eval"


@dataclass
class SingleEncoderConfig:
    input_channels: int = 1
    branch_channels: int = 48
    kernel_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11])
    output_channels: int = 128
    use_projection: bool = True


@dataclass
class QualityAssessorConfig:
    hidden_channels: int = 64
    output_channels: int = 1


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
    ecg_encoder: SingleEncoderConfig = field(default_factory=SingleEncoderConfig)
    ppg_encoder: SingleEncoderConfig = field(default_factory=SingleEncoderConfig)
    quality_assessor: QualityAssessorConfig = field(default_factory=QualityAssessorConfig)


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    reconstruction_weight: float = 0.0
    derivative_weight: float = 0.0


@dataclass
class TrainConfig:
    stage_name: str = "ecg_pretrain"
    batch_size: int = 32
    val_batch_size: int = 32
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
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def sync_legacy_fields(self) -> None:
        if self.path.dataset_path:
            self.data.data_path = self.path.dataset_path
        elif self.data.data_path:
            self.path.dataset_path = self.data.data_path

        if self.path.train_output_dir:
            self.runtime.output_dir = self.path.train_output_dir
        elif self.runtime.output_dir:
            self.path.train_output_dir = self.runtime.output_dir

        self.model.ecg_encoder.output_channels = self.model.cond_channels
        self.model.ppg_encoder.output_channels = self.model.cond_channels

    def validate(self) -> None:
        self.sync_legacy_fields()
        if not self.runtime.experiment_name:
            raise ValueError("runtime.experiment_name 不能为空")
        if self.runtime.seed < 0:
            raise ValueError("runtime.seed 必须 >= 0")
        if self.train.batch_size <= 0 or self.train.val_batch_size <= 0:
            raise ValueError("batch_size 和 val_batch_size 必须 > 0")
        if not (0.0 <= self.data.val_ratio < 1.0):
            raise ValueError("val_ratio 必须在 [0, 1) 区间")
        if self.data.window_length <= 0 or self.data.window_stride <= 0:
            raise ValueError("window_length 和 window_stride 必须 > 0")
        if not self.path.train_output_dir:
            raise ValueError("path.train_output_dir 不能为空")
        if not self.model.name:
            raise ValueError("model.name 不能为空")
        if self.model.signal_length <= 0:
            raise ValueError("model.signal_length 必须 > 0")
        if self.model.cond_channels <= 0 or self.model.base_channels <= 0:
            raise ValueError("模型通道数必须 > 0")
        if self.model.joint_channels <= 0:
            raise ValueError("joint_channels 必须 > 0")
        for encoder_name, encoder_cfg in (("ecg_encoder", self.model.ecg_encoder), ("ppg_encoder", self.model.ppg_encoder)):
            if encoder_cfg.input_channels <= 0:
                raise ValueError(f"{encoder_name}.input_channels 必须 > 0")
            if encoder_cfg.branch_channels <= 0:
                raise ValueError(f"{encoder_name}.branch_channels 必须 > 0")
            if not encoder_cfg.kernel_sizes:
                raise ValueError(f"{encoder_name}.kernel_sizes 不能为空")
            if any(kernel <= 0 or kernel % 2 == 0 for kernel in encoder_cfg.kernel_sizes):
                raise ValueError(f"{encoder_name}.kernel_sizes 必须全为正奇数")
            if encoder_cfg.output_channels != self.model.cond_channels:
                raise ValueError(f"{encoder_name}.output_channels 必须等于 model.cond_channels")
        if self.model.quality_assessor.hidden_channels <= 0:
            raise ValueError("quality_assessor.hidden_channels 必须 > 0")
        if self.model.quality_assessor.output_channels != 1:
            raise ValueError("quality_assessor.output_channels 当前必须为 1")
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
        self.sync_legacy_fields()
        return asdict(self)

    @classmethod
    def from_flat_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
        runtime = RuntimeConfig(**payload.get("runtime", {}))
        data = DataConfig(**payload.get("data", {}))
        path = PathConfig(**payload.get("path", {}))

        model_payload = dict(payload.get("model", {}))
        ecg_encoder = SingleEncoderConfig(**model_payload.pop("ecg_encoder", {}))
        ppg_encoder = SingleEncoderConfig(**model_payload.pop("ppg_encoder", {}))
        quality_assessor = QualityAssessorConfig(**model_payload.pop("quality_assessor", {}))
        model = ModelConfig(
            **model_payload,
            ecg_encoder=ecg_encoder,
            ppg_encoder=ppg_encoder,
            quality_assessor=quality_assessor,
        )

        loss = LossConfig(**payload.get("loss", {}))
        train = TrainConfig(**payload.get("train", {}))
        cfg = cls(runtime=runtime, data=data, path=path, model=model, loss=loss, train=train)
        cfg.validate()
        return cfg
