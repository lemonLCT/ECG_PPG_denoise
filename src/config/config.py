from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Type, TypeVar


T = TypeVar("T", bound="ExperimentConfig")


@dataclass
class RuntimeConfig:
    experiment_name: str = "multimodal_diffusion"
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "artifacts/runs/train"
    use_amp: bool = True


@dataclass
class DataConfig:
    data_path: str = ""
    num_workers: int = 0
    val_ratio: float = 0.3
    window_length: int = 500
    window_stride: int = 250
    synthetic_num_samples: int = 1024


@dataclass
class PathConfig:
    dataset_path: str = ""
    qt_root: str = "D:/Code/data/db/QTDB"
    ppg_fieldstudy_pickle_path: str = "D:/Code/data/PPG_FieldStudy/S1/S1.pkl"
    artifact_param_path: str = "D:/Code/data/数据处理脚本/artifact_param.mat"
    checkpoint_path: str = ""
    infer_input_path: str = ""
    eval_input_path: str = ""
    train_output_dir: str = "artifacts/runs/train"
    infer_output_path: str = "artifacts/infer/denoised_result.npz"
    eval_output_dir: str = "artifacts/eval"


@dataclass
class BidmcPathConfig:
    bidmc_root: str = "D:/Code/data/bidmc-ppg-and-respiration-dataset-1.0.0"
    nstdb_root: str = "D:/Code/data/db/QTDB/mit-bih-noise-stress-test-database-1.0.0"


@dataclass
class BidmcDataConfig:
    window_length: int = 500
    window_stride: int = 250
    sampling_rate_hz: float = 125.0
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalization: str = "window_minmax"
    ecg_channel_name: str = "II"
    ecg_channel_fallback_names: list[str] = field(default_factory=lambda: ["ECG", "MLII"])
    ppg_channel_name: str = "PLETH"
    ppg_channel_fallback_names: list[str] = field(default_factory=lambda: ["PPG"])
    ecg_noise_version: int = 1
    ecg_noise_ratio_range: list[float] = field(default_factory=lambda: [0.2, 2.0])
    bidmc_ppg_noise_seed: int = 2024
    ppg_noise_ratio_range: list[float] = field(default_factory=lambda: [0.2, 2.0])
    ppg_noise_artifact_param_path: str = ""
    ppg_noise_artifact_types: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    ppg_noise_dur_mu: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0])
    ppg_noise_rms_shape: list[float] = field(default_factory=lambda: [2.0, 2.0, 2.0, 2.0])
    ppg_noise_rms_scale: list[float] = field(default_factory=lambda: [0.35, 0.45, 0.55, 0.75])
    ppg_noise_slope_mean: list[float] = field(default_factory=lambda: [-6.0, -8.0, -10.0, -12.0])
    ppg_noise_slope_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class BidmcConfig:
    path: BidmcPathConfig = field(default_factory=BidmcPathConfig)
    data: BidmcDataConfig = field(default_factory=BidmcDataConfig)


@dataclass
class SingleEncoderConfig:
    input_channels: int = 1
    branch_channels: int = 16
    kernel_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11])
    output_channels: int = 96
    use_projection: bool = False
    gn_groups: int = 8

    @property
    def resolved_output_channels(self) -> int:
        if self.use_projection:
            return self.output_channels
        return self.branch_channels * len(self.kernel_sizes)


@dataclass
class QualityAssessorConfig:
    hidden_channels: int = 64
    output_channels: int = 1


@dataclass
class ModelConfig:
    name: str = "modality_flexible_conditional_diffusion"
    signal_length: int = 512
    cond_channels: int = 96
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
    derivative_weight: float = 0.0


@dataclass
class TrainConfig:
    stage_name: str = "joint"
    batch_size: int = 16
    val_batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    save_every_epochs: int = 50
    max_steps_per_epoch: int = 5000


@dataclass
class ExperimentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    path: PathConfig = field(default_factory=PathConfig)
    bidmc: BidmcConfig = field(default_factory=BidmcConfig)
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

        ecg_channels = self.model.ecg_encoder.resolved_output_channels
        ppg_channels = self.model.ppg_encoder.resolved_output_channels
        self.model.ecg_encoder.output_channels = ecg_channels
        self.model.ppg_encoder.output_channels = ppg_channels
        if ecg_channels == ppg_channels:
            self.model.cond_channels = ecg_channels

    def validate(self) -> None:
        self.sync_legacy_fields()
        if not self.runtime.experiment_name:
            raise ValueError("runtime.experiment_name 不能为空")
        if self.runtime.seed < 0:
            raise ValueError("runtime.seed 必须 >= 0")
        if self.train.batch_size <= 0 or self.train.val_batch_size <= 0:
            raise ValueError("batch_size 和 val_batch_size 必须 > 0")
        if not (0.0 <= self.data.val_ratio < 1.0):
            raise ValueError("data.val_ratio 必须在 [0, 1) 区间")
        if self.data.window_length <= 0 or self.data.window_stride <= 0:
            raise ValueError("data.window_length 和 data.window_stride 必须 > 0")
        if not self.path.train_output_dir:
            raise ValueError("path.train_output_dir 不能为空")
        if self.bidmc.data.window_length <= 0 or self.bidmc.data.window_stride <= 0:
            raise ValueError("bidmc.data.window_length 和 bidmc.data.window_stride 必须 > 0")
        if abs((self.bidmc.data.train_ratio + self.bidmc.data.val_ratio + self.bidmc.data.test_ratio) - 1.0) > 1e-6:
            raise ValueError("bidmc.data.train_ratio + val_ratio + test_ratio 必须等于 1")
        if self.bidmc.data.normalization != "window_minmax":
            raise ValueError("bidmc.data.normalization 当前仅支持 window_minmax")
        if self.model.signal_length <= 0:
            raise ValueError("model.signal_length 必须 > 0")
        if self.model.cond_channels <= 0 or self.model.base_channels <= 0:
            raise ValueError("模型通道数必须 > 0")
        if self.model.joint_channels <= 0:
            raise ValueError("joint_channels 必须 > 0")
        if self.train.stage_name not in {"ecg_pretrain", "ppg_pretrain", "joint"}:
            raise ValueError("train.stage_name 必须是 ecg_pretrain / ppg_pretrain / joint 之一")
        if self.train.epochs <= 0:
            raise ValueError("epochs 必须 > 0")
        if self.train.lr <= 0.0:
            raise ValueError("lr 必须 > 0")
        if self.train.max_steps_per_epoch <= 0:
            raise ValueError("max_steps_per_epoch 必须 > 0")

    def to_dict(self) -> Dict[str, Any]:
        self.sync_legacy_fields()
        return asdict(self)

    @classmethod
    def from_flat_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
        runtime = RuntimeConfig(**payload.get("runtime", {}))
        data = DataConfig(**payload.get("data", {}))
        path = PathConfig(**payload.get("path", {}))

        bidmc_payload = dict(payload.get("bidmc", {}))
        bidmc = BidmcConfig(
            path=BidmcPathConfig(**bidmc_payload.get("path", {})),
            data=BidmcDataConfig(**bidmc_payload.get("data", {})),
        )

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
        cfg = cls(runtime=runtime, data=data, path=path, bidmc=bidmc, model=model, loss=loss, train=train)
        cfg.validate()
        return cfg
