# vla_models/config.py
"""
VLA模型配置管理

统一的配置类和加载器
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import yaml
from pathlib import Path


# ==================== 配置类 ====================

@dataclass
class VisionEncoderConfig:
    """视觉编码器配置"""
    type: str = "clip"  # clip, siglip, dinov2, resnet, etc.
    pretrained_path: str = ""
    freeze: bool = False
    projection_dim: int = 768
    image_size: Tuple[int, int] = (224, 224)
    num_frames: int = 1  # 多帧支持


@dataclass
class LanguageModelConfig:
    """语言模型配置"""
    base_model: str = "gpt2"  # gpt2, qwen2.5, llama, etc.
    lora_enabled: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class PolicyHeadConfig:
    """策略头配置"""
    type: str = "waypoint"  # waypoint, direct_control, hybrid
    action_dim: int = 15  # 5 waypoints * 3 coords
    hidden_dim: int = 512
    num_layers: int = 2
    activation: str = "relu"
    output_activation: str = "tanh"  # tanh, none, sigmoid


@dataclass
class ValueHeadConfig:
    """价值头配置（actor-critic）"""
    enabled: bool = False
    hidden_dim: int = 256


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    version: str = "1.0.0"
    checkpoint_path: Optional[str] = None
    vision_encoder: Optional[VisionEncoderConfig] = None
    language_model: Optional[LanguageModelConfig] = None
    policy_head: Optional[PolicyHeadConfig] = None
    value_head: Optional[ValueHeadConfig] = None

    def __post_init__(self):
        if self.vision_encoder is None:
            self.vision_encoder = VisionEncoderConfig()
        if self.language_model is None:
            self.language_model = LanguageModelConfig()
        if self.policy_head is None:
            self.policy_head = PolicyHeadConfig()
        if self.value_head is None:
            self.value_head = ValueHeadConfig()


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "adamw"  # adamw, adam, sgd, etc.
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class ParamGroupConfig:
    """参数组配置"""
    vision_encoder: float = 1e-5
    language_model: float = 1e-4
    policy_head: float = 1e-3
    value_head: float = 1e-3


@dataclass
class LRSchedulerConfig:
    """学习率调度器配置"""
    type: str = "cosine"  # cosine, linear, constant, etc.
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    num_cycles: float = 0.5  # for cosine


@dataclass
class TrainingConfig:
    """训练配置"""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    param_groups: ParamGroupConfig = field(default_factory=ParamGroupConfig)
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class ObservationConfig:
    """观察配置"""
    camera: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "width": 224,
        "height": 224,
        "fov": 90,
    })
    bev_map: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "width": 200,
        "height": 200,
        "resolution": 0.5,  # 米/像素
    })
    state: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "include_velocity": True,
        "include_heading": True,
    })


@dataclass
class ActionConfig:
    """动作配置"""
    type: str = "waypoint"  # waypoint, continuous
    num_waypoints: int = 5
    waypoint_horizon: float = 3.0  # 秒
    throttle_range: Tuple[float, float] = (0, 1)
    steer_range: Tuple[float, float] = (-1, 1)
    brake_range: Tuple[float, float] = (0, 1)


@dataclass
class RewardConfig:
    """奖励配置"""
    type: str = "driving"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "collision": -10.0,
        "success": 10.0,
        "progress": 1.0,
        "lane_keeping": 0.5,
        "comfort": 0.1,
    })


@dataclass
class EnvironmentConfig:
    """环境配置"""
    type: str = "navsim"  # navsim, bench2drive
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)


@dataclass
class VLAConfig:
    """完整的VLA配置"""
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "VLAConfig":
        """
        从YAML文件加载配置

        参数:
            path: YAML文件路径

        返回:
            VLAConfig实例
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "VLAConfig":
        """从字典构建配置"""
        # 构建子配置
        model_config = ModelConfig(**data["model"])

        training_data = data.get("training", {})
        training_config = TrainingConfig(
            optimizer=OptimizerConfig(**training_data.get("optimizer", {})),
            lr_scheduler=LRSchedulerConfig(**training_data.get("lr_scheduler", {})),
            param_groups=ParamGroupConfig(**training_data.get("param_groups", {})),
            **{k: v for k, v in training_data.items()
               if k not in ["optimizer", "lr_scheduler", "param_groups"]},
        )

        env_data = data.get("environment", {})
        env_config = EnvironmentConfig(
            observation=ObservationConfig(**env_data.get("observation", {})),
            action=ActionConfig(**env_data.get("action", {})),
            **{k: v for k, v in env_data.items()
               if k not in ["observation", "action"]},
        )

        reward_config = RewardConfig(**data.get("reward", {}))

        return cls(
            model=model_config,
            training=training_config,
            environment=env_config,
            reward=reward_config,
        )

    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        data = {
            "model": self._model_to_dict(),
            "training": self._training_to_dict(),
            "environment": self._environment_to_dict(),
            "reward": self._reward_to_dict(),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _model_to_dict(self) -> dict:
        """转换模型配置为字典"""
        return {
            "name": self.model.name,
            "version": self.model.version,
            "checkpoint_path": self.model.checkpoint_path,
            "vision_encoder": {
                "type": self.model.vision_encoder.type,
                "pretrained_path": self.model.vision_encoder.pretrained_path,
                "freeze": self.model.vision_encoder.freeze,
                "projection_dim": self.model.vision_encoder.projection_dim,
                "image_size": self.model.vision_encoder.image_size,
                "num_frames": self.model.vision_encoder.num_frames,
            },
            "language_model": {
                "base_model": self.model.language_model.base_model,
                "lora_enabled": self.model.language_model.lora_enabled,
                "lora_r": self.model.language_model.lora_r,
                "lora_alpha": self.model.language_model.lora_alpha,
                "lora_dropout": self.model.language_model.lora_dropout,
                "lora_target_modules": self.model.language_model.lora_target_modules,
            },
            "policy_head": {
                "type": self.model.policy_head.type,
                "action_dim": self.model.policy_head.action_dim,
                "hidden_dim": self.model.policy_head.hidden_dim,
                "num_layers": self.model.policy_head.num_layers,
                "activation": self.model.policy_head.activation,
                "output_activation": self.model.policy_head.output_activation,
            },
            "value_head": {
                "enabled": self.model.value_head.enabled,
                "hidden_dim": self.model.value_head.hidden_dim,
            },
        }

    def _training_to_dict(self) -> dict:
        """转换训练配置为字典"""
        return {
            "optimizer": {
                "type": self.training.optimizer.type,
                "lr": self.training.optimizer.lr,
                "weight_decay": self.training.optimizer.weight_decay,
                "betas": self.training.optimizer.betas,
                "eps": self.training.optimizer.eps,
            },
            "lr_scheduler": {
                "type": self.training.lr_scheduler.type,
                "warmup_ratio": self.training.lr_scheduler.warmup_ratio,
                "min_lr_ratio": self.training.lr_scheduler.min_lr_ratio,
                "num_cycles": self.training.lr_scheduler.num_cycles,
            },
            "param_groups": {
                "vision_encoder": self.training.param_groups.vision_encoder,
                "language_model": self.training.param_groups.language_model,
                "policy_head": self.training.param_groups.policy_head,
                "value_head": self.training.param_groups.value_head,
            },
            "num_epochs": self.training.num_epochs,
            "batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "max_grad_norm": self.training.max_grad_norm,
        }

    def _environment_to_dict(self) -> dict:
        """转换环境配置为字典"""
        return {
            "type": self.environment.type,
            "observation": {
                "camera": self.environment.observation.camera,
                "bev_map": self.environment.observation.bev_map,
                "state": self.environment.observation.state,
            },
            "action": {
                "type": self.environment.action.type,
                "num_waypoints": self.environment.action.num_waypoints,
                "waypoint_horizon": self.environment.action.waypoint_horizon,
                "throttle_range": self.environment.action.throttle_range,
                "steer_range": self.environment.action.steer_range,
                "brake_range": self.environment.action.brake_range,
            },
        }

    def _reward_to_dict(self) -> dict:
        """转换奖励配置为字典"""
        return {
            "type": self.reward.type,
            "weights": self.reward.weights,
        }
