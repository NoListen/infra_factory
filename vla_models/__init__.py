# vla_models/__init__.py
"""
VLA Models Package

统一的VLA模型接口框架，支持AReaL、RLinf、Verl三个训练框架。
"""

__version__ = "0.1.0"

# 核心接口
from vla_models.interface import (
    VLAInterface,
    VLAForDriving,
    VLAInput,
    VLAPrediction,
)

# 注册机制
from vla_models.model_registry import (
    VLAModelRegistry,
    register_vla_model,
    register_framework_adapter,
    register_env_adapter,
    create_model,
    list_available_models,
)

# 配置
from vla_models.config import (
    VLAConfig,
    ModelConfig,
    TrainingConfig,
    EnvironmentConfig,
    VisionEncoderConfig,
    LanguageModelConfig,
    PolicyHeadConfig,
)

__all__ = [
    # 版本
    "__version__",
    # 接口
    "VLAInterface",
    "VLAForDriving",
    "VLAInput",
    "VLAPrediction",
    # 注册
    "VLAModelRegistry",
    "register_vla_model",
    "register_framework_adapter",
    "register_env_adapter",
    "create_model",
    "list_available_models",
    # 配置
    "VLAConfig",
    "ModelConfig",
    "TrainingConfig",
    "EnvironmentConfig",
    "VisionEncoderConfig",
    "LanguageModelConfig",
    "PolicyHeadConfig",
]
