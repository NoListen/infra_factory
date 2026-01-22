# vla_models/model_registry.py
"""
VLA模型注册机制

统一管理所有VLA模型及其框架适配器
"""

from typing import Dict, Type, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VLAModelRegistry:
    """
    VLA模型注册器

    管理所有VLA模型和框架适配器
    """

    # 注册的模型类
    _models: Dict[str, Type] = {}

    # 框架适配器
    _adapters: Dict[str, Dict[str, Callable]] = {
        "areal": {},
        "rlinf": {},
        "verl": {},
    }

    # 环境适配器
    _env_adapters: Dict[str, Dict[str, Type]] = {
        "navsim": {
            "areal": None,
            "rlinf": None,
            "verl": None,
        },
        "bench2drive": {
            "areal": None,
            "rlinf": None,
            "verl": None,
        },
    }

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type,
        framework: str = "base",
    ):
        """
        注册VLA模型

        参数:
            name: 模型名称（如 "your_driving_vla"）
            model_class: 模型类（继承自VLAInterface）
            framework: 主要框架（base/areal/rlinf/verl）
        """
        cls._models[name] = model_class
        logger.info(f"Registered VLA model: {name} (framework: {framework})")

    @classmethod
    def register_adapter(
        cls,
        framework: str,
        model_name: str,
        adapter_fn: Callable,
    ):
        """
        注册框架适配器

        参数:
            framework: 框架名称（areal/rlinf/verl）
            model_name: 模型名称
            adapter_fn: 适配器函数，接收配置返回适配后的模型
        """
        if framework not in cls._adapters:
            raise ValueError(f"Unknown framework: {framework}")
        cls._adapters[framework][model_name] = adapter_fn
        logger.info(f"Registered {framework} adapter for {model_name}")

    @classmethod
    def register_env_adapter(
        cls,
        env_name: str,
        framework: str,
        adapter_class: Type,
    ):
        """
        注册环境适配器

        参数:
            env_name: 环境名称（navsim/bench2drive）
            framework: 框架名称
            adapter_class: 适配器类
        """
        if env_name not in cls._env_adapters:
            cls._env_adapters[env_name] = {}
        if framework not in cls._env_adapters[env_name]:
            cls._env_adapters[env_name] = {}
        cls._env_adapters[env_name][framework] = adapter_class
        logger.info(f"Registered {framework} adapter for {env_name}")

    @classmethod
    def get_model(cls, name: str, **kwargs) -> Any:
        """
        获取模型实例

        参数:
            name: 模型名称
            **kwargs: 模型初始化参数

        返回:
            模型实例
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model: {name}. Available models: {available}"
            )
        return cls._models[name](**kwargs)

    @classmethod
    def get_adapter(
        cls,
        framework: str,
        model_name: str,
    ):
        """
        获取框架适配器

        参数:
            framework: 框架名称
            model_name: 模型名称

        返回:
            适配器函数
        """
        if framework not in cls._adapters:
            raise ValueError(f"Unknown framework: {framework}")
        if model_name not in cls._adapters[framework]:
            raise ValueError(
                f"No {framework} adapter for {model_name}. "
                f"Available: {list(cls._adapters[framework].keys())}"
            )
        return cls._adapters[framework][model_name]

    @classmethod
    def get_env_adapter(
        cls,
        env_name: str,
        framework: str,
    ):
        """
        获取环境适配器

        参数:
            env_name: 环境名称
            framework: 框架名称

        返回:
            环境适配器类
        """
        if env_name not in cls._env_adapters:
            raise ValueError(f"Unknown environment: {env_name}")
        if framework not in cls._env_adapters[env_name]:
            raise ValueError(f"Unknown framework: {framework}")
        adapter_class = cls._env_adapters[env_name][framework]
        if adapter_class is None:
            raise ValueError(
                f"No {framework} adapter for {env_name}. "
                "Register an adapter first."
            )
        return adapter_class

    @classmethod
    def list_models(cls) -> list[str]:
        """列出所有注册的模型"""
        return list(cls._models.keys())

    @classmethod
    def list_frameworks(cls) -> list[str]:
        """列出所有支持的框架"""
        return list(cls._adapters.keys())

    @classmethod
    def list_environments(cls) -> list[str]:
        """列出所有支持的环境"""
        return list(cls._env_adapters.keys())


# ==================== 装饰器 ====================

def register_vla_model(
    name: str,
    framework: str = "base",
):
    """
    注册VLA模型的装饰器

    使用:
    ```python
    @register_vla_model("my_driving_vla", framework="areal")
    class MyDrivingVLA(VLAForDriving):
        def __init__(self, ...):
            ...
    ```
    """
    def decorator(cls: Type):
        VLAModelRegistry.register_model(name, cls, framework)
        return cls
    return decorator


def register_framework_adapter(
    framework: str,
    model_name: str,
):
    """
    注册框架适配器的装饰器

    使用:
    ```python
    @register_framework_adapter("areal", "my_driving_vla")
    def areal_adapter(model_config, engine_config):
        from my_vla import MyDrivingVLA
        from areal.models import MyVLAForAReaL
        base_model = MyDrivingVLA(**model_config)
        return MyVLAForAReaL(base_model, **engine_config)
    ```
    """
    def decorator(fn: Callable):
        VLAModelRegistry.register_adapter(framework, model_name, fn)
        return fn
    return decorator


def register_env_adapter(
    env_name: str,
    framework: str,
):
    """
    注册环境适配器的装饰器

    使用:
    ```python
    @register_env_adapter("navsim", "areal")
    class AReaLNavSimAdapter(NavSimEnvironmentInterface):
        def __init__(self, ...):
            ...
    ```
    """
    def decorator(cls: Type):
        VLAModelRegistry.register_env_adapter(env_name, framework, cls)
        return cls
    return decorator


# ==================== 便捷函数 ====================

def create_model(
    name: str,
    framework: Optional[str] = None,
    **kwargs
) -> Any:
    """
    创建VLA模型实例

    参数:
        name: 模型名称
        framework: 框架名称（如果为None，返回基础模型）
        **kwargs: 模型初始化参数

    返回:
        模型实例
    """
    if framework is None:
        # 返回基础模型
        return VLAModelRegistry.get_model(name, **kwargs)
    else:
        # 返回框架适配后的模型
        adapter_fn = VLAModelRegistry.get_adapter(framework, name)
        return adapter_fn(**kwargs)


def list_available_models(
    framework: Optional[str] = None,
) -> Dict[str, list]:
    """
    列出可用的模型

    参数:
        framework: 框架名称（如果指定，只列出该框架的模型）

    返回:
        {
            "base_models": [...],
            "areal_adapters": [...],
            "rlinf_adapters": [...],
            "verl_adapters": [...],
        }
    """
    result = {
        "base_models": VLAModelRegistry.list_models(),
    }

    for fw in VLAModelRegistry.list_frameworks():
        adapter_models = list(VLAModelRegistry._adapters[fw].keys())
        result[f"{fw}_adapters"] = adapter_models

    if framework:
        # 只返回指定框架的信息
        return {
            "base_models": result["base_models"],
            f"{framework}_adapters": result[f"{framework}_adapters"],
        }

    return result
