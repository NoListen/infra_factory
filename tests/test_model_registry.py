# tests/test_model_registry.py
"""
测试模型注册机制

测试VLAModelRegistry的功能
"""

import pytest
from unittest.mock import Mock

from vla_models.model_registry import (
    VLAModelRegistry,
    register_vla_model,
    register_framework_adapter,
    register_env_adapter,
    create_model,
    list_available_models,
)


# ==================== 测试模型注册 ====================

class TestVLAModelRegistry:
    """测试模型注册器"""

    def setup_method(self):
        """每个测试前清空注册表"""
        VLAModelRegistry._models.clear()
        VLAModelRegistry._adapters = {
            "areal": {},
            "rlinf": {},
            "verl": {},
        }

    def test_register_model(self):
        """测试注册模型"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        assert "test_model" in VLAModelRegistry._models
        assert VLAModelRegistry._models["test_model"] == TestModel

    def test_register_model_decorator(self):
        """测试使用装饰器注册模型"""
        @register_vla_model("decorated_model")
        class DecoratedModel:
            pass

        assert "decorated_model" in VLAModelRegistry._models

    def test_register_model_with_framework(self):
        """测试注册模型时指定框架"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel, framework="areal")

        assert "test_model" in VLAModelRegistry._models

    def test_register_adapter(self):
        """测试注册适配器"""
        def test_adapter(config):
            return Mock()

        VLAModelRegistry.register_adapter("areal", "test_model", test_adapter)

        assert "test_model" in VLAModelRegistry._adapters["areal"]
        assert VLAModelRegistry._adapters["areal"]["test_model"] == test_adapter

    def test_register_adapter_decorator(self):
        """测试使用装饰器注册适配器"""
        @register_framework_adapter("areal", "decorated_model")
        def decorated_adapter(config):
            return Mock()

        assert "decorated_model" in VLAModelRegistry._adapters["areal"]

    def test_register_env_adapter(self):
        """测试注册环境适配器"""
        class TestEnvAdapter:
            pass

        VLAModelRegistry.register_env_adapter("navsim", "areal", TestEnvAdapter)

        assert VLAModelRegistry._env_adapters["navsim"]["areal"] == TestEnvAdapter

    def test_get_model(self):
        """测试获取模型实例"""
        class TestModel:
            def __init__(self, **kwargs):
                self.config = kwargs

        VLAModelRegistry.register_model("test_model", TestModel)

        model = VLAModelRegistry.get_model("test_model", param1="value1")

        assert isinstance(model, TestModel)
        assert model.config["param1"] == "value1"

    def test_get_model_not_found(self):
        """测试获取不存在的模型"""
        with pytest.raises(ValueError, match="Unknown model"):
            VLAModelRegistry.get_model("nonexistent_model")

    def test_get_adapter(self):
        """测试获取适配器"""
        def test_adapter(config):
            return Mock()

        VLAModelRegistry.register_adapter("areal", "test_model", test_adapter)

        adapter = VLAModelRegistry.get_adapter("areal", "test_model")

        assert adapter == test_adapter

    def test_get_adapter_framework_not_found(self):
        """测试获取不存在的框架适配器"""
        with pytest.raises(ValueError, match="Unknown framework"):
            VLAModelRegistry.get_adapter("unknown_framework", "test_model")

    def test_get_adapter_model_not_found(self):
        """测试获取不存在模型的适配器"""
        with pytest.raises(ValueError, match="No.*adapter for"):
            VLAModelRegistry.get_adapter("areal", "nonexistent_model")

    def test_get_env_adapter(self):
        """测试获取环境适配器"""
        class TestEnvAdapter:
            pass

        VLAModelRegistry.register_env_adapter("navsim", "areal", TestEnvAdapter)

        adapter = VLAModelRegistry.get_env_adapter("navsim", "areal")

        assert adapter == TestEnvAdapter

    def test_get_env_adapter_not_found(self):
        """测试获取不存在的环境适配器"""
        with pytest.raises(ValueError, match="Unknown environment"):
            VLAModelRegistry.get_env_adapter("unknown_env", "areal")

    def test_list_models(self):
        """测试列出所有模型"""
        VLAModelRegistry.register_model("model1", Mock)
        VLAModelRegistry.register_model("model2", Mock)

        models = VLAModelRegistry.list_models()

        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_list_frameworks(self):
        """测试列出所有框架"""
        frameworks = VLAModelRegistry.list_frameworks()

        assert "areal" in frameworks
        assert "rlinf" in frameworks
        assert "verl" in frameworks

    def test_list_environments(self):
        """测试列出所有支持的环境"""
        envs = VLAModelRegistry.list_environments()

        assert "navsim" in envs
        assert "bench2drive" in envs


# ==================== 测试便捷函数 ====================

class TestConvenienceFunctions:
    """测试便捷函数"""

    def setup_method(self):
        """每个测试前清空注册表"""
        VLAModelRegistry._models.clear()
        VLAModelRegistry._adapters = {
            "areal": {},
            "rlinf": {},
            "verl": {},
        }

    def test_create_model_without_framework(self):
        """测试创建不带框架的模型"""
        class TestModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        VLAModelRegistry.register_model("test_model", TestModel)

        model = create_model("test_model", param1="value1")

        assert isinstance(model, TestModel)
        assert model.kwargs["param1"] == "value1"

    def test_create_model_with_framework(self):
        """测试创建带框架的模型"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        def areal_adapter(**kwargs):
            adapted = Mock()
            adapted.kwargs = kwargs
            return adapted

        VLAModelRegistry.register_adapter("areal", "test_model", areal_adapter)

        model = create_model("test_model", framework="areal", param1="value1")

        assert model.kwargs["param1"] == "value1"

    def test_create_model_with_unknown_framework(self):
        """测试使用未知框架创建模型"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        with pytest.raises(ValueError, match="No.*adapter"):
            create_model("test_model", framework="unknown_framework")

    def test_list_available_models_all(self):
        """测试列出所有可用模型"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        VLAModelRegistry.register_adapter("areal", "test_model", lambda **_: Mock())
        VLAModelRegistry.register_adapter("rlinf", "test_model", lambda **_: Mock())

        result = list_available_models()

        assert "test_model" in result["base_models"]
        assert "test_model" in result["areal_adapters"]
        assert "test_model" in result["rlinf_adapters"]

    def test_list_available_models_filter_by_framework(self):
        """测试按框架过滤列出模型"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        VLAModelRegistry.register_adapter("areal", "test_model", lambda **_: Mock())
        VLAModelRegistry.register_adapter("rlinf", "test_model", lambda **_: Mock())

        result = list_available_models(framework="areal")

        assert "test_model" in result["base_models"]
        assert "test_model" in result["areal_adapters"]
        assert "rlinf_adapters" not in result  # 不应该包含其他框架


# ==================== 集成测试 ====================

class TestModelRegistryIntegration:
    """集成测试：完整的工作流程"""

    def setup_method(self):
        """每个测试前清空注册表"""
        VLAModelRegistry._models.clear()
        VLAModelRegistry._adapters = {
            "areal": {},
            "rlinf": {},
            "verl": {},
        }
        VLAModelRegistry._env_adapters = {
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

    def test_complete_registration_workflow(self):
        """测试完整的注册工作流程"""
        # 1. 定义模型类
        class MyDrivingVLA:
            def __init__(self, model_path, **kwargs):
                self.model_path = model_path
                self.kwargs = kwargs

        # 2. 注册模型
        VLAModelRegistry.register_model("my_driving_vla", MyDrivingVLA, framework="base")

        # 3. 注册框架适配器
        def areal_adapter(model_path, **kwargs):
            model = MyDrivingVLA(model_path, **kwargs)
            # 模拟框架适配
            adapted = Mock()
            adapted.base_model = model
            adapted.framework = "areal"
            return adapted

        VLAModelRegistry.register_adapter("areal", "my_driving_vla", areal_adapter)

        # 4. 注册环境适配器
        class AReaLNavSimAdapter:
            pass

        VLAModelRegistry.register_env_adapter("navsim", "areal", AReaLNavSimAdapter)

        # 5. 验证注册
        assert "my_driving_vla" in VLAModelRegistry.list_models()
        assert "my_driving_vla" in VLAModelRegistry._adapters["areal"]
        assert VLAModelRegistry._env_adapters["navsim"]["areal"] == AReaLNavSimAdapter

        # 6. 创建模型实例
        base_model = VLAModelRegistry.get_model("my_driving_vla", model_path="/fake/path")
        assert isinstance(base_model, MyDrivingVLA)

        # 7. 创建适配后的模型
        adapted_model = VLAModelRegistry.get_adapter("areal", "my_driving_vla")
        adapted_instance = adapted_model(model_path="/fake/path", lr=1e-4)
        assert adapted_instance.framework == "areal"

        # 8. 获取环境适配器
        env_adapter = VLAModelRegistry.get_env_adapter("navsim", "areal")
        assert env_adapter == AReaLNavSimAdapter

    def test_multiple_models_registration(self):
        """测试注册多个模型"""
        # 注册多个模型
        for i in range(5):
            model_name = f"model_{i}"
            VLAModelRegistry.register_model(model_name, Mock)

        models = VLAModelRegistry.list_models()

        assert len(models) == 5
        for i in range(5):
            assert f"model_{i}" in models

    def test_cross_framework_registration(self):
        """测试跨框架注册"""
        class TestModel:
            pass

        VLAModelRegistry.register_model("test_model", TestModel)

        # 为每个框架注册适配器
        for framework in ["areal", "rlinf", "verl"]:
            adapter_fn = lambda **kw: Mock(framework=framework, **kw)
            VLAModelRegistry.register_adapter(framework, "test_model", adapter_fn)

        # 验证每个框架都有适配器
        for framework in ["areal", "rlinf", "verl"]:
            adapter = VLAModelRegistry.get_adapter(framework, "test_model")
            instance = adapter()
            assert instance.framework == framework

    def test_cross_environment_registration(self):
        """测试跨环境注册"""
        # 注册同一个适配器到多个环境的多个框架
        class UniversalAdapter:
            pass

        for env in ["navsim", "bench2drive"]:
            for framework in ["areal", "rlinf", "verl"]:
                # 为不同的组合创建不同的适配器类
                adapter_class = type(
                    f"{env.capitalize()}{framework.capitalize()}Adapter",
                    (UniversalAdapter,),
                    {}
                )
                VLAModelRegistry.register_env_adapter(env, framework, adapter_class)

        # 验证所有组合都已注册
        for env in ["navsim", "bench2drive"]:
            for framework in ["areal", "rlinf", "verl"]:
                adapter = VLAModelRegistry.get_env_adapter(env, framework)
                assert adapter is not None
                assert env.capitalize() in adapter.__name__
                assert framework.capitalize() in adapter.__name__


# ==================== 装饰器测试 ====================

class TestDecorators:
    """测试装饰器功能"""

    def setup_method(self):
        """每个测试前清空注册表"""
        VLAModelRegistry._models.clear()
        VLAModelRegistry._adapters = {
            "areal": {},
            "rlinf": {},
            "verl": {},
        }

    def test_register_vla_model_decorator_preserves_class(self):
        """测试模型注册装饰器不改变类"""
        @register_vla_model("test_model", framework="base")
        class TestModel:
            def __init__(self):
                self.value = 42

        model = TestModel()
        assert model.value == 42

    def test_register_vla_model_decorator_multiple(self):
        """测试多次使用装饰器"""
        @register_vla_model("model1")
        class Model1:
            pass

        @register_vla_model("model2")
        class Model2:
            pass

        models = VLAModelRegistry.list_models()
        assert "model1" in models
        assert "model2" in models

    def test_register_framework_adapter_decorator_preserves_function(self):
        """测试适配器装饰器不改变函数"""
        @register_framework_adapter("areal", "test_model")
        def test_adapter(**kwargs):
            return Mock(**kwargs)

        # 函数应该仍然可以正常调用
        result = test_adapter(param1="value1")
        assert result.param1 == "value1"

    def test_register_env_adapter_decorator_preserves_class(self):
        """测试环境适配器装饰器不改变类"""
        @register_env_adapter("navsim", "areal")
        class TestEnvAdapter:
            def __init__(self, config):
                self.config = config

        adapter = TestEnvAdapter({"test": "config"})
        assert adapter.config == {"test": "config"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
