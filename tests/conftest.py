# tests/conftest.py
"""
测试配置和共享fixtures
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any


# ==================== GPU配置 ====================

def pytest_configure(config):
    """Pytest配置"""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def device():
    """获取测试设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def float32_dtype(device):
    """确定是否使用float32（CPU通常需要）"""
    if device.type == "cpu":
        return torch.float32
    else:
        return torch.float16 if torch.cuda.is_bf16_supported() else torch.float32


@pytest.fixture
def sample_text():
    """示例文本"""
    return "Drive safely to the destination"


@pytest.fixture
def sample_image_path(tmp_path):
    """创建示例图像文件"""
    from PIL import Image
    import numpy as np

    # 创建一个简单的RGB图像
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    return str(img_path)


@pytest.fixture
def sample_image_tensor():
    """示例图像张量"""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_text():
    """示例文本批次"""
    return [
        "Turn left at the intersection",
        "Follow the lane for 100m",
        "Stop at the red light",
    ]


@pytest.fixture
def sample_driving_state():
    """示例驾驶状态"""
    return {
        "position": np.array([10.5, -3.2, 0.0]),
        "heading": 0.78,  # ~45 degrees in radians
        "velocity": np.array([5.0, 0.2, 0.0]),  # m/s
    }


@pytest.fixture
def sample_route():
    """示例路线"""
    return [
        {
            "position": np.array([15.0, -3.0, 0.0]),
            "command": "LANE_FOLLOW",
        },
        {
            "position": np.array([20.0, -3.0, 0.0]),
            "command": "STRAIGHT",
        },
        {
            "position": np.array([25.0, 0.0, 0.0]),
            "command": "TURN_LEFT",
        },
    ]


@pytest.fixture
def sample_waypoints():
    """示例路点"""
    return np.array([
        [12.0, -2.5, 0.0],
        [14.0, -2.0, 0.0],
        [16.0, -1.5, 0.0],
        [18.0, -1.0, 0.0],
        [20.0, -0.5, 0.0],
    ])


@pytest.fixture
def sample_control_signals():
    """示例控制信号"""
    return {
        "throttle": 0.5,
        "steer": 0.2,
        "brake": 0.0,
    }


# ==================== Mock模型 ====================

@pytest.fixture
def mock_vla_model():
    """Mock VLA模型"""
    from unittest.mock import Mock, MagicMock

    model = Mock(spec_set=[
        "encode_text",
        "encode_image",
        "encode_multimodal",
        "forward",
        "compute_logprobs",
        "predict",
        "get_param_groups",
        "get_trainable_params",
        "freeze_vision_encoder",
        "unfreeze_vision_encoder",
        "state_dict",
        "load_state_dict",
        "save_pretrained",
        "from_pretrained",
        "action_dim",
        "observation_space",
    ])

    # 设置默认返回值
    model.encode_text.return_value = {
        "input_ids": torch.randint(0, 1000, (2, 50)),
        "attention_mask": torch.ones(2, 50, dtype=torch.long),
    }

    model.encode_image.return_value = torch.randn(2, 3, 224, 224)

    model.forward.return_value = {
        "logits": torch.randn(2, 50, 10000),
        "hidden_states": torch.randn(2, 50, 768),
    }

    model.compute_logprobs.return_value = torch.randn(2, 50)

    model.predict.return_value = MagicMock(
        token_ids=torch.randint(0, 1000, (2, 10)),
        logprobs=torch.randn(2, 10),
        actions=torch.randn(2, 15),
    )

    model.get_param_groups.return_value = {
        "vision_encoder": [],
        "language_model": [],
        "policy_head": [],
    }

    model.get_trainable_params.return_value = []

    model.state_dict.return_value = {}
    model.load_state_dict.return_value = None
    model.save_pretrained.return_value = None
    model.from_pretrained.return_value = None

    model.action_dim = 15
    model.observation_space = {
        "image": (3, 224, 224),
        "language": "text",
    }

    return model


@pytest.fixture
def mock_driving_vla_model(mock_vla_model):
    """Mock驾驶VLA模型"""
    from unittest.mock import Mock, MagicMock

    model = mock_vla_model

    # 添加驾驶特定方法
    model.process_ego_state = Mock(return_value="Vehicle at (10.5, -3.2, 0.0)")
    model.process_route = Mock(return_value="Route: follow lane for 100m")
    model.encode_bev_map = Mock(return_value=torch.randn(1, 64, 200, 200))
    model.predict_waypoints = Mock(return_value=np.random.randn(5, 3))
    model.waypoints_to_controls = Mock(return_value={
        "throttle": 0.5,
        "steer": 0.2,
        "brake": 0.0,
    })
    model.predict_controls = Mock(return_value={
        "throttle": 0.5,
        "steer": 0.2,
        "brake": 0.0,
    })

    return model


# ==================== Mock环境 ====================

@pytest.fixture
def mock_environment():
    """Mock环境"""
    from unittest.mock import Mock

    env = Mock(spec_set=["reset", "step", "close"])

    # reset返回值
    env.reset.return_value = {
        "camera": [Mock()] * 3,  # 多相机
        "ego_state": {
            "position": np.array([0.0, 0.0, 0.0]),
            "heading": 0.0,
            "velocity": np.array([0.0, 0.0, 0.0]),
        },
        "route": [],
        "goal": {"position": np.array([100.0, 0.0, 0.0])},
    }

    # step返回值
    env.step.return_value = (
        {  # observation
            "camera": [Mock()] * 3,
            "ego_state": {
                "position": np.array([1.0, 0.0, 0.0]),
                "heading": 0.1,
                "velocity": np.array([1.0, 0.0, 0.0]),
            },
        },
        0.5,  # reward
        False,  # done
        {},  # info
    )

    return env


@pytest.fixture
def mock_areal_environment():
    """Mock AReaL环境（异步）"""
    from unittest.mock import AsyncMock, Mock

    env = AsyncMock()
    env._initialized = False

    # 异步方法
    env.ainitialize = AsyncMock(return_value=None)
    env.aexecute = AsyncMock(return_value={"success": True})
    env.aclose = AsyncMock(return_value=None)

    # 同步方法
    env.list_tools = Mock(return_value=[
        {
            "name": "reset",
            "description": "Reset environment",
        },
        {
            "name": "step_with_waypoints",
            "description": "Execute driving with waypoints",
        },
    ])

    return env


# ==================== 测试数据生成器 ====================

@pytest.fixture
def text_encoder_output():
    """文本编码器输出"""
    return {
        "input_ids": torch.tensor([
            [1, 234, 567, 890, 2],
            [1, 345, 678, 901, 2],
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.long),
    }


@pytest.fixture
def image_encoder_output():
    """图像编码器输出"""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def model_forward_output():
    """模型前向传播输出"""
    return {
        "logits": torch.randn(2, 50, 10000),
        "hidden_states": torch.randn(2, 50, 768),
        "vision_features": torch.randn(2, 49, 768),
    }


# ==================== 配置fixtures ====================

@pytest.fixture
def default_model_config():
    """默认模型配置"""
    return {
        "name": "test_driving_vla",
        "version": "0.1.0",
        "checkpoint_path": "/fake/path/to/checkpoint",

        "vision_encoder": {
            "type": "clip",
            "pretrained_path": "/fake/path/to/vision_encoder",
            "freeze": False,
            "projection_dim": 768,
            "image_size": (224, 224),
        },

        "language_model": {
            "base_model": "gpt2",
            "lora_enabled": False,
        },

        "policy_head": {
            "type": "waypoint",
            "action_dim": 15,  # 5 waypoints * 3 coords
            "hidden_dim": 512,
        },
    }


@pytest.fixture
def default_training_config():
    """默认训练配置"""
    return {
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
        "param_groups": {
            "vision_encoder": 1e-5,
            "language_model": 1e-4,
            "policy_head": 1e-3,
        },
    }


@pytest.fixture
def default_env_config():
    """默认环境配置"""
    return {
        "type": "navsim",
        "observation": {
            "camera": {
                "enabled": True,
                "width": 224,
                "height": 224,
            },
        },
        "action": {
            "type": "waypoint",
            "num_waypoints": 5,
        },
    }
