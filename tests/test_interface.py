# tests/test_interface.py
"""
测试VLA接口实现

测试VLAInterface和VLAForDriving的接口规范
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from vla_models.interface import (
    VLAInterface,
    VLAForDriving,
    VLAInput,
    VLAPrediction,
)


# ==================== Mock实现 ====================

class MockVLAModel(VLAInterface):
    """Mock VLA模型用于测试"""

    def __init__(self, config=None):
        self.config = config or {}
        self._action_dim = 15
        self._vocab_size = 10000
        self._hidden_size = 768

    # ==================== 输入处理 ====================

    def encode_text(self, text, max_length=None):
        """编码文本"""
        from transformers import AutoTokenizer

        # 使用gpt2作为默认tokenizer（测试用）
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            # 如果网络不可用，使用mock
            tokenizer = MockTokenizer()

        if isinstance(text, str):
            text = [text]

        outputs = tokenizer(
            text,
            max_length=max_length or 50,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def encode_image(self, images, size=None):
        """编码图像"""
        from torchvision import transforms

        if isinstance(images, str):
            # 从文件加载
            img = Image.open(images)
        elif isinstance(images, np.ndarray):
            # 从numpy数组
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            img = Image.fromarray(images)
        elif isinstance(images, Image.Image):
            img = images
        elif isinstance(images, list):
            # 批量处理
            return torch.stack([self.encode_image(img, size) for img in images])
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        # 调整大小
        if size:
            img = img.resize((size[1], size[0]))

        # 转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        return transform(img).unsqueeze(0)

    def encode_multimodal(self, text, images=None, **kwargs):
        """编码多模态输入"""
        result = self.encode_text(text)

        if images is not None:
            result["pixel_values"] = self.encode_image(images)

        return result

    # ==================== 核心推理 ====================

    def forward(self, input_ids, attention_mask, pixel_values=None, **kwargs):
        """前向传播"""
        batch_size, seq_len = input_ids.shape

        outputs = {
            "logits": torch.randn(batch_size, seq_len, self._vocab_size),
            "hidden_states": torch.randn(batch_size, seq_len, self._hidden_size),
        }

        if pixel_values is not None:
            outputs["vision_features"] = torch.randn(batch_size, 49, self._hidden_size)

        return outputs

    def compute_logprobs(self, input_ids, attention_mask, pixel_values=None):
        """计算log probabilities"""
        outputs = self.forward(input_ids, attention_mask, pixel_values)
        logits = outputs["logits"]

        # 计算log probs
        log_probs = torch.log_softmax(logits, dim=-1)

        # 收集对应token的log prob
        batch_ids = torch.arange(input_ids.shape[0]).unsqueeze(1)
        seq_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        token_log_probs = log_probs[batch_ids, seq_ids, input_ids]

        return token_log_probs

    @torch.no_grad()
    def predict(self, inputs: VLAInput, **generation_kwargs):
        """生成预测"""
        # 编码输入
        if inputs.input_ids is None:
            encoded = self.encode_text(inputs.text)
            inputs.input_ids = encoded["input_ids"]
            inputs.attention_mask = encoded["attention_mask"]

        if inputs.pixel_values is None and inputs.images is not None:
            inputs.pixel_values = self.encode_image(inputs.images)

        # 生成token ids（简化版）
        batch_size = inputs.input_ids.shape[0]
        max_new_tokens = generation_kwargs.get("max_new_tokens", 10)
        token_ids = torch.randint(0, self._vocab_size, (batch_size, max_new_tokens))

        # 计算logprobs
        logprobs = torch.randn(batch_size, max_new_tokens)

        # 生成动作
        actions = torch.randn(batch_size, self._action_dim)

        return VLAPrediction(
            token_ids=token_ids,
            logprobs=logprobs,
            actions=actions,
        )

    # ==================== 模型信息 ====================

    def get_param_groups(self):
        """返回参数组"""
        return {
            "vision_encoder": [],
            "language_model": [],
            "policy_head": [],
        }

    def get_trainable_params(self):
        """返回可训练参数"""
        return []

    def freeze_vision_encoder(self):
        """冻结视觉编码器"""
        pass

    def unfreeze_vision_encoder(self):
        """解冻视觉编码器"""
        pass

    # ==================== 序列化 ====================

    def state_dict(self):
        """保存checkpoint"""
        return {}

    def load_state_dict(self, state_dict):
        """加载checkpoint"""
        pass

    def save_pretrained(self, path):
        """保存到HuggingFace格式"""
        Path(path).mkdir(parents=True, exist_ok=True)

    def from_pretrained(self, path):
        """从HuggingFace格式加载"""
        pass

    # ==================== 属性 ====================

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_space(self):
        return {
            "image": (3, 224, 224),
            "language": "text",
        }


class MockTokenizer:
    """Mock tokenizer用于离线测试"""

    def __init__(self):
        self.vocab_size = 10000

    def __call__(self, text, max_length=50, padding="max_length", truncation=True, return_tensors="pt"):
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
            text = [text]

        # 生成随机token ids
        import torch
        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}


class MockDrivingVLA(MockVLAModel, VLAForDriving):
    """Mock驾驶VLA模型"""

    def __init__(self, config=None):
        super().__init__(config)
        self._num_waypoints = 5

    # ==================== 驾驶特定方法 ====================

    def process_ego_state(self, position, heading, velocity):
        """处理自车状态"""
        position_str = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        heading_deg = heading * 180 / np.pi
        speed = np.linalg.norm(velocity)
        return f"Vehicle at {position_str}, heading {heading_deg:.1f} degrees, speed {speed:.2f} m/s"

    def process_route(self, route):
        """处理路线信息"""
        if not route:
            return "No route specified"

        descriptions = []
        for i, point in enumerate(route):
            cmd = point["command"]
            pos = point["position"]
            descriptions.append(f"{i+1}. {cmd} to ({pos[0]:.1f}, {pos[1]:.1f})")

        return "Route: " + " -> ".join(descriptions)

    def encode_bev_map(self, bev_map):
        """编码鸟瞰图"""
        # 简化：直接展平并投影
        from torch.nn import functional as F

        if isinstance(bev_map, np.ndarray):
            bev_map = torch.from_numpy(bev_map).permute(2, 0, 1).float() / 255.0

        # 调整大小
        bev_map = F.interpolate(
            bev_map.unsqueeze(0),
            size=(200, 200),
            mode="bilinear",
        ).squeeze(0)

        return bev_map

    def predict_waypoints(self, text, images, num_waypoints=5, **generation_kwargs):
        """预测路点"""
        # 生成随机路点用于测试
        waypoints = np.random.randn(num_waypoints, 3)
        # 确保z坐标为0（地面）
        waypoints[:, 2] = 0
        return waypoints

    def waypoints_to_controls(self, waypoints, current_state):
        """路点转控制信号"""
        # 简单的pure pursuit
        target = waypoints[0]
        current_pos = current_state["position"]
        current_heading = current_state["heading"]

        # 计算目标方向
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        target_heading = np.arctan2(dy, dx)

        # 计算转向角
        steer = target_heading - current_heading
        steer = np.clip(steer, -1, 1)

        # 简单速度控制
        velocity = np.linalg.norm(current_state.get("velocity", [0, 0, 0]))
        throttle = 0.5 if velocity < 5.0 else 0.0

        return {
            "throttle": float(throttle),
            "steer": float(steer),
            "brake": 0.0,
        }

    def predict_controls(self, text, images, **generation_kwargs):
        """直接预测控制信号"""
        # 生成随机控制信号
        return {
            "throttle": float(np.random.uniform(0, 1)),
            "steer": float(np.random.uniform(-1, 1)),
            "brake": float(np.random.uniform(0, 1)),
        }


# ==================== 测试 ====================

class TestVLAInterface:
    """测试VLAInterface接口"""

    def test_init(self):
        """测试初始化"""
        model = MockVLAModel()
        assert model.action_dim == 15
        assert "image" in model.observation_space

    def test_encode_text_single(self, sample_text):
        """测试编码单个文本"""
        model = MockVLAModel()
        result = model.encode_text(sample_text)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 1  # batch_size=1

    def test_encode_text_batch(self, sample_batch_text):
        """测试编码文本批次"""
        model = MockVLAModel()
        result = model.encode_text(sample_batch_text)

        assert result["input_ids"].shape[0] == len(sample_batch_text)

    def test_encode_text_with_max_length(self, sample_text):
        """测试指定最大长度"""
        model = MockVLAModel()
        max_length = 20
        result = model.encode_text(sample_text, max_length=max_length)

        assert result["input_ids"].shape[1] == max_length

    def test_encode_image_from_path(self, sample_image_path):
        """测试从文件路径编码图像"""
        model = MockVLAModel()
        result = model.encode_image(sample_image_path)

        assert result.shape[0] in [1, 3]  # channel first
        assert result.shape[1] == 224  # height
        assert result.shape[2] == 224  # width

    def test_encode_image_from_numpy(self):
        """测试从numpy数组编码图像"""
        model = MockVLAModel()
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model.encode_image(img_array)

        assert result.shape == (1, 3, 224, 224)

    def test_encode_image_from_pil(self):
        """测试从PIL Image编码"""
        model = MockVLAModel()
        img = Image.new("RGB", (224, 224), color="red")
        result = model.encode_image(img)

        assert result.shape == (1, 3, 224, 224)

    def test_encode_image_batch(self):
        """测试批量编码图像"""
        model = MockVLAModel()
        images = [Image.new("RGB", (224, 224), color="red") for _ in range(3)]
        result = model.encode_image(images)

        assert result.shape[0] == 3  # batch_size
        assert result.shape[1] == 3  # channels

    def test_encode_image_with_size(self):
        """测试指定图像大小"""
        model = MockVLAModel()
        img = Image.new("RGB", (512, 512), color="blue")
        result = model.encode_image(img, size=(256, 256))

        assert result.shape[2] == 256  # resized height
        assert result.shape[3] == 256  # resized width

    def test_encode_multimodal_text_only(self, sample_text):
        """测试编码仅文本的多模态输入"""
        model = MockVLAModel()
        result = model.encode_multimodal(sample_text)

        assert "input_ids" in result
        assert "pixel_values" not in result

    def test_encode_multimodal_with_image(self, sample_text):
        """测试编码文本+图像的多模态输入"""
        model = MockVLAModel()
        img = Image.new("RGB", (224, 224))
        result = model.encode_multimodal(sample_text, images=img)

        assert "input_ids" in result
        assert "pixel_values" in result

    def test_forward_text_only(self, text_encoder_output):
        """测试仅文本的前向传播"""
        model = MockVLAModel()
        result = model.forward(
            text_encoder_output["input_ids"],
            text_encoder_output["attention_mask"],
        )

        assert "logits" in result
        assert "hidden_states" in result
        assert result["logits"].shape[-1] == model._vocab_size

    def test_forward_with_image(self, text_encoder_output, image_encoder_output):
        """测试带图像的前向传播"""
        model = MockVLAModel()
        result = model.forward(
            text_encoder_output["input_ids"],
            text_encoder_output["attention_mask"],
            pixel_values=image_encoder_output,
        )

        assert "vision_features" in result

    def test_compute_logprobs(self, text_encoder_output):
        """测试计算log probabilities"""
        model = MockVLAModel()
        logprobs = model.compute_logprobs(
            text_encoder_output["input_ids"],
            text_encoder_output["attention_mask"],
        )

        assert logprobs.shape == text_encoder_output["input_ids"].shape
        # log probs应该是负数
        assert (logprobs < 0).any()

    def test_predict_basic(self, sample_text):
        """测试基本预测"""
        model = MockVLAModel()
        inputs = VLAInput(text=sample_text)
        result = model.predict(inputs, max_new_tokens=10)

        assert isinstance(result, VLAPrediction)
        assert result.token_ids is not None
        assert result.actions is not None
        assert result.token_ids.shape[1] == 10

    def test_predict_with_image(self, sample_text):
        """测试带图像的预测"""
        model = MockVLAModel()
        img = Image.new("RGB", (224, 224))
        inputs = VLAInput(text=sample_text, images=img)
        result = model.predict(inputs, max_new_tokens=10)

        assert result.token_ids is not None

    def test_get_param_groups(self):
        """测试获取参数组"""
        model = MockVLAModel()
        groups = model.get_param_groups()

        assert "vision_encoder" in groups
        assert "language_model" in groups
        assert "policy_head" in groups

    def test_get_trainable_params(self):
        """测试获取可训练参数"""
        model = MockVLAModel()
        params = model.get_trainable_params()

        assert isinstance(params, list)

    def test_freeze_unfreeze_vision_encoder(self):
        """测试冻结/解冻视觉编码器"""
        model = MockVLAModel()
        # 应该不抛出异常
        model.freeze_vision_encoder()
        model.unfreeze_vision_encoder()

    def test_state_dict(self):
        """测试保存state dict"""
        model = MockVLAModel()
        state = model.state_dict()

        assert isinstance(state, dict)

    def test_load_state_dict(self):
        """测试加载state dict"""
        model = MockVLAModel()
        state = {"mock": "state"}

        # 应该不抛出异常
        model.load_state_dict(state)

    def test_save_pretrained(self, tmp_path):
        """测试保存模型"""
        model = MockVLAModel()
        save_path = str(tmp_path / "test_model")

        model.save_pretrained(save_path)

        # 检查目录创建
        assert Path(save_path).exists()

    def test_action_dim_property(self):
        """测试action_dim属性"""
        model = MockVLAModel()
        assert model.action_dim == 15

    def test_observation_space_property(self):
        """测试observation_space属性"""
        model = MockVLAModel()
        space = model.observation_space

        assert "image" in space
        assert space["image"] == (3, 224, 224)


class TestVLAForDriving:
    """测试VLAForDriving接口"""

    def test_init(self):
        """测试初始化"""
        model = MockDrivingVLA()
        assert isinstance(model, VLAInterface)
        assert isinstance(model, VLAForDriving)

    def test_process_ego_state(self, sample_driving_state):
        """测试处理自车状态"""
        model = MockDrivingVLA()
        text = model.process_ego_state(
            sample_driving_state["position"],
            sample_driving_state["heading"],
            sample_driving_state["velocity"],
        )

        assert isinstance(text, str)
        assert "Vehicle at" in text
        assert "heading" in text.lower()
        assert "speed" in text.lower()

    def test_process_ego_state_format(self):
        """测试自车状态格式化"""
        model = MockDrivingVLA()
        text = model.process_ego_state(
            np.array([10.5, -3.2, 0.0]),
            0.78,  # ~45 degrees
            np.array([5.0, 0.2, 0.0]),
        )

        # 检查数值格式化（应该有两位小数）
        assert "(10.50, -3.20, 0.00)" in text or "(10.5, -3.2, 0.0)" in text

    def test_process_route(self, sample_route):
        """测试处理路线"""
        model = MockDrivingVLA()
        text = model.process_route(sample_route)

        assert isinstance(text, str)
        assert "Route" in text or "route" in text.lower()

    def test_process_route_empty(self):
        """测试处理空路线"""
        model = MockDrivingVLA()
        text = model.process_route([])

        assert "No route" in text or "empty" in text.lower()

    def test_encode_bev_map(self):
        """测试编码BEV地图"""
        model = MockDrivingVLA()
        bev_map = np.random.randn(200, 200, 3)

        result = model.encode_bev_map(bev_map)

        assert result.shape[0] == 3  # channels
        assert result.shape[1] == 200  # height
        assert result.shape[2] == 200  # width

    def test_predict_waypoints(self):
        """测试预测路点"""
        model = MockDrivingVLA()
        text = "Drive forward"
        images = torch.randn(1, 3, 224, 224)

        waypoints = model.predict_waypoints(text, images, num_waypoints=5)

        assert waypoints.shape == (5, 3)
        # 路点应该在地面上
        assert np.allclose(waypoints[:, 2], 0)

    def test_predict_waypoints_custom_num(self):
        """测试预测指定数量的路点"""
        model = MockDrivingVLA()
        text = "Turn left"
        images = torch.randn(1, 3, 224, 224)

        waypoints = model.predict_waypoints(text, images, num_waypoints=10)

        assert waypoints.shape == (10, 3)

    def test_waypoints_to_controls(self, sample_driving_state, sample_waypoints):
        """测试路点转控制信号"""
        model = MockDrivingVLA()
        controls = model.waypoints_to_controls(sample_waypoints, sample_driving_state)

        assert "throttle" in controls
        assert "steer" in controls
        assert "brake" in controls

        # 检查范围
        assert 0 <= controls["throttle"] <= 1
        assert -1 <= controls["steer"] <= 1
        assert 0 <= controls["brake"] <= 1

    def test_waypoints_to_controls_stationary(self):
        """测试静态车辆的路点转控制"""
        model = MockDrivingVLA()
        waypoints = np.array([
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
        ])

        state = {
            "position": np.array([0.0, 0.0, 0.0]),
            "heading": 0.0,
            "velocity": np.array([0.0, 0.0, 0.0]),
        }

        controls = model.waypoints_to_controls(waypoints, state)

        # 应该向前行驶
        assert controls["throttle"] > 0
        assert abs(controls["steer"]) < 0.1  # 直行
        assert controls["brake"] == 0

    def test_waypoints_to_controls_turn_left(self):
        """测试左转的路点转控制"""
        model = MockDrivingVLA()
        waypoints = np.array([
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
        ])

        state = {
            "position": np.array([0.0, 0.0, 0.0]),
            "heading": 0.0,
            "velocity": np.array([0.0, 0.0, 0.0]),
        }

        controls = model.waypoints_to_controls(waypoints, state)

        # 应该左转
        assert controls["steer"] > 0

    def test_predict_controls(self):
        """测试直接预测控制信号"""
        model = MockDrivingVLA()
        text = "Drive forward and turn left"
        images = torch.randn(1, 3, 224, 224)

        controls = model.predict_controls(text, images)

        assert "throttle" in controls
        assert "steer" in controls
        assert "brake" in controls

        # 检查类型
        assert isinstance(controls["throttle"], float)
        assert isinstance(controls["steer"], float)
        assert isinstance(controls["brake"], float)


# ==================== 边界情况测试 ====================

class TestVLAInterfaceEdgeCases:
    """测试边界情况"""

    def test_encode_empty_text(self):
        """测试编码空文本"""
        model = MockVLAModel()
        result = model.encode_text("")

        assert "input_ids" in result

    def test_encode_very_long_text(self):
        """测试编码非常长的文本"""
        model = MockVLAModel()
        long_text = "word " * 1000

        result = model.encode_text(long_text, max_length=128)

        # 应该被截断
        assert result["input_ids"].shape[1] == 128

    def test_encode_invalid_image_path(self):
        """测试编码不存在的图像路径"""
        model = MockVLAModel()

        with pytest.raises(Exception):
            model.encode_image("/nonexistent/path/image.png")

    def test_encode_invalid_image_array(self):
        """测试编码无效的图像数组"""
        model = MockVLAModel()

        # 错误的通道数
        with pytest.raises(Exception):
            model.encode_image(np.random.randn(224, 224))

    def test_forward_empty_batch(self):
        """测试空批次的前向传播"""
        model = MockVLAModel()

        with pytest.raises(Exception):
            model.forward(
                torch.empty(0, 10, dtype=torch.long),
                torch.empty(0, 10, dtype=torch.long),
            )

    def test_predict_with_vla_input_pre_encoded(self, text_encoder_output, image_encoder_output):
        """测试预测时使用预编码的输入"""
        model = MockVLAModel()
        inputs = VLAInput(
            text="test",
            input_ids=text_encoder_output["input_ids"],
            attention_mask=text_encoder_output["attention_mask"],
            pixel_values=image_encoder_output,
        )

        result = model.predict(inputs, max_new_tokens=5)

        assert result.token_ids is not None
        assert result.token_ids.shape[1] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
