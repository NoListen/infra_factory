# VLA模型整合接口指南

## 1. 框架接口要求对比

### 1.1 AReaL 接口要求

#### 核心接口

```python
# areal/api/engine_api.py
class TrainEngine(ABC):
    @abc.abstractmethod
    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[[torch.Tensor, dict], torch.Tensor | None],
        forward_only: bool = False,
    ) -> None:
        """
        处理micro-batch的前向和反向传播

        参数:
            mb_list: micro-batch列表
            process_output_fn: 处理模型输出的函数，计算loss
            forward_only: 是否只做前向传播
        """

    @abc.abstractmethod
    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict], torch.Tensor],
    ) -> dict[str, float]:
        """
        训练一个批次

        参数:
            input_: 输入数据，可能包含:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - pixel_values: [batch, num_frames, C, H, W]  # VLA特有
                - images: [batch, C, H, W]  # 单帧图像
            loss_fn: 损失函数，接收 (logprobs, input_data)
            loss_weight_fn: 计算每个micro-batch的权重
        """

class InferenceEngine(ABC):
    @abc.abstractmethod
    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """
        异步生成响应

        参数:
            req: ModelRequest包含:
                - input_ids: token ids
                - pixel_values: 图像张量
                - images: 图像列表
                - max_tokens: 最大生成长度
                - temperature: 采样温度
                - ...

        返回:
            ModelResponse包含:
                - token_ids: 生成的token ids
                - logprobs: log probabilities
                - ...
        """

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """
        异步更新权重

        参数:
            meta: 权重更新元数据
            param_specs: 参数规范列表
        """
```

#### 模型适配器模式

```python
# areal/models/your_vla_adapter.py
from areal.api.io_struct import ModelRequest, ModelResponse

class YourVLAModelAdapter:
    """VLA模型适配器示例"""

    def __init__(self, model_config, engine_config):
        self.model = self._load_model(model_config)
        self.tokenizer = self._load_tokenizer(model_config)
        self.image_processor = self._load_image_processor(model_config)

    def _load_model(self, config):
        """加载VLA模型"""
        # 返回你的模型实例
        pass

    def _load_tokenizer(self, config):
        """加载tokenizer"""
        pass

    def _load_image_processor(self, config):
        """加载图像处理器"""
        pass

    def forward(self, batch: dict) -> dict:
        """
        前向传播

        输入batch格式:
        {
            "input_ids": [batch, seq_len],
            "attention_mask": [batch, seq_len],
            "pixel_values": [batch, C, H, W],  # 可选
            "images": list[PIL.Image],  # 可选
        }

        返回格式:
        {
            "logits": [batch, seq_len, vocab_size],
            "hidden_states": [batch, seq_len, hidden_dim],  # 可选
        }
        """
        # 1. 处理图像输入
        if "pixel_values" in batch:
            vision_features = self.model.vision_encoder(batch["pixel_values"])
        else:
            vision_features = None

        # 2. 处理文本输入
        text_outputs = self.model.language_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            vision_features=vision_features,
        )

        return text_outputs

    def compute_logprobs(self, batch: dict) -> torch.Tensor:
        """
        计算log probabilities

        返回: [batch, seq_len] 的log probs
        """
        outputs = self.forward(batch)
        logits = outputs["logits"]

        # 计算log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # 收集对应token的log prob
        input_ids = batch["input_ids"]
        batch_ids = torch.arange(input_ids.shape[0]).unsqueeze(1)
        seq_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        log_probs = log_probs[batch_ids, seq_ids, input_ids]

        return log_probs

    @torch.no_grad()
    def generate(self, batch: dict, generation_kwargs: dict) -> dict:
        """
        生成动作/文本

        generation_kwargs包含:
        - max_new_tokens: 最大生成长度
        - temperature: 采样温度
        - top_p: nucleus sampling参数
        - ...
        """
        # 处理图像
        if "pixel_values" in batch:
            vision_features = self.model.vision_encoder(batch["pixel_values"])

        # 生成
        outputs = self.model.language_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            vision_features=vision_features,
            **generation_kwargs,
        )

        return {
            "token_ids": outputs["sequences"],
            "logprobs": outputs.get("logprobs"),
        }

    def get_param_specs(self) -> list[ParamSpec]:
        """
        获取参数规范，用于分布式权重更新

        返回格式:
        [
            ParamSpec(name="vision_encoder", shape=[...], dtype=...),
            ParamSpec(name="language_model", shape=[...], dtype=...),
            ParamSpec(name="policy_head", shape=[...], dtype=...),
        ]
        """
        from areal.api.io_struct import ParamSpec

        specs = []
        for name, param in self.model.named_parameters():
            specs.append(ParamSpec(
                name=name,
                shape=list(param.shape),
                dtype=str(param.dtype),
            ))
        return specs
```

### 1.2 RLinf 接口要求

#### 配置驱动方式

```python
# rlinf/config.py
class SupportedModel(Enum):
    # 添加你的VLA模型
    YOUR_VLA = ("your_vla", "embodied")

# 注册模型工厂
def get_your_vla_model(cfg):
    from your_vla_model import YourVLAModel
    return YourVLAModel(cfg.model.model_path)
```

#### 配置文件格式

```yaml
# config/your_vla_grpo.yaml
runner:
  task_type: "embodied"

actor:
  training_backend: "fsdp"  # 或 "megatron"
  model:
    model_type: "your_vla"  # 使用注册的模型名称
    model_path: "/path/to/your/model"

    # VLA特定配置
    vision_encoder:
      type: "your_encoder"  # 如 "clip", "siglip"
      pretrained_path: "/path/to/vision_encoder"
      freeze: false  # 是否冻结视觉编码器

    language_model:
      base_model: "qwen2.5-7b"  # 或其他LLM
      lora_r: 64  # 如果使用LoRA
      lora_alpha: 16

    policy_head:
      type: "gaussian"  # 或 "deterministic"
      action_dim: 7  # 根据任务定义
      num_action_chunks: 10  # 动作chunk数量

rollout:
  rollout_backend: "sglang"  # 或 "vllm"
  model:
    model_type: "your_vla"
    model_path: "/path/to/your/model"

env:
  train:
    env_type: "navsim"  # 或 "bench2drive"
    total_num_envs: 1000
    max_steps_per_rollout_epoch: 100

algorithm:
  loss_type: "actor_critic"  # PPO
  adv_type: "ppo"
  group_size: 4  # GRPO group size
```

#### 模型实现模板

```python
# rlinf/models/your_vla_model.py
import torch
import torch.nn as nn
from rlinf.models import register_model

@register_model("your_vla")
class YourVLAModel(nn.Module):
    """RLinf VLA模型实现模板"""

    def __init__(self, cfg):
        super().__init__()

        # 1. 视觉编码器
        self.vision_encoder = self._build_vision_encoder(cfg.model.vision_encoder)

        # 2. 语言模型
        self.language_model = self._build_language_model(cfg.model.language_model)

        # 3. 策略头（VLA特有）
        self.policy_head = self._build_policy_head(cfg.model.policy_head)

        # 4. 价值头（如果使用actor-critic）
        if cfg.algorithm.loss_type == "actor_critic":
            self.value_head = nn.Linear(
                self.language_model.config.hidden_size,
                1
            )

    def _build_vision_encoder(self, cfg):
        """构建视觉编码器"""
        # 示例：使用CLIP
        from transformers import CLIPVisionModel
        encoder = CLIPVisionModel.from_pretrained(cfg.pretrained_path)

        if cfg.get("freeze", False):
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    def _build_language_model(self, cfg):
        """构建语言模型"""
        # 示例：使用Qwen2.5
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model)

        # 添加LoRA（如果配置）
        if cfg.get("lora_r", 0) > 0:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)

        return model

    def _build_policy_head(self, cfg):
        """构建策略头"""
        return nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim * cfg.num_action_chunks),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        images=None,
    ):
        """
        前向传播

        参数:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            pixel_values: [batch, num_frames, C, H, W] 或 [batch, C, H, W]
            images: PIL.Image列表（可选）

        返回:
            dict包含:
            - logits: [batch, seq_len, vocab_size]
            - actions: [batch, num_action_chunks, action_dim]（VLA特有）
            - values: [batch, seq_len, 1]（如果有价值头）
        """
        outputs = {}

        # 1. 处理视觉输入
        if pixel_values is not None:
            vision_features = self.vision_encoder(pixel_values)
            # 将视觉特征投影到语言模型空间
            vision_embeds = self.project_vision_features(vision_features)
        else:
            vision_embeds = None

        # 2. 语言模型前向
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_embeds,
            output_hidden_states=True,
        )

        outputs["logits"] = lm_outputs.logits
        hidden_states = lm_outputs.hidden_states[-1]

        # 3. 策略头（生成动作）
        actions = self.policy_head(hidden_states)
        actions = actions.view(
            -1, self.policy_head.num_action_chunks, self.policy_head.action_dim
        )
        outputs["actions"] = actions

        # 4. 价值头（如果有）
        if hasattr(self, "value_head"):
            values = self.value_head(hidden_states)
            outputs["values"] = values

        return outputs

    def generate_actions(self, batch, **generation_kwargs):
        """生成动作（用于推理）"""
        with torch.no_grad():
            outputs = self.forward(**batch)
            actions = outputs["actions"]
            return actions
```

### 1.3 Verl 接口要求

#### DataProto 驱动

```python
# verl/protocol.py
from verl.protocol import DataProto
from tensordict import TensorDict

def prepare_vla_batch(images, text, actions, rewards):
    """
    准备VLA训练批次

    参数:
        images: list of PIL.Image 或 torch.Tensor [batch, C, H, W]
        text: list of str 或 tokenized [batch, seq_len]
        actions: torch.Tensor [batch, action_dim]
        rewards: list of float

    返回:
        DataProto对象
    """
    # 1. 处理图像
    if isinstance(images, list):
        pixel_values = torch.stack([
            image_processor(img) for img in images
        ])
    else:
        pixel_values = images

    # 2. 处理文本
    if isinstance(text, list):
        text_inputs = tokenizer(text, return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
    else:
        input_ids = text
        attention_mask = torch.ones_like(text)

    # 3. 构建DataProto
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "actions": actions,
        },
        non_tensors={
            "rewards": np.array(rewards),
            "text": text if isinstance(text, list) else None,
        },
        meta_info={
            "task_type": "vla_training",
        }
    )
```

#### Verl模型包装器

```python
# verl/models/your_vla_model.py
import torch
import torch.nn as nn
from verl.protocol import DataProto

class YourVLAForVerl(nn.Module):
    """Verl VLA模型包装器"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 你的VLA模型

    def forward(self, batch: DataProto) -> DataProto:
        """
        前向传播

        参数:
            batch: DataProto，包含:
                - input_ids
                - attention_mask
                - pixel_values
                - ...

        返回:
            DataProto，包含:
                - logits
                - actions
                - values（如果有）
        """
        # 从DataProto提取数据
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        pixel_values = batch.batch.get("pixel_values", None)

        # 调用你的模型
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        # 构建输出DataProto
        return DataProto.from_dict(
            tensors={
                "logits": outputs["logits"],
                "actions": outputs.get("actions", None),
            }
        )

    def generate(self, batch: DataProto, **kwargs) -> DataProto:
        """生成"""
        with torch.no_grad():
            outputs = self.base_model.generate_actions(
                input_ids=batch.batch["input_ids"],
                pixel_values=batch.batch.get("pixel_values", None),
                **kwargs
            )

        return DataProto.from_dict(
            tensors={
                "actions": outputs,
            }
        )
```

## 2. 通用VLA接口设计

### 2.1 最小接口规范

```python
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List
import torch
import numpy as np
from PIL import Image

class VLAInterface(ABC):
    """
    VLA模型的最小接口规范

    所有VLA模型应该实现这个接口以适配不同训练框架
    """

    # ==================== 输入处理 ====================

    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        编码文本输入

        参数:
            text: 文本字符串或列表
            max_length: 最大长度（截断/填充）

        返回:
            {
                "input_ids": [batch, seq_len],
                "attention_mask": [batch, seq_len],
            }
        """
        pass

    @abstractmethod
    def encode_image(
        self,
        images: Union[str, np.ndarray, Image.Image, List],
        size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        编码图像输入

        参数:
            images:
                - str: 文件路径
                - np.ndarray: [H, W, C] 或 [C, H, W]
                - PIL.Image
                - List: 批量图像
            size: (height, width) 调整大小

        返回:
            pixel_values: [batch, C, H, W] 或 [batch, num_frames, C, H, W]
        """
        pass

    @abstractmethod
    def encode_multimodal(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[str, np.ndarray, Image.Image, List]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        编码多模态输入

        返回包含所有必要字段的字典
        """
        pass

    # ==================== 核心推理 ====================

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            pixel_values: [batch, C, H, W] 或 [batch, num_frames, C, H, W]

        返回:
            {
                "logits": [batch, seq_len, vocab_size],
                "hidden_states": [batch, seq_len, hidden_dim] (可选),
                "vision_features": (可选),
            }
        """
        pass

    @abstractmethod
    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算token级别的log probabilities

        返回:
            log_probs: [batch, seq_len]
        """
        pass

    @torch.no_grad()
    @abstractmethod
    def generate_actions(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[torch.Tensor, List]] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        生成动作（VLA的核心功能）

        参数:
            text: 指令文本
            images: 观察图像
            generation_kwargs:
                - max_new_tokens: 最大生成长度
                - temperature: 采样温度
                - top_k, top_p: 采样参数
                - num_return_sequences: 生成数量

        返回:
            actions: [batch, action_dim] 或 [batch, num_chunks, action_dim]
        """
        pass

    # ==================== 模型信息 ====================

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """
        返回参数组（用于不同学习率/优化器设置）

        返回:
            {
                "vision_encoder": [params...],
                "language_model": [params...],
                "policy_head": [params...],
            }
        """
        pass

    @abstractmethod
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """返回所有可训练参数"""
        pass

    @abstractmethod
    def freeze_vision_encoder(self):
        """冻结视觉编码器参数"""
        pass

    @abstractmethod
    def unfreeze_vision_encoder(self):
        """解冻视觉编码器参数"""
        pass

    # ==================== 序列化 ====================

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """保存checkpoint"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """加载checkpoint"""
        pass

    @abstractmethod
    def save_pretrained(self, path: str):
        """保存到HuggingFace格式"""
        pass

    @abstractmethod
    def from_pretrained(self, path: str):
        """从HuggingFace格式加载"""
        pass

    # ==================== VLA特定 ====================

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """动作空间维度"""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, tuple]:
        """
        观察空间规范

        返回:
            {
                "image": (C, H, W),
                "language": "text",  # 或 (seq_len,)
            }
        """
        pass

    def action_to_controls(self, actions: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        将模型输出转换为实际控制信号

        例如:
        - 对于导航: {"throttle": float, "steer": float, "brake": float}
        - 对于机械臂: {"joint_positions": [7], "gripper": float}

        参数:
            actions: [batch, action_dim]

        返回:
            controls: 每个样本的控制字典
        """
        raise NotImplementedError("子类需要实现此方法")
```

### 2.2 导航驾驶专用接口

```python
class VLAForDriving(VLAInterface):
    """
    专门用于导航驾驶的VLA接口

    支持 NavSim 和 Bench2Drive
    """

    @abstractmethod
    def process_ego_state(
        self,
        position: np.ndarray,  # [3] x, y, z
        heading: float,  # 弧度
        velocity: np.ndarray,  # [3] vx, vy, vz
    ) -> str:
        """
        将自车状态转换为文本描述

        示例:
        "The vehicle is at position (x=10.5, y=-3.2) heading 45 degrees at speed 5.2 m/s"
        """
        pass

    @abstractmethod
    def process_route(
        self,
        route: List[Dict],  # [{"position": [x, y, z], "command": "LANE_FOLLOW"}, ...]
    ) -> str:
        """
        将路线信息转换为文本描述

        示例:
        "Route: Go straight for 100m, then turn left at the intersection..."
        """
        pass

    @abstractmethod
    def encode_bev_map(
        self,
        bev_map: np.ndarray,  # [H, W, C] 鸟瞰图
    ) -> torch.Tensor:
        """
        编码鸟瞰图（可选的视觉输入）

        返回:
            bev_features: [batch, C', H', W']
        """
        pass

    @abstractmethod
    def predict_waypoints(
        self,
        text: str,
        images: torch.Tensor,
        num_waypoints: int = 5,
    ) -> np.ndarray:
        """
        预测未来路点（VLA用于驾驶的典型输出）

        返回:
            waypoints: [num_waypoints, 3] 每个路点的(x, y, z)坐标
        """
        pass

    def waypoints_to_controls(
        self,
        waypoints: np.ndarray,
        current_state: Dict,
    ) -> Dict[str, float]:
        """
        将路点转换为控制信号

        典型实现:
        1. 计算到第一个路点的方向
        2. 使用Pure Pursuit或Stanley控制器
        3. 返回油门、转向、刹车

        返回:
            {
                "throttle": float,  # [0, 1]
                "steer": float,     # [-1, 1]
                "brake": float,     # [0, 1]
            }
        """
        # 默认实现（可覆盖）
        from typing import Dict
        import numpy as np

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

        # 简单的速度控制
        velocity = np.linalg.norm(current_state["velocity"])
        throttle = 0.5 if velocity < 5.0 else 0.0

        return {
            "throttle": float(throttle),
            "steer": float(steer),
            "brake": 0.0,
        }
```

## 3. 模型注册机制

### 3.1 统一注册器

```python
# vla_models/model_registry.py
from typing import Dict, Type, Callable
import logging

logger = logging.getLogger(__name__)

class VLAModelRegistry:
    """VLA模型注册器"""

    _models: Dict[str, Type[VLAInterface]] = {}
    _adapters: Dict[str, Dict[str, Callable]] = {
        "areal": {},
        "rlinf": {},
        "verl": {},
    }

    @classmethod
    def register_model(cls, name: str, model_class: Type[VLAInterface]):
        """注册VLA模型"""
        cls._models[name] = model_class
        logger.info(f"Registered VLA model: {name}")

    @classmethod
    def register_adapter(cls, framework: str, model_name: str, adapter_fn: Callable):
        """注册框架适配器"""
        cls._adapters[framework][model_name] = adapter_fn
        logger.info(f"Registered {framework} adapter for {model_name}")

    @classmethod
    def get_model(cls, name: str, **kwargs) -> VLAInterface:
        """获取模型实例"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name](**kwargs)

    @classmethod
    def get_adapter(cls, framework: str, model_name: str):
        """获取框架适配器"""
        if framework not in cls._adapters:
            raise ValueError(f"Unknown framework: {framework}")
        if model_name not in cls._adapters[framework]:
            raise ValueError(f"No {framework} adapter for {model_name}")
        return cls._adapters[framework][model_name]

    @classmethod
    def list_models(cls) -> list[str]:
        """列出所有注册的模型"""
        return list(cls._models.keys())

    @classmethod
    def list_frameworks(cls) -> list[str]:
        """列出所有支持的框架"""
        return list(cls._adapters.keys())


# 装饰器
def register_vla_model(name: str):
    """注册VLA模型的装饰器"""
    def decorator(cls: Type[VLAInterface]):
        VLAModelRegistry.register_model(name, cls)
        return cls
    return decorator


def register_framework_adapter(framework: str, model_name: str):
    """注册框架适配器的装饰器"""
    def decorator(fn: Callable):
        VLAModelRegistry.register_adapter(framework, model_name, fn)
        return fn
    return decorator
```

### 3.2 使用示例

```python
# vla_models/your_driving_vla.py
from vla_models.model_registry import register_vla_model, register_framework_adapter
from vla_models.interface import VLAForDriving

@register_vla_model("your_driving_vla")
class YourDrivingVLA(VLAForDriving):
    """你的驾驶VLA模型"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.model = self._load_model(model_path)

    # 实现所有抽象方法...

# AReaL适配器
@register_framework_adapter("areal", "your_driving_vla")
def areal_adapter(model_config, engine_config):
    from your_driving_vla import YourDrivingVLA
    from areal.models import YourVLAForAReaL

    base_model = YourDrivingVLA(**model_config)
    return YourVLAForAReaL(base_model, **engine_config)

# RLinf适配器
@register_framework_adapter("rlinf", "your_driving_vla")
def rlinf_adapter(cfg):
    from your_driving_vla import YourDrivingVLA
    from rlinf.models import YourVLAForRLinf

    base_model = YourDrivingVLA(model_path=cfg.model.model_path)
    return YourVLAForRLinf(base_model, cfg)

# Verl适配器
@register_framework_adapter("verl", "your_driving_vla")
def verl_adapter(cfg):
    from your_driving_vla import YourDrivingVLA
    from verl.models import YourVLAForVerl

    base_model = YourDrivingVLA(model_path=cfg.model.model_path)
    return YourVLAForVerl(base_model)

# 使用
from vla_models.model_registry import VLAModelRegistry

# 获取模型
model = VLAModelRegistry.get_model("your_driving_vla", model_path="/path/to/model")

# 获取框架适配器
areal_adapter_fn = VLAModelRegistry.get_adapter("areal", "your_driving_vla")
areal_model = areal_adapter_fn(model_config, engine_config)
```

## 4. 配置管理

### 4.1 统一配置格式

```yaml
# vla_models/configs/your_driving_vla.yaml
model:
  name: "your_driving_vla"
  version: "1.0.0"

  # 模型路径
  checkpoint_path: "/path/to/checkpoint"

  # 视觉编码器
  vision_encoder:
    type: "your_vision_encoder"  # clip, siglip, dinov2, etc.
    pretrained_path: "/path/to/vision_encoder"
    freeze: false
    projection_dim: 768

    # 输入配置
    image_size: [224, 224]
    num_frames: 1  # 多帧支持

  # 语言模型
  language_model:
    base_model: "qwen2.5-3b"  # 或其他LLM
    lora_enabled: true
    lora_r: 64
    lora_alpha: 16
    lora_dropout: 0.1

  # 策略头（VLA特有）
  policy_head:
    type: "waypoint"  # waypoint, direct_control, hybrid
    action_dim: 15  # 3 * num_waypoints (for waypoint prediction)
    hidden_dim: 512
    num_layers: 2
    activation: "relu"
    output_activation: "tanh"  # 或 "none"

  # 价值头（可选，用于actor-critic）
  value_head:
    enabled: true
    hidden_dim: 256

# 训练配置
training:
  # 优化器
  optimizer:
    type: "adamw"
    lr: 1e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]

  # 学习率调度
  lr_scheduler:
    type: "cosine"
    warmup_ratio: 0.1
    min_lr_ratio: 0.1

  # 参数组
  param_groups:
    vision_encoder:
      lr: 1e-5  # 视觉编码器使用更小的学习率
    language_model:
      lr: 1e-4
    policy_head:
      lr: 1e-3  # 策略头使用更大的学习率
    value_head:
      lr: 1e-3

# 环境配置
environment:
  type: "navsim"  # navsim, bench2drive

  # 观察空间
  observation:
    camera:
      enabled: true
      width: 224
      height: 224
      fov: 90

    bev_map:
      enabled: false
      width: 200
      height: 200
      resolution: 0.5  # 米/像素

    state:
      enabled: true
      include_velocity: true
      include_heading: true

  # 动作空间
  action:
    type: "waypoint"  # waypoint, continuous
    num_waypoints: 5
    waypoint_horizon: 3.0  # 秒

    # 对于continuous类型
    throttle_range: [0, 1]
    steer_range: [-1, 1]
    brake_range: [0, 1]

# 奖励函数
reward:
  type: "driving"
  weights:
    collision: -10.0
    success: 10.0
    progress: 1.0
    lane_keeping: 0.5
    comfort: 0.1
```

### 4.2 配置加载器

```python
# vla_models/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path

@dataclass
class VisionEncoderConfig:
    type: str
    pretrained_path: str
    freeze: bool = False
    projection_dim: int = 768
    image_size: tuple = (224, 224)
    num_frames: int = 1

@dataclass
class LanguageModelConfig:
    base_model: str
    lora_enabled: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class PolicyHeadConfig:
    type: str = "waypoint"
    action_dim: int = 15
    hidden_dim: int = 512
    num_layers: int = 2
    activation: str = "relu"
    output_activation: str = "tanh"

@dataclass
class ValueHeadConfig:
    enabled: bool = True
    hidden_dim: int = 256

@dataclass
class ModelConfig:
    name: str
    version: str = "1.0.0"
    checkpoint_path: Optional[str] = None
    vision_encoder: Optional[VisionEncoderConfig] = None
    language_model: Optional[LanguageModelConfig] = None
    policy_head: Optional[PolicyHeadConfig] = None
    value_head: Optional[ValueHeadConfig] = None

@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

@dataclass
class ParamGroupConfig:
    vision_encoder: float = 1e-5
    language_model: float = 1e-4
    policy_head: float = 1e-3
    value_head: float = 1e-3

@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: Dict[str, Any] = field(default_factory=dict)
    param_groups: ParamGroupConfig = field(default_factory=ParamGroupConfig)

@dataclass
class EnvironmentConfig:
    type: str = "navsim"
    observation: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardConfig:
    type: str = "driving"
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class VLAConfig:
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "VLAConfig":
        """从YAML文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)

        # 递归构建配置对象
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "VLAConfig":
        """从字典构建配置"""
        # 这里需要实现递归构建逻辑
        # 简化版本：
        model_config = ModelConfig(**data["model"])
        training_config = TrainingConfig(**data.get("training", {}))
        env_config = EnvironmentConfig(**data.get("environment", {}))
        reward_config = RewardConfig(**data.get("reward", {}))

        return cls(
            model=model_config,
            training=training_config,
            environment=env_config,
            reward=reward_config,
        )
```
