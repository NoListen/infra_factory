# vla_models/interface.py
"""
VLA模型统一接口定义

定义所有VLA模型应该实现的核心接口，确保可以在不同训练框架中使用。
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


@dataclass
class VLAPrediction:
    """
    VLA模型预测结果

    统一的预测输出格式
    """
    token_ids: torch.Tensor  # [batch, seq_len] 生成的token ids
    logprobs: Optional[torch.Tensor] = None  # [batch, seq_len] log probabilities
    actions: Optional[torch.Tensor] = None  # [batch, action_dim] 或 [batch, num_chunks, action_dim]
    waypoints: Optional[torch.Tensor] = None  # [batch, num_waypoints, 3] 驾驶路点
    control_signals: Optional[Dict[str, torch.Tensor]] = None  # 直接控制信号
    hidden_states: Optional[torch.Tensor] = None  # [batch, seq_len, hidden_dim]


@dataclass
class VLAInput:
    """
    VLA模型输入

    统一的输入格式
    """
    text: Union[str, List[str]]
    images: Optional[Union[str, np.ndarray, Image.Image, List]] = None
    input_ids: Optional[torch.Tensor] = None  # [batch, seq_len]
    attention_mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    pixel_values: Optional[torch.Tensor] = None  # [batch, C, H, W]
    ego_state: Optional[Dict[str, Any]] = None  # 自车状态（驾驶场景）
    route_info: Optional[List[Dict]] = None  # 路线信息（驾驶场景）


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
        size: Optional[Tuple[int, int]] = None,
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
    def predict(
        self,
        inputs: VLAInput,
        **generation_kwargs
    ) -> VLAPrediction:
        """
        生成预测（统一接口）

        参数:
            inputs: VLAInput对象
            generation_kwargs:
                - max_new_tokens: 最大生成长度
                - temperature: 采样温度
                - top_k, top_p: 采样参数
                - do_sample: 是否采样
                - num_return_sequences: 生成数量

        返回:
            VLAPrediction对象
        """
        pass

    # ==================== 模型信息 ====================

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
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
    def get_trainable_params(self) -> List[nn.Parameter]:
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
    def observation_space(self) -> Dict[str, Tuple]:
        """
        观察空间规范

        返回:
            {
                "image": (C, H, W),
                "language": "text",  # 或 (seq_len,)
            }
        """
        pass

    def action_to_controls(
        self,
        actions: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        将模型输出转换为实际控制信号

        例如:
        - 对于导航: {"throttle": float, "steer": float, "brake": float}
        - 对于机械臂: {"joint_positions": [7], "gripper": float}

        参数:
            actions: [batch, action_dim]

        返回:
            controls: 每个样本的控制字典列表
        """
        raise NotImplementedError("子类需要实现此方法")


class VLAForDriving(VLAInterface):
    """
    专门用于导航驾驶的VLA接口

    支持 NavSim 和 Bench2Drive 等驾驶模拟器
    """

    # ==================== 驾驶特定方法 ====================

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
        "The vehicle is at position (x=10.5, y=-3.2, z=0.0) heading 45 degrees (0.78 rad) at speed 5.2 m/s"
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
        **generation_kwargs
    ) -> np.ndarray:
        """
        预测未来路点（VLA用于驾驶的典型输出）

        返回:
            waypoints: [num_waypoints, 3] 每个路点的(x, y, z)坐标
        """
        pass

    @abstractmethod
    def waypoints_to_controls(
        self,
        waypoints: np.ndarray,
        current_state: Dict[str, Any],
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
        pass

    @abstractmethod
    def predict_controls(
        self,
        text: str,
        images: torch.Tensor,
        **generation_kwargs
    ) -> Dict[str, float]:
        """
        直接预测控制信号（另一种驾驶输出）

        返回:
            {
                "throttle": float,
                "steer": float,
                "brake": float,
            }
        """
        pass


class VLAForManipulation(VLAInterface):
    """
    专门用于机器人操作的VLA接口

    支持 ManiSkill 等操作环境
    """

    @abstractmethod
    def process_ee_state(
        self,
        position: np.ndarray,  # [3] x, y, z
        orientation: np.ndarray,  # [4] quaternion 或 [3, 3] rotation matrix
        gripper_state: float,  # [0, 1] 0=closed, 1=open
    ) -> str:
        """
        将末端执行器状态转换为文本描述

        示例:
        "The gripper is at position (0.5, 0.2, 0.3) with orientation (...), gripper is 50% open"
        """
        pass

    @abstractmethod
    def predict_joint_positions(
        self,
        text: str,
        images: torch.Tensor,
        **generation_kwargs
    ) -> np.ndarray:
        """
        预测关节位置

        返回:
            joint_positions: [num_joints] 每个关节的角度
        """
        pass

    @abstractmethod
    def predict_ee_pose(
        self,
        text: str,
        images: torch.Tensor,
        **generation_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测末端执行器位姿

        返回:
            position: [3] x, y, z
            orientation: [4] quaternion
        """
        pass


class VLAForNavigation(VLAInterface):
    """
    专门室内导航的VLA接口

    支持 Habitat, AI2-THOR 等导航环境
    """

    @abstractmethod
    def predict_navigation_action(
        self,
        text: str,
        images: torch.Tensor,
        **generation_kwargs
    ) -> str:
        """
        预测导航动作

        返回:
            action: "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", etc.
        """
        pass

    @abstractmethod
    def predict_goal_position(
        self,
        text: str,
        images: torch.Tensor,
        **generation_kwargs
    ) -> np.ndarray:
        """
        预测目标位置（用于点目标导航）

        返回:
            position: [3] x, y, z
        """
        pass
