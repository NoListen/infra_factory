# VLA模型接口框架

统一的VLA (Vision-Language-Action) 模型接口框架，支持AReaL、RLinf、Verl三个训练框架，用于NavSim和Bench2Drive等驾驶环境的在线强化学习训练。

## 项目结构

```
infra_factory/
├── docs/                           # 文档目录
│   ├── 01_framework_comparison.md  # 框架对比分析
│   ├── 02_model_integration_guide.md  # 模型整合指南
│   ├── 03_async_training_analysis.md  # 异步训练分析
│   └── 04_online_rl_interfaces.md   # 在线RL接口分析
│
├── vla_models/                     # VLA模型接口包
│   ├── __init__.py                 # 包入口
│   ├── interface.py                # 核心接口定义
│   ├── model_registry.py           # 模型注册机制
│   └── config.py                   # 配置管理
│
├── tests/                          # 单元测试
│   ├── conftest.py                 # 测试配置和fixtures
│   ├── test_interface.py           # 接口测试
│   ├── test_model_registry.py      # 注册机制测试
│   └── test_environment_adapters.py # 环境适配器测试
│
├── AReaL/                          # AReaL框架（子模块）
├── RLinf/                          # RLinf框架（子模块）
├── verl/                           # Verl框架（子模块）
└── README.md                       # 本文件
```

## 文档概览

### 1. 框架对比分析 (`docs/01_framework_comparison.md`)

三个训练框架的详细对比：
- **AReaL**: 彻底的异步设计，适合在线RL和高频环境交互
- **RLinf**: 混合引擎+配置驱动，适合快速实验
- **Verl**: Ray分布式框架，适合大规模训练

### 2. 模型整合指南 (`docs/02_model_integration_guide.md`)

详细说明如何将VLA模型整合到各框架：
- AReaL接口要求
- RLinf配置方式
- Verl DataProto协议
- 统一VLA接口设计

### 3. 异步训练分析 (`docs/03_async_training_analysis.md`)

AReaL异步训练移植到Verl的可行性分析：
- 核心差异
- 工作量评估（8-12周）
- 关键挑战
- 替代方案建议

### 4. 在线RL接口分析 (`docs/04_online_rl_interfaces.md`)

在线强化学习环境接口详细分析：
- AReaL：完善的异步环境接口
- RLinf：进程级并行，机器人优化
- Verl：实验性支持
- NavSim/Bench2Drive适配器设计

## VLA模型接口包

### 核心接口

#### VLAInterface

所有VLA模型的基础接口：

```python
from vla_models import VLAInterface, VLAInput, VLAPrediction

class MyVLA(VLAInterface):
    def encode_text(self, text, max_length=None):
        # 编码文本
        pass

    def encode_image(self, images, size=None):
        # 编码图像
        pass

    def forward(self, input_ids, attention_mask, pixel_values=None):
        # 前向传播
        pass

    def predict(self, inputs: VLAInput, **kwargs):
        # 生成预测
        pass
```

#### VLAForDriving

驾驶场景专用接口：

```python
from vla_models import VLAForDriving

class DrivingVLA(VLAForDriving):
    def process_ego_state(self, position, heading, velocity):
        # 处理自车状态为文本
        pass

    def process_route(self, route):
        # 处理路线为文本
        pass

    def predict_waypoints(self, text, images, num_waypoints=5):
        # 预测未来路点
        pass

    def waypoints_to_controls(self, waypoints, current_state):
        # 路点转换为控制信号
        pass
```

### 模型注册

```python
from vla_models import register_vla_model, register_framework_adapter

# 注册模型
@register_vla_model("my_driving_vla")
class MyDrivingVLA(VLAForDriving):
    def __init__(self, model_path):
        # 初始化模型
        pass

# 注册AReaL适配器
@register_framework_adapter("areal", "my_driving_vla")
def areal_adapter(model_config, engine_config):
    from areal.models import MyVLAForAReaL
    base_model = MyDrivingVLA(**model_config)
    return MyVLAForAReaL(base_model, **engine_config)
```

### 环境适配

```python
from vla_models import register_env_adapter

# 注册NavSim环境适配器
@register_env_adapter("navsim", "areal")
class AReaLNavSimAdapter:
    async def ainitialize(self):
        # 异步初始化
        pass

    async def aexecute(self, tool_name, tool_args):
        # 异步执行环境操作
        pass
```

## 单元测试

完整的测试套件覆盖所有接口：

```bash
# 运行所有测试
pytest tests/ -v

# 跳过需要GPU的测试
pytest tests/ -m "not gpu" -v

# 运行特定测试文件
pytest tests/test_interface.py -v

# 查看测试覆盖率
pytest tests/ --cov=vla_models --cov-report=html
```

### 测试覆盖

- ✅ VLAInterface接口测试
- ✅ VLAForDriving驾驶接口测试
- ✅ 模型注册机制测试
- ✅ 环境适配器测试
- ✅ 配置管理测试

## 快速开始

### 1. 定义VLA模型

```python
# my_vla_model.py
from vla_models import VLAForDriving
from vla_models import register_vla_model, register_framework_adapter

@register_vla_model("my_driving_vla")
class MyDrivingVLA(VLAForDriving):
    def __init__(self, model_path, **kwargs):
        # 加载预训练模型
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def encode_text(self, text, max_length=None):
        return self.tokenizer(text, max_length=max_length, return_tensors="pt")

    def encode_image(self, images, size=None):
        return self.processor(images, return_tensors="pt")

    def forward(self, input_ids, attention_mask, pixel_values=None, **kwargs):
        return self.model(input_ids, attention_mask, pixel_values)

    def predict_waypoints(self, text, images, num_waypoints=5, **kwargs):
        inputs = self.encode_multimodal(text, images)
        outputs = self.forward(**inputs)
        # 解析输出为路点坐标
        waypoints = self._parse_waypoints(outputs)
        return waypoints

    def waypoints_to_controls(self, waypoints, current_state):
        # 实现pure pursuit或Stanley控制器
        return pure_pursuit_control(waypoints, current_state)
```

### 2. 注册框架适配器

```python
# adapters.py
from vla_models import register_framework_adapter

# AReaL适配器
@register_framework_adapter("areal", "my_driving_vla")
def areal_adapter(model_config, engine_config):
    from my_vla_model import MyDrivingVLA
    from areal.models import MyVLAForAReaL

    base_model = MyDrivingVLA(**model_config)
    return MyVLAForAReaL(base_model, **engine_config)

# RLinf适配器
@register_framework_adapter("rlinf", "my_driving_vla")
def rlinf_adapter(cfg):
    from my_vla_model import MyDrivingVLA
    from rlinf.models import MyVLAForRLinf

    base_model = MyDrivingVLA(model_path=cfg.model.model_path)
    return MyVLAForRLinf(base_model, cfg)
```

### 3. 配置文件

```yaml
# config/my_driving_vla.yaml
model:
  name: "my_driving_vla"
  version: "1.0.0"
  checkpoint_path: "/path/to/checkpoint"

  vision_encoder:
    type: "clip"
    pretrained_path: "/path/to/vision_encoder"
    freeze: false
    image_size: [224, 224]

  language_model:
    base_model: "gpt2"
    lora_enabled: true
    lora_r: 64

  policy_head:
    type: "waypoint"
    action_dim: 15  # 5 waypoints * 3

training:
  optimizer:
    type: "adamw"
    lr: 1e-4

  param_groups:
    vision_encoder: 1e-5
    language_model: 1e-4
    policy_head: 1e-3

environment:
  type: "navsim"
  observation:
    camera:
      enabled: true
      width: 224
      height: 224
  action:
    type: "waypoint"
    num_waypoints: 5

reward:
  type: "driving"
  weights:
    collision: -10.0
    success: 10.0
    progress: 1.0
```

### 4. 使用示例

```python
# train.py
from vla_models import create_model, VLAConfig

# 加载配置
config = VLAConfig.from_yaml("config/my_driving_vla.yaml")

# 创建基础模型
model = create_model("my_driving_vla", model_path=config.model.checkpoint_path)

# 或者创建框架适配的模型
areal_model = create_model(
    "my_driving_vla",
    framework="areal",
    model_config=config.model,
    engine_config=config.training
)

# 使用模型进行预测
from vla_models import VLAInput

inputs = VLAInput(
    text="Drive safely to the destination",
    images="camera_view.png"
)

prediction = model.predict(inputs, max_new_tokens=50)
waypoints = prediction.waypoints
controls = model.waypoints_to_controls(waypoints, current_state)
```

## 环境支持

### NavSim

- AReaL适配器：`AReaLNavSimAdapter`
- RLinf适配器：`RLinfNavSimAdapter`
- Verl适配器：`VerlNavSimAdapter`

### Bench2Drive

- 适配器待实现（接口已定义）

## 扩展性

### 添加新环境

1. 实现环境接口
2. 为每个框架创建适配器
3. 注册到`VLAModelRegistry`

```python
@register_env_adapter("new_env", "areal")
class AReaLNewEnvAdapter:
    async def ainitialize(self):
        pass

    async def aexecute(self, tool_name, tool_args):
        pass
```

### 添加新框架

1. 扩展`VLAModelRegistry._adapters`
2. 实现框架特定的适配器模式

## 测试

运行测试确保接口实现正确：

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-asyncio

# 运行测试
pytest tests/ -v --cov=vla_models --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

## 下一步

1. 实现真实的VLA模型
2. 为NavSim/Bench2Drive创建完整的环境适配器
3. 集成到AReaL/RLinf/Verl训练流程
4. 编写训练脚本和示例

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License
