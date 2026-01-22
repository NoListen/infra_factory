# VLA模型接口框架

统一的VLA (Vision-Language-Action) 模型接口框架，支持AReaL、RLinf、Verl、siiRL四个训练框架，用于NavSim和Bench2Drive等驾驶环境的在线强化学习训练。

## 项目结构

```
infra_factory/
├── docs/                           # 文档目录
│   ├── 01_framework_comparison.md  # 四框架对比分析（含siiRL）
│   ├── 02_model_integration_guide.md  # 模型整合指南
│   ├── 03_async_training_analysis.md  # 异步训练分析
│   └── 04_online_rl_interfaces.md   # 在线RL接口分析（含siiRL）
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
├── siiRL/                          # siiRL框架（子模块）
└── README.md                       # 本文件
```

## 文档概览

### 1. 框架对比分析 (`docs/01_framework_comparison.md`)

四个训练框架的详细对比：
- **AReaL**: 彻底的异步设计，适合在线RL和高频环境交互
- **RLinf**: 混合引擎+配置驱动，适合快速实验
- **Verl**: Ray分布式框架，适合大规模训练
- **siiRL**: 多控制器DAG架构，专为VLA和大规模训练设计

### 2. 模型整合指南 (`docs/02_model_integration_guide.md`)

详细说明如何将VLA模型整合到各框架：
- AReaL接口要求
- RLinf配置方式
- Verl DataProto协议
- siiRL DAG Worker接口
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
- siiRL：DAG架构的异步VLA接口
- NavSim/Bench2Drive适配器设计

## 框架对比速查

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **核心理念** | 异步设计 | 混合引擎 | Ray分布式 | 多控制器DAG |
| **异步支持** | asyncio | 进程级 | Ray | Ray+异步推理 |
| **VLA支持** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **分布式** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **工作流** | 类定义 | 配置 | 训练器 | DAG函数 |
| **扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐（最佳）|

### VLA训练推荐排名

| 排名 | 框架 | 主要理由 |
|------|------|----------|
| 1️⃣ | **siiRL** | DAG架构、SRPO算法、异步推理 |
| 2️⃣ | **AReaL** | 异步训练、Agent交互 |
| 3️⃣ | **RLinf** | 已有VLA支持、快速实验 |
| 4️⃣ | **Verl** | 大规模训练能力 |

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

## 快速开始

### 1. 定义VLA模型

```python
# my_vla_model.py
from vla_models import VLAForDriving, register_vla_model

@register_vla_model("my_driving_vla")
class MyDrivingVLA(VLAForDriving):
    def __init__(self, model_path, **kwargs):
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(model_path)

    def encode_text(self, text, max_length=None):
        return self.tokenizer(text, max_length=max_length, return_tensors="pt")

    def encode_image(self, images, size=None):
        return image_processor(images)

    def predict_waypoints(self, text, images, num_waypoints=5, **kwargs):
        inputs = self.encode_multimodal(text, images)
        outputs = self.model.generate(**inputs)
        return parse_waypoints(outputs)
```

### 2. 注册框架适配器

```python
# adapters.py
from vla_models import register_framework_adapter

# siiRL适配器
@register_framework_adapter("siirl", "my_driving_vla")
def siirl_adapter(cfg):
    from my_vla_model import MyDrivingVLA
    from siirl.models import MyVLAForSiirl

    base_model = MyDrivingVLA(cfg.model.model_path)
    return MyVLAForSiirl(base_model, cfg)
```

### 3. 配置文件

```yaml
# config/my_driving_vla.yaml
model:
  name: "my_driving_vla"
  checkpoint_path: "/path/to/checkpoint"

  vision_encoder:
    type: "clip"
    pretrained_path: "/path/to/vision_encoder"

  language_model:
    base_model: "gpt2"
    lora_enabled: true

  policy_head:
    type: "waypoint"
    action_dim: 15

training:
  optimizer:
    type: "adamw"
    lr: 1e-4

environment:
  type: "navsim"

reward:
  type: "driving"
  weights:
    collision: -10.0
    success: 10.0
```

### 4. 使用示例

```python
from vla_models import create_model, VLAConfig

# 加载配置
config = VLAConfig.from_yaml("config/my_driving_vla.yaml")

# 创建模型
model = create_model("my_driving_vla", model_path=config.model.checkpoint_path)

# 使用模型
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
- siiRL适配器：`NavSimVLAAdapter`（原生异步支持）

### Bench2Drive

- AReaL适配器：`AReaLBench2DriveAdapter`
- RLinf适配器：`RLinfBench2DriveAdapter`
- Verl适配器：`VerlBench2DriveAdapter`
- siiRL适配器：`Bench2DriveVLAAdapter`（原生异步支持）

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

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License
