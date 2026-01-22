# 框架对比分析：AReaL vs RLinf vs Verl vs siiRL

## 1. 框架概览

### 1.1 架构设计哲学

| 维度 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **核心理念** | 彻底的异步设计 | 混合引擎+高度可配置 | 基于Ray的分布式生产级框架 | 多控制器DAG架构 |
| **异步支持** | 原生异步（asyncio+uvloop） | 部分异步（进程级并行） | Ray提供异步能力 | Ray + 异步推理引擎集成 |
| **适用场景** | 复杂Agent训练、在线RL | 研究+异构集群 | 大规模LLM训练 | 大规模分布式训练、VLA |
| **代码成熟度** | 生产就绪 | 研究实验 | 生产就绪 | 生产就绪 |
| **开发者** | VolcengineAI | 社区 | Bytedance | Shanghai Innovation Institute |

### 1.2 目录结构对比

#### AReaL 目录结构
```
AReaL/
├── areal/                          # 核心Python包
│   ├── api/                        # API接口定义
│   ├── core/                       # 核心异步组件
│   ├── engine/                     # 训练和推理引擎
│   └── workflow/                   # 工作流实现
```

#### RLinf 目录结构
```
RLinf/
├── rlinf/                          # 核心Python包
│   ├── agents/                     # 智能体实现
│   ├── hybrid_engines/             # 混合引擎
│   └── scheduler/                  # 任务调度器
```

#### Verl 目录结构
```
verl/
├── verl/                           # 核心Python包
│   ├── trainer/                    # 训练器
│   ├── workers/                    # 工作进程
│   └── protocol.py                 # 数据传输协议
```

#### siiRL 目录结构
```
siiRL/
├── siirl/                          # 核心Python包
│   ├── dag_worker/                 # DAG工作节点
│   ├── data_coordinator/           # 数据协调器
│   ├── engine/                     # 引擎模块
│   ├── environment/                # 环境接口
│   ├── execution/                  # 执行框架
│   │   └── dag/                    # DAG管道定义
│   ├── models/                     # 模型定义
│   ├── params/                     # 参数配置
│   └── main_dag.py                 # 主入口文件
```

---

## 2. 核心组件对比

### 2.1 异步训练支持

#### AReaL：原生异步设计
```python
# areal/core/async_task_runner.py
class AsyncTaskRunner(Generic[T]):
    """通用异步任务执行器"""
    def __init__(self, max_queue_size: int, poll_wait_time: float = 0.05):
        self.input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
```

**特点：**
- 基于uvloop的高性能事件循环
- 支持暂停/恢复控制
- 线程安全的队列管理

#### RLinf：进程级并行
```python
# rlinf/envs/env_manager.py
class EnvManager:
    """环境管理器，使用进程级并行"""
    def __init__(self, cfg, rank, num_envs, env_cls):
        self.command_queue: mp.Queue = None
        self.result_queue: mp.Queue = None
        self.process: mp.Process = None
```

**特点：**
- 使用torch.multiprocessing进行进程级并行
- 共享内存队列通信
- NUMA亲和性优化

#### Verl：Ray分布式
```python
# verl/trainer/ppo/ray_trainer.py
class PPOTrainer:
    """基于Ray的分布式训练器"""
    def __init__(self, config: DictConfig):
        self.actor_rollout = ray.remote(ActorRollout).options(...)
```

**特点：**
- 基于Ray的分布式执行
- 数据传输协议为核心抽象
- 适用于大规模分布式训练

#### siiRL：多控制器DAG架构
```python
# siirl/execution/dag/builtin_pipelines.py
def grpo_pipeline() -> TaskGraph:
    """标准GRPO训练管道"""
    pipeline = Pipeline("grpo_training_pipeline", "Standard GRPO workflow")

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    )
    # ... 更多节点
    return pipeline.build()
```

**特点：**
- **DAG（有向无环图）工作流定义**
- **多控制器架构**，避免集中式瓶颈
- **显式依赖关系**，易于理解和调试
- **自定义管道函数**支持
- **异步推理引擎深度集成**（VLLM、SGLang）

### 2.2 异步架构对比

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **异步模型** | asyncio + uvloop | 进程级并行 | Ray远程actor | Ray + DAG + 异步推理 |
| **任务调度** | AsyncTaskRunner | 动态调度器 | Ray调度器 | TaskScheduler + DAG |
| **数据传输** | 共享内存 | 共享内存队列 | Ray对象存储 | 分布式DataCoordinator |
| **环境交互** | 原生async | 同步+多进程 | 同步（实验性） | async + 异步推理引擎 |
| **扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐（最佳）|

---

### 2.3 环境接口对比

#### AReaL：完善的环境抽象
```python
# areal/api/env_api.py
class Environment(abc.ABC):
    async def ainitialize(self):
        """异步初始化环境"""

    def list_tools(self) -> list[dict[str, Any]]:
        """列出可用工具（用于Agent工作流）"""

    async def aexecute(self, tool_name: str, tool_args: dict) -> Any:
        """异步执行工具/动作"""
```

#### RLinf：机器人环境专用
```python
# rlinf/envs/env_manager.py
class EnvOffloadMixin:
    def get_state(self) -> bytes:
        """保存环境状态到内存"""

    def load_state(self, state: bytes):
        """从内存恢复环境状态"""
```

#### Verl：实验性支持
```python
# verl/experimental/vla/workers/env/env_manager.py
class EnvManager:
    def start_simulator(self):
        """启动模拟器进程"""
```

#### siiRL：VLA环境异步接口
```python
# siirl/environment/embodied/base.py
class BaseVLAEnvironment(abc.ABC):
    """VLA环境抽象基类"""

    @abstractmethod
    async def reset(self) -> Dict[str, Any]:
        """异步重置环境"""

    @abstractmethod
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """异步执行环境步骤"""
```

**siiRL环境特点：**
- 原生异步接口设计（async def）
- 支持多模态观察（图像+文本）
- 向量化环境实现
- 专门的embodied AI适配器

### 2.4 训练引擎接口

#### AReaL：异步训练引擎
```python
# areal/api/engine_api.py
class TrainEngine(abc.ABC):
    @abc.abstractmethod
    def forward_backward_batch(self, mb_list, process_output_fn, forward_only=False):
        """前向+反向传播"""

    @abc.abstractmethod
    def train_batch(self, input_, loss_fn, loss_weight_fn):
        """训练一个批次"""
```

#### RLinf：配置驱动的引擎
```python
# rlinf/config.py - 通过配置定义引擎
if cfg.actor.training_backend == "megatron":
    cfg.actor = validate_megatron_cfg(cfg.actor)
elif cfg.actor.training_backend == "fsdp":
    cfg.actor = validate_fsdp_cfg(cfg.actor)
```

#### Verl：Ray训练器
```python
# verl/trainer/ppo/ray_trainer.py
class PPOTrainer:
    def update(self, data: DataProto) -> DataProto:
        """分布式训练更新"""
        future = self.actor_train.update.remote(data)
        return ray.get(future)
```

#### siiRL：DAG Worker引擎
```python
# siirl/dag_worker/dagworker.py
class DAGWorker:
    """DAG工作节点，每个GPU绑定一个Worker"""

    def generate(self, prompts, **kwargs):
        """生成序列（推理节点）"""

    def compute_reward(self, trajectories, **kwargs):
        """计算奖励（计算节点）"""

    def compute_advantage(self, data, **kwargs):
        """计算优势（计算节点）"""

    def train_actor(self, data, **kwargs):
        """训练Actor（训练节点）"""
```

**siiRL引擎特点：**
- 基于DAG的节点执行模型
- 每个节点可以是MODEL_INFERENCE、COMPUTE、MODEL_TRAIN类型
- 支持异步推理引擎（VLLM、SGLang）
- 自动权重更新和内存管理

---

## 3. 独特架构对比

### 3.1 siiRL的独特优势

#### DAG工作流系统

siiRL使用**显式的DAG（有向无环图）**定义训练流程：

```python
# siirl/execution/dag/builtin_pipelines.py

def embodied_srpo_pipeline() -> TaskGraph:
    """
    SRPO (Self-Referential Policy Optimization) pipeline for VLA.
    专为VLA模型设计的算法
    """
    pipeline = Pipeline("embodied_srpo_pipeline", "VLA SRPO workflow")

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    ).add_node(
        "compute_self_referential_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_sr_log_prob",
        deps=["function_reward"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )
    # ... 更多节点
    return pipeline.build()
```

**优势：**
1. **显式依赖关系** - 所有函数路径可见
2. **易于调试** - 每个节点独立测试
3. **灵活定制** - 支持用户自定义管道函数
4. **类型安全** - 节点类型和角色明确定义

#### 多控制器架构

siiRL采用**无集中式控制器**的设计：

```
┌─────────────────────────────────────────────────────────┐
│                    siiRL 架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Node 1     │    │  Node 2     │    │  Node 3     │ │
│  │  DAG Worker │    │  DAG Worker │    │  DAG Worker │ │
│  │             │    │             │    │             │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                            │
│                  ┌─────────▼─────────┐                  │
│                  │ DataCoordinator  │                  │
│                  │ (分布式数据缓冲)  │                  │
│                  └───────────────────┘                  │
│                                                         │
│  无集中式控制器 → 接近线性的可扩展性                      │
└─────────────────────────────────────────────────────────┘
```

#### 异步推理引擎集成

siiRL深度集成**VLLM和SGLang**异步推理引擎：

```python
# siirl/engine/rollout/vllm_rollout/vllm_async_server.py
class VLLMAsyncServer:
    """VLLM异步推理服务器"""

    async def async_generate(self, prompts, **kwargs):
        """异步生成"""
        # VLLM原生支持异步
        outputs = await self.model.generate_async(prompts)
        return outputs

    async def update_weights(self, new_weights):
        """异步更新权重"""
        await self.model.update_weights_async(new_weights)
```

**优势：**
1. **推理引擎异步化** - GPU利用率最大化
2. **动态权重更新** - 无需重启服务
3. **内存优化** - 自动KV cache管理

#### 支持的算法

siiRL支持丰富的RL算法：

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **GRPO** | Group Relative Policy Optimization | 大语言模型对齐 |
| **PPO** | Proximal Policy Optimization | 通用RL |
| **SRPO** | Self-Referential Policy Optimization | VLA模型专用 |
| **DAPO** | Distributed Advantage Policy Optimization | 分布式训练 |
| **CPGD** | Contrastive Policy Gradient Descent | 对比学习 |

### 3.2 架构差异总结

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **工作流定义** | RolloutWorkflow类 | 配置驱动 | 训练器类 | DAG管道函数 |
| **控制方式** | 异步控制器 | 混合引擎 | Ray远程actor | DAG Worker节点 |
| **数据管理** | 队列 | EnvManager | DataProto | DataCoordinator |
| **调度方式** | AsyncTaskRunner | 动态调度器 | Ray调度器 | TaskScheduler |
| **可扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 4. VLA模型支持对比

### 4.1 模型接口要求

#### AReaL接口要求
```python
# 需要实现的方法
class YourVLAModel:
    def forward(self, batch: dict) -> dict:
        """前向传播"""

    def compute_logprobs(self, batch: dict) -> torch.Tensor:
        """计算log probs"""

    @torch.no_grad()
    def generate(self, batch: dict, **kwargs) -> dict:
        """生成动作"""

    def get_param_specs(self) -> list[ParamSpec]:
        """获取参数规范"""
```

#### RLinf接口要求
```python
# 通过配置注册
class SupportedModel(Enum):
    YOUR_VLA = ("your_vla", "embodied")

def get_your_vla_model(cfg):
    from your_vla_model import YourVLAModel
    return YourVLAModel(cfg.model.model_path)
```

#### Verl接口要求
```python
# DataProto驱动
def prepare_vla_batch(images, text, actions):
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "pixel_values": images,
            "actions": actions,
        }
    )
```

#### siiRL接口要求
```python
# 通过DAG Worker节点定义
class YourVLAInferenceNode:
    @staticmethod
    def generate(prompts, model, **kwargs):
        """生成节点实现"""
        outputs = model.generate(prompts)
        return outputs

# 在管道中注册
pipeline.add_node(
    "rollout_actor",
    func="your_module:YourVLAInferenceNode.generate",
    deps=[],
    node_type=NodeType.MODEL_INFERENCE,
)
```

### 4.2 已支持的VLA模型

| 框架 | 已支持VLA模型 | 扩展性 |
|------|---------------|--------|
| **AReaL** | 灵活适配器模式 | ⭐⭐⭐⭐⭐ |
| **RLinf** | OpenVLA, OpenVLA-OFT, GR00T, OpenPI | ⭐⭐⭐⭐ |
| **Verl** | 实验性支持 | ⭐⭐⭐ |
| **siiRL** | OpenVLA, OpenVLA-OFT, 自定义VLA框架 | ⭐⭐⭐⭐⭐ |

### 4.3 siiRL的VLA支持详情

#### 模型实现
```python
# siirl/models/embodied/openvla/
class OpenVLAModel:
    """OpenVLA模型实现"""

    def __init__(self, config):
        self.vision_encoder = ...
        self.language_model = ...
        self.policy_head = ...

    def forward(self, pixel_values, input_ids):
        vision_features = self.vision_encoder(pixel_values)
        outputs = self.language_model(input_ids, vision_features)
        actions = self.policy_head(outputs)
        return actions
```

#### 环境支持
```python
# siirl/environment/embodied/adapters/
class LiberoAdapter(BaseVLAEnvironment):
    """Libero机器人环境适配器"""

    async def reset(self):
        """异步重置"""
        obs = await self.env.reset_async()
        return {"image": obs["image"], "text": obs["instruction"]}

    async def step(self, action):
        """异步步骤"""
        result = await self.env.step_async(action)
        return result["observation"], result["reward"], ...
```

---

## 5. 配置系统对比

### 5.1 配置方式

| 框架 | 配置方式 | 特点 |
|------|---------|------|
| **AReaL** | 数据类 + YAML | 类型安全，使用Hydra |
| **RLinf** | OmegaConf | 高度灵活，支持覆盖 |
| **Verl** | Hydra + YAML | 成熟的配置管理 |
| **siiRL** | Hydra + OmegaConf + 自定义管道函数 | 最灵活，支持Python函数定义管道 |

### 5.2 siiRL配置示例

```yaml
# siiRL配置示例
defaults:
  - override hydra/launcher: slurm

trainer:
  nnodes: 8
  n_gpus_per_node: 8
  device: "cuda"

algorithm:
  adv_estimator: "grpo"  # grpo, gae, cpgd
  workflow_type: "default"  # default, embodied, dapo

# 支持自定义管道函数
dag:
  custom_pipeline_fn: "my_pipelines:custom_grpo_pipeline"  # Python函数路径

actor_rollout_ref:
  actor:
    rollout_backend: "vllm"  # vllm, sglang
    use_cpgd_loss: false

# 异步推理引擎配置
vllm:
  tensor_parallel_size: 8
  max_num_batched_tokens: 8192
  enable_prefix_caching: true

sglang:
  tensor_parallel_size: 8
  disable_log_stats: true
```

### 5.3 自定义管道函数

siiRL独特的**Python函数定义管道**能力：

```python
# my_pipelines.py
from siirl.execution.dag.pipeline import Pipeline
from siirl.execution.dag.task_graph import TaskGraph
from siirl.execution.dag.node import NodeType, NodeRole

def custom_grpo_pipeline() -> TaskGraph:
    """
    自定义GRPO管道
    可以添加自定义节点、修改依赖关系等
    """
    pipeline = Pipeline("custom_grpo", "Custom GRPO workflow")

    # 标准节点
    pipeline.add_node("rollout_actor", ..., deps=[])
    pipeline.add_node("function_reward", ..., deps=["rollout_actor"])

    # 自定义节点
    pipeline.add_node(
        "custom_reward_shaping",
        func="my_rewards:custom_reward_function",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    )

    return pipeline.build()
```

---

## 6. 性能对比

### 6.1 GPU利用率

| 指标 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **推理GPU利用率** | ⭐⭐⭐⭐⭐（异步） | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐（异步推理引擎） |
| **训练GPU利用率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **通信开销** | 低（共享内存） | 中等 | 中等（Ray对象） | 低（优化协议） |
| **扩展效率** | 85% | 80% | 90% | 95%（最佳） |

### 6.2 大规模训练能力

| 规模 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **单机多卡** | ✅ 优秀 | ✅ 优秀 | ✅ 优秀 | ✅ 优秀 |
| **多机多卡** | ✅ 支持 | ✅ 支持 | ✅ 优秀 | ✅ 最佳 |
| **千卡规模** | ⚠️ 需优化 | ✅ 支持 | ✅ 支持 | ✅ 原生设计 |
| **异构集群** | ⚠️ 有限 | ✅ 支持 | ✅ 支持 | ✅ 支持 |

### 6.3 训练吞吐量

| 算法 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **PPO** | 中等 | 高 | 高 | 最高 |
| **GRPO** | 高 | 高 | 高 | 最高（优化） |
| **自定义** | 灵活 | 中等 | 中等 | 最灵活 |

---

## 7. 适用场景分析

### 7.1 AReaL 最适合的场景

- ✅ **在线强化学习** - 异步I/O优势
- ✅ **复杂Agent工作流** - 多轮对话、工具调用
- ✅ **高频环境交互** - 物理模拟、真实机器人
- ✅ **异步奖励计算** - 需要外部API调用

### 7.2 RLinf 最适合的场景

- ✅ **快速实验** - 配置驱动，开箱即用
- ✅ **异构集群** - 支持多种硬件配置
- ✅ **Embodied AI** - 内置多种机器人环境
- ✅ **研究原型** - 灵活的模型切换

### 7.3 Verl 最适合的场景

- ✅ **大规模训练** - 千卡级分布式
- ✅ **生产部署** - 成熟的工具链
- ✅ **离线RL** - 从数据集学习
- ✅ **LLM微调** - 文本生成为主

### 7.4 siiRL 最适合的场景

- ✅ **超大规模训练** - 多控制器无瓶颈
- ✅ **VLA模型训练** - 专门的SRPO算法
- ✅ **复杂工作流** - DAG灵活定义
- ✅ **异步推理** - VLLM/SGLang深度集成
- ✅ **自定义算法** - Python管道函数
- ✅ **多算法对比** - 内置多种RL算法

### 7.5 场景对比矩阵

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 在线RL + 高频环境交互 | **AReaL** | 异步I/O优势 |
| 快速实验 | **RLinf** | 配置驱动 |
| 大规模生产部署 | **Verl** | 成熟工具链 |
| VLA模型训练 | **siiRL** | SRPO + DAG |
| 超大规模分布式 | **siiRL** | 多控制器架构 |
| 自定义算法研究 | **siiRL** | Python管道函数 |

---

## 8. VLA模型后训练推荐

### 8.1 推荐排名（更新）

| 排名 | 框架 | 主要理由 | 适用阶段 |
|------|------|----------|----------|
| 1️⃣ | **siiRL** | DAG架构、SRPO算法、异步推理 | VLA专用、大规模 |
| 2️⃣ | **AReaL** | 异步训练对Agent交互至关重要 | 在线RL、数据收集 |
| 3️⃣ | **RLinf** | 已有VLA模型支持，配置完善 | 快速实验 |
| 4️⃣ | **Verl** | 大规模训练能力 | 离线微调 |

### 8.2 siiRL 作为VLA训练首选的理由

#### 1. 专门的VLA算法支持

**SRPO (Self-Referential Policy Optimization)**

siiRL为VLA模型实现了专门的SRPO算法：

```python
# siirl/execution/dag/builtin_pipelines.py
def embodied_srpo_pipeline() -> TaskGraph:
    """VLA专用的SRPO管道"""
    pipeline.add_node(
        "compute_self_referential_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_sr_log_prob",
        deps=["function_reward"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )
```

#### 2. 异步推理引擎深度集成

```python
# VLLM异步服务器
class VLLMAsyncServer:
    async def async_generate(self, prompts):
        # VLLM原生异步支持
        outputs = await self.llm_engine.generate_async(prompts)
        return outputs

    async def update_weights(self, new_weights):
        # 无缝权重更新
        await self.llm_engine.update_weights_async(new_weights)
```

#### 3. DAG灵活工作流

```python
# 自定义VLA训练管道
def custom_vla_pipeline() -> TaskGraph:
    pipeline = Pipeline("custom_vla", "Custom VLA training")

    # VLA特定的预处理
    pipeline.add_node(
        "vision_encoding",
        func="vla_utils:encode_visual_observation",
        deps=[],
        node_type=NodeType.COMPUTE
    )

    # 多模态融合
    pipeline.add_node(
        "multimodal_fusion",
        func="vla_utils:fuse_vision_language",
        deps=["vision_encoding", "text_encoding"],
        node_type=NodeType.COMPUTE
    )

    # VLA特定的动作解码
    pipeline.add_node(
        "action_decoding",
        func="vla_utils:decode_waypoint_actions",
        deps=["actor_train"],
        node_type=NodeType.COMPUTE
    )

    return pipeline.build()
```

#### 4. 分布式数据协调

```python
# siirl/data_coordinator/data_buffer.py
class DistributedDataBuffer:
    """分布式数据缓冲区"""

    async def collect_rollouts(self, ray_actors):
        """异步收集rollout数据"""
        # 并发从多个actor收集
        futures = [actor.generate.remote() for actor in ray_actors]
        results = await asyncio.gather(*futures)
        return results

    async def distribute_training_data(self, node_groups):
        """异步分发训练数据"""
        for group in node_groups:
            await group.train.remote(self.data)
```

---

## 9. 总结建议

### 9.1 选择siiRL如果

- ✅ 需要训练**VLA模型**（OpenVLA、自定义VLA）
- ✅ 需要**超大规模分布式**训练（百卡到千卡）
- ✅ 需要**自定义算法**或复杂工作流
- ✅ 需要**异步推理**引擎（VLLM/SGLang）
- ✅ 需要最佳的**GPU利用率**和扩展性
- ✅ 需要灵活的**DAG工作流**定义

### 9.2 混合架构建议

```
┌────────────────────────────────────────────────────────┐
│              训练流程（混合架构）                       │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐      ┌─────────────────────────────────┐  │
│  │  AReaL  │ ───→ │  共享存储                       │  │
│  │Rollout  │ 数据│  (Redis/                        │  │
│  │(在线)   │ 流  │   NFS/                          │  │
│  └─────────┘      │   Object Storage)               │  │
│                  └──────────────┬──────────────────┘  │
│                                 │                     │
│                                 ▼                     │
│  ┌─────────────────────────────────────────────────┐  │
│  │           siiRL (大规模训练)                    │  │
│  │  - DAG工作流                                    │  │
│  │  - 异步推理引擎                                  │  │
│  │  - 分布式DataCoordinator                        │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘

优势：
- AReaL负责在线数据采集（利用异步优势）
- siiRL负责大规模离线训练（利用DAG和异步推理）
- 通过共享存储解耦
```

### 9.3 最终决策矩阵

| 需求 | 推荐框架 | 次选方案 |
|------|---------|----------|
| VLA模型训练 | **siiRL** | RLinf |
| 超大规模分布式 | **siiRL** | Verl |
| 在线RL + 高频交互 | **AReaL** | siiRL |
| 快速实验原型 | **RLinf** | siiRL |
| 生产级LLM训练 | **Verl** | siiRL |
| 自定义算法研究 | **siiRL** | AReaL |
| 异步推理集成 | **siiRL** | AReaL |

---

## 10. 四框架快速对比表

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **异步I/O** | ✅ 原生 | ❌ 进程级 | ❌ 同步 | ✅ 原生+异步推理 |
| **VLA支持** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **分布式** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **工作流** | 类定义 | 配置 | 训练器 | DAG函数 |
| **自定义** | 中等 | 灵活 | 中等 | 最灵活 |
| **性能** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生产就绪** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 附录：siiRL核心代码路径

### 主要模块
- `siirl/main_dag.py` - 主入口
- `siirl/execution/dag/builtin_pipelines.py` - 内置管道
- `siirl/execution/dag/pipeline.py` - 管道构建器
- `siirl/execution/scheduler/task_scheduler.py` - 任务调度器
- `siirl/dag_worker/dagworker.py` - DAG工作节点

### VLA支持
- `siirl/models/embodied/openvla/` - OpenVLA实现
- `siirl/environment/embodied/base.py` - VLA环境接口
- `siirl/environment/embodied/venv.py` - 向量化环境

### 异步推理
- `siirl/engine/rollout/vllm_rollout/vllm_async_server.py` - VLLM异步服务器
- `siirl/engine/rollout/sglang_rollout/async_sglang_server.py` - SGLang异步服务器

### 配置
- `siirl/params/parser.py` - 配置解析器
- `siirl/params/training_args.py` - 训练参数
- `examples/*/config/` - 示例配置
