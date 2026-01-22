# siiRL 关键特性解析

## 概述

本文档详细回答关于 siiRL 的三个核心问题：
1. siiRL 是否支持异步训练？
2. siiRL 是否支持 VLA 模型？
3. siiRL 的 DAG 架构是什么意思？

---

## 1. 异步训练支持

### 1.1 简短回答

**是的，siiRL 完全支持异步训练**，并且是目前四个框架中对异步支持最完善的。

### 1.2 异步架构层次

siiRL 的异步支持体现在三个层面：

#### 层面1：异步推理引擎集成

```python
# siirl/engine/rollout/async_server.py
class AsyncServerBase(abc.ABC):
    """异步推理服务器基类"""

    @abc.abstractmethod
    async def chat_completion(self, raw_request: Request):
        """OpenAI chat completion API"""
        pass

    @abc.abstractmethod
    async def generate(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str
    ) -> List[int]:
        """异步生成响应"""
        pass

    @abc.abstractmethod
    async def init_engine(self):
        """异步初始化引擎"""
        pass

    @abc.abstractmethod
    async def wake_up(self):
        """唤醒引擎，加载模型权重和构建 KV cache"""
        pass

    @abc.abstractmethod
    async def sleep(self):
        """休眠引擎，卸载模型权重并丢弃 KV cache"""
        pass
```

**支持的异步推理引擎：**
- **VLLM**: `siirl/engine/rollout/vllm_rollout/vllm_async_server.py`
- **SGLang**: `siirl/engine/rollout/sglang_rollout/async_sglang_server.py`

#### 层面2：异步 VLA 环境接口

```python
# siirl/environment/embodied/base.py
class BaseVLAEnvironment(abc.ABC):
    """
    VLA 环境抽象基类

    特点：
    1. 原生异步接口设计 (async def)
    2. 支持多模态观察 (图像+文本)
    3. 返回标准 Gym 格式的 5 元组
    """

    @abstractmethod
    async def reset(self) -> Dict[str, Any]:
        """
        异步重置环境

        返回:
            Dict[str, Any]: 多模态观察
                例如: {"image": np.array, "text": "task prompt"}
        """
        pass

    @abstractmethod
    async def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        异步执行环境步骤

        参数:
            action: 动作字典
                例如: {"continuous_action": np.array([...])}

        返回:
            Tuple[Dict, float, bool, bool, Dict]:
                - observation: 下一步观察
                - reward: 奖励
                - terminated: 是否自然终止
                - truncated: 是否被截断
                - info: 额外信息
        """
        pass
```

#### 层面3：向量化环境

```python
# siirl/environment/embodied/venv.py
class SubprocVectorEnv(BaseVectorEnv):
    """子进程向量化环境实现"""

    def __init__(self, env_fns, num_envs):
        # 创建多个子进程
        self.procs = []
        for i in range(num_envs):
            proc = mp.Process(target=_worker, args=(env_fns[i],))
            proc.start()
            self.procs.append(proc)

    async def reset(self):
        # 并发重置所有环境
        futures = [p.reset.remote() for p in self.procs]
        return await asyncio.gather(*futures)

    async def step(self, actions):
        # 并发执行步骤
        futures = [p.step.remote(act) for p, act in zip(self.procs, actions)]
        return await asyncio.gather(*futures)
```

### 1.3 与其他框架对比

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **异步接口** | asyncio | ❌ 进程级 | ❌ 同步 | ✅ asyncio + 异步推理引擎 |
| **推理引擎** | 自研 | ❌ | ❌ | ✅ VLLM/SGLang |
| **VLA 环境** | ✅ 支持 | ⚠️ 有限 | ⚠️ 实验性 | ✅ 原生支持 |
| **并发能力** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### 1.4 siiRL 异步训练流程示例

```python
# 完整的异步训练流程
async def async_training_loop():
    # 1. 初始化异步推理引擎
    await async_server.init_engine()
    await async_server.wake_up()

    # 2. 异步环境重置
    observations = await vector_env.reset()

    # 3. 并发生成动作
    tasks = [
        async_server.generate(obs, sampling_params, f"req_{i}")
        for i, obs in enumerate(observations)
    ]
    actions = await asyncio.gather(*tasks)

    # 4. 并发环境步骤
    results = await vector_env.step(actions)

    # 5. 休眠引擎（可选，用于节省显存）
    await async_server.sleep()
```

---

## 2. VLA 模型支持

### 2.1 简短回答

**是的，siiRL 对 VLA 模型有专门的支持**，是目前四个框架中 VLA 支持最完善的。

### 2.2 VLA 支持层次

#### 层次1：VLA 模型支持

```python
# siirl/utils/embodied/openvla_utils.py
# 支持 OpenVLA 和 OpenVLA-OFT 模型

def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    更新 checkpoint 的 AutoMap 配置

    将 AutoConfig 和 AutoModelForVision2Seq 设置为 OpenVLA 特定类：
    - AutoConfig: "configuration_prismatic.OpenVLAConfig"
    - AutoModelForVision2Seq: "modeling_prismatic.OpenVLAForActionPrediction"
    """
    # 线程安全的配置更新
    # 使用文件锁和原子写
    ...
```

**支持的 VLA 模型：**
- OpenVLA
- OpenVLA-OFT (Our Fine-Tuned)
- 自定义 VLA 模型

#### 层次2：SRPO 算法（专为 VLA 设计）

```python
# siirl/execution/dag/builtin_pipelines.py
def embodied_srpo_pipeline() -> TaskGraph:
    """
    Embodied AI GRPO 训练管道，包含数据过滤和 VJEPA 奖励计算

    工作流：
        1. rollout_actor: 使用 VLA agent 进行环境交互
        2. embodied_sampling: 数据验证和过滤
        3. data_rebalance: 过滤后数据重平衡
        4. compute_reward: 基于 VJEPA 的奖励计算
        5. calculate_advantages: GRPO 组优势计算
        6. actor_old_log_prob: 计算旧 actor 对数概率（仅前向）
        7. reference_log_prob: 计算参考模型对数概率
        8. actor_train: GRPO actor 训练
    """
    pipeline = Pipeline(
        "embodied_grpo_training_pipeline",
        "Embodied AI GRPO training with data filtering and VJEPA reward"
    )

    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    ).add_node(
        "dynaminc_sampling",
        func="siirl.user_interface.filter_interface.embodied.embodied_local_rank_sampling",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.DYNAMIC_SAMPLING
    ).add_node(
        "compute_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["dynaminc_sampling"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    )
    # ... 更多节点
```

#### 层次3：VLA 特定的数据处理

```python
# siirl/dag_worker/core_algos.py
def compute_response_mask(data: TensorDict):
    """
    计算序列响应部分的注意力掩码

    处理两种场景：
    1. 2D 响应 (NLP): (batch_size, response_length)
    2. 3D 响应 (Embodied AI/VLA): (batch_size, traj_len, action_token_len)
    """
    responses = data["responses"]
    attention_mask = data["attention_mask"]
    batch_size = responses.size(0)

    # 处理 3D 响应 (Embodied AI)
    if responses.ndim == 3:
        traj_len = responses.size(1)
        action_token_len = responses.size(2)

        if attention_mask.ndim == 3:
            # 从最后一维提取响应部分
            response_mask = attention_mask[:, :, -action_token_len:]
            # 展平为 2D
            response_mask = response_mask.reshape(batch_size, -1)
        else:
            response_length = traj_len * action_token_len
            response_mask = attention_mask[:, -response_length:]

    # 处理 2D 响应 (NLP)
    elif responses.ndim == 2:
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

    return response_mask
```

### 2.3 VLA 训练示例配置

```bash
# siirl/examples/embodied_srpo_trainer/run_openvla_oft_libero_goal.sh
#!/bin/bash

# Embodied SRPO 训练示例
# 使用 OpenVLA-OFT 模型在 Libero-Goal 环境中训练

python siirl/main_dag.py \
    --pipeline embodied_srpo_pipeline \
    --model.openvla \
    --model.model_path /path/to/openvla-oft \
    --env.libero \
    --env.task libero_goal \
    --rollout.num_envs 64 \
    --rollout.traj_len 50 \
    --training.use_vje_reward \
    --training.batch_size 32 \
    --training.lr 1e-5
```

### 2.4 与其他框架 VLA 支持对比

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **VLA 模型支持** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **专用算法** | ❌ | ❌ | ❌ | ✅ SRPO |
| **3D 动作处理** | ⚠️ 需定制 | ⚠️ 需定制 | ❌ | ✅ 原生 |
| **环境适配器** | NavSim/Bench2Drive | NavSim/Bench2Drive | 实验性 | Libero/NavSim 等 |
| **异步推理** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 3. DAG 架构

### 3.1 简短回答

**DAG (Directed Acyclic Graph，有向无环图)** 是 siiRL 的核心执行模型，用于定义和管理复杂的训练工作流。

### 3.2 什么是 DAG 架构？

#### 3.2.1 概念定义

DAG 架构是一种将训练流程建模为**有向无环图**的方法：
- **节点 (Node)**: 代表一个计算任务（如数据加载、模型推理、训练等）
- **边 (Edge)**: 代表节点之间的依赖关系
- **有向**: 数据流动有明确方向
- **无环**: 不存在循环依赖，确保可以拓扑排序执行

#### 3.2.2 为什么使用 DAG？

```
传统方式（顺序执行）：
[数据加载] → [模型推理] → [奖励计算] → [优势计算] → [训练]
          ↓ 等待     ↓ 等待       ↓ 等待

DAG 方式（并行执行）：
[数据加载] ──────────────────────────┐
     ↓                               ↓
[模型推理] ──→ [奖励计算] ──→ [优势计算] ──→ [训练]
     ↓           ↑
[参考模型] ──────┘

优点：
1. 自动并行化：无依赖的节点可并行执行
2. 灵活组合：轻松添加/删除节点
3. 可视化调试：直观查看数据流
4. 资源优化：智能调度计算资源
```

### 3.3 siiRL DAG 实现细节

#### 3.3.1 节点类型 (NodeType)

```python
# siirl/execution/dag/node.py
class NodeType(Enum):
    """DAG 中节点的类型"""

    COMPUTE = "COMPUTE"              # 通用计算任务
    DATA_LOAD = "DATA_LOAD"          # 从 DataLoader 加载数据
    ENV_INTERACT = "ENV_INTERACT"    # 与环境交互
    MODEL_INFERENCE = "MODEL_INFERENCE"  # 模型推理
    MODEL_TRAIN = "MODEL_TRAIN"      # 模型训练
    PUT_TO_BUFFER = "PUT_TO_BUFFER"  # 数据放入分布式缓冲区
    GET_FROM_BUFFER = "GET_FROM_BUFFER"  # 从分布式缓冲区获取数据
    BARRIER_SYNC = "BARRIER_SYNC"    # 全局同步点
    CUSTOM = "CUSTOM"                # 用户自定义节点类型
```

#### 3.3.2 节点角色 (NodeRole)

```python
# siirl/execution/dag/node.py
class NodeRole(Enum):
    """节点在分布式强化学习框架中的角色"""

    DEFAULT = "DEFAULT"              # 默认
    ACTOR = "ACTOR"                  # Actor 网络
    ADVANTAGE = "ADVANTAGE"          # 优势函数计算
    CRITIC = "CRITIC"                # Critic 网络
    ROLLOUT = "ROLLOUT"              # Rollout（数据收集）
    REFERENCE = "REFERENCE"          # 参考模型
    REWARD = "REWARD"                # 奖励计算
    DYNAMIC_SAMPLING = "DYNAMIC_SAMPLING"  # 动态采样
```

#### 3.3.3 Pipeline 构建器 API

```python
# siirl/execution/dag/pipeline.py
class Pipeline:
    """
    简化的 Pipeline 构建器 API

    用户可以直接指定每个节点执行的函数，使整个工作流程透明易懂
    """

    def __init__(self, pipeline_id: str, description: str = ""):
        """
        初始化 Pipeline 构建器

        Args:
            pipeline_id: pipeline 的唯一标识符
            description: 人类可读的描述
        """
        self.pipeline_id = pipeline_id
        self.description = description
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        func: Union[str, Callable],
        deps: Optional[List[str]] = None,
        config: Optional[NodeConfig] = None,
        **kwargs
    ) -> "Pipeline":
        """
        添加节点到 pipeline

        Args:
            node_id: 节点的唯一标识符
            func: 要执行的函数，可以是：
                  - 字符串路径: "module.path:ClassName.method"
                  - 可调用对象: 直接函数引用
            deps: 此节点依赖的节点 ID 列表
            config: 节点配置（可选）
            **kwargs: 额外的节点参数（如 only_forward_compute）

        Returns:
            self: 支持方法链式调用
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        deps = deps or []
        config = config or NodeConfig()

        self._nodes[node_id] = {
            "func": func,
            "deps": deps,
            "config": config,
            "kwargs": kwargs
        }
        return self

    def build(self) -> TaskGraph:
        """
        构建 TaskGraph

        Returns:
            TaskGraph: 验证后的任务图，准备执行
        """
        task_graph = TaskGraph(graph_id=self.pipeline_id)

        for node_id, node_info in self._nodes.items():
            # 创建 Node 实例
            node = Node(
                node_id=node_id,
                node_type=kwargs.get("node_type", NodeType.COMPUTE),
                node_role=kwargs.get("node_role", NodeRole.DEFAULT),
                dependencies=node_info["deps"],
                agent_group=node_info["config"].agent_group,
                config=node_info["config"].config,
                **kwargs
            )

            # 绑定执行函数
            func = node_info["func"]
            if isinstance(func, str):
                node.executable_ref = func
                node._resolve_executable()
            else:
                node.executable = func

            task_graph.add_node(node)

        # 构建邻接表并验证
        task_graph.build_adjacency_lists()
        valid, msg = task_graph.validate_graph()
        if not valid:
            raise ValueError(f"Invalid pipeline: {msg}")

        return task_graph
```

### 3.4 内置 Pipeline 示例

#### 3.4.1 GRPO Pipeline

```python
# siirl/execution/dag/builtin_pipelines.py
def grpo_pipeline() -> TaskGraph:
    """
    标准 GRPO (Group Relative Policy Optimization) pipeline

    工作流：
        1. rollout_actor: 使用策略模型生成序列
        2. function_reward: 计算生成序列的奖励
        3. calculate_advantages: 计算优势估计
        4. actor_old_log_prob: 用旧策略计算对数概率（仅前向）
        5. reference_log_prob: 用参考模型计算对数概率
        6. actor_train: 训练 actor 模型

    返回:
        TaskGraph: 验证后的任务图
    """
    pipeline = Pipeline("grpo_training_pipeline", "Standard GRPO workflow")

    # 所有函数路径都显式可见！
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
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR,
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.REFERENCE
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )

    return pipeline.build()
```

#### 3.4.2 PPO Pipeline

```python
def ppo_pipeline() -> TaskGraph:
    """
    标准 PPO (Proximal Policy Optimization) pipeline

    工作流：
        1. rollout_actor: 使用策略模型生成序列
        2. function_reward: 计算生成序列的奖励
        3. compute_value: 计算价值函数估计（仅前向）
        4. calculate_advantages: 计算 GAE
        5. actor_old_log_prob: 用旧策略计算对数概率（仅前向）
        6. reference_log_prob: 用参考模型计算对数概率
        7. actor_train: 训练 actor 模型
        8. critic_train: 训练 critic（价值）模型
    """
    pipeline = Pipeline("ppo_training_pipeline", "Standard PPO workflow")

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
        "compute_value",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_value",
        deps=["function_reward"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.CRITIC,
        only_forward_compute=True
    ).add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["compute_value"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    ).add_node(
        "actor_old_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_old_log_prob",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR,
        only_forward_compute=True
    ).add_node(
        "reference_log_prob",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_ref_log_prob",
        deps=["actor_old_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.REFERENCE
    ).add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["reference_log_prob"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    ).add_node(
        "critic_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_critic",
        deps=["actor_train"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.CRITIC
    )

    return pipeline.build()
```

#### 3.4.3 DAPO Pipeline

```python
def dapo_pipeline() -> TaskGraph:
    """
    DAPO (Data-Augmented Policy Optimization) pipeline

    DAPO 是 GRPO 的变体，基于指标方差进行动态采样过滤。
    关键区别：计算奖励后，过滤掉方差为 0 的轨迹组
    （全对或全错），因为它们不提供学习信号。
    """
    pipeline = Pipeline("dapo_training_pipeline", "DAPO workflow")

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
        "dynamic_sampling",
        func="siirl.user_interface.filter_interface.dapo.dynamic_sampling",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.DYNAMIC_SAMPLING
    ).add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["dynamic_sampling"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    )
    # ... 其余节点类似 GRPO

    return pipeline.build()
```

### 3.5 DAG 可视化

```python
# Pipeline 可视化
pipeline = Pipeline("my_pipeline", "My custom training")
# ... 添加节点

# 保存 DAG 可视化图
pipeline.visualize(
    output_path="my_pipeline",
    directory="./visualizations/"
)
# 生成: ./visualizations/my_pipeline.png
```

### 3.6 DAG 与其他框架对比

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **工作流定义** | Python 类 | YAML 配置 | Trainer 类 | DAG 函数 |
| **可视化** | ❌ | ⚠️ 有限 | ❌ | ✅ 内置 |
| **并行化** | 手动 | 手动 | 手动 | ✅ 自动 |
| **扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **调试友好** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### 3.7 自定义 Pipeline 示例

```python
from siirl.execution.dag.pipeline import Pipeline
from siirl.execution.dag.node import NodeType, NodeRole
from siirl.execution.dag.builtin_pipelines import grpo_pipeline

# 方法1：完全自定义
def my_custom_pipeline():
    """自定义训练 pipeline"""
    pipeline = Pipeline("my_custom", "My custom workflow")

    # 添加自定义节点
    pipeline.add_node(
        "custom_data_loader",
        func="my_module:MyDataLoader.load",
        deps=[],
        node_type=NodeType.DATA_LOAD
    ).add_node(
        "custom_preprocess",
        func="my_module:preprocess_data",
        deps=["custom_data_loader"],
        node_type=NodeType.COMPUTE
    ).add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=["custom_preprocess"],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    )
    # ... 继续添加其他节点

    return pipeline.build()

# 方法2：基于现有 pipeline 修改
def my_grpo_variant():
    """GRPO 变体"""
    # 从内置 pipeline 开始
    base_pipeline = grpo_pipeline()

    # 在 reward 和 advantage 之间插入自定义节点
    # 注意：需要修改 TaskGraph 结构
    # （具体实现略）

    return modified_pipeline
```

---

## 4. 总结

### 4.1 三个问题的答案总结

| 问题 | 答案 | 详细程度 |
|------|------|----------|
| **异步训练支持** | ✅ 完全支持 | ⭐⭐⭐⭐⭐ 最完善 |
| **VLA 模型支持** | ✅ 专门支持 | ⭐⭐⭐⭐⭐ 最完善 |
| **DAG 架构** | ✅ 核心特性 | 独特优势 |

### 4.2 为什么 siiRL 是 VLA 训练的首选？

1. **异步训练**: 完整的异步支持 + VLLM/SGLang 集成
2. **VLA 支持**: 专门的 SRPO 算法 + 3D 动作处理 + 多个 VLA 模型支持
3. **DAG 架构**: 灵活的工作流定义 + 自动并行化 + 可视化调试
4. **可扩展性**: 多控制器设计 + 向量化环境 + 分布式训练

### 4.3 代码位置索引

| 功能 | 代码位置 |
|------|----------|
| 异步推理服务器 | `siirl/engine/rollout/async_server.py` |
| VLLM 集成 | `siirl/engine/rollout/vllm_rollout/` |
| SGLang 集成 | `siirl/engine/rollout/sglang_rollout/` |
| VLA 环境 | `siirl/environment/embodied/base.py` |
| 向量化环境 | `siirl/environment/embodied/venv.py` |
| OpenVLA 工具 | `siirl/utils/embodied/openvla_utils.py` |
| Pipeline 构建器 | `siirl/execution/dag/pipeline.py` |
| 节点定义 | `siirl/execution/dag/node.py` |
| 内置 Pipelines | `siirl/execution/dag/builtin_pipelines.py` |
| 核心 RL 算法 | `siirl/dag_worker/core_algos.py` |
| SRPO Pipeline | `siirl/execution/dag/builtin_pipelines.py:embodied_srpo_pipeline` |
| 示例脚本 | `siirl/examples/embodied_srpo_trainer/` |
