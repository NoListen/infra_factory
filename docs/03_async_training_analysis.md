# AReaL异步训练移植分析

## 1. 核心差异概述

### 1.1 架构对比

| 特性 | AReaL | Verl | 移植复杂度 |
|------|-------|------|-----------|
| 异步模型 | asyncio + uvloop | Ray远程actor | ⭐⭐⭐⭐ 高 |
| 任务执行 | AsyncTaskRunner | Ray.remote | ⭐⭐⭐ 中 |
| 数据传输 | 共享内存 | Ray对象存储 | ⭐⭐⭐ 中 |
| 工作流接口 | async def arun_episode | 同步函数 | ⭐⭐⭐⭐⭐ 很高 |
| 权重更新 | 异步RPC | Ray对象存储 | ⭐⭐ 低 |
| 环境交互 | 原生异步 | 实验性支持 | ⭐⭐⭐⭐⭐ 很高 |

### 1.2 AReaL的异步优势

#### AsyncTaskRunner核心实现

```python
# areal/core/async_task_runner.py
class AsyncTaskRunner(Generic[T]):
    """
    通用异步任务执行器

    特点：
    1. 基于uvloop的高性能事件循环
    2. 线程安全的队列管理
    3. 支持暂停/恢复
    4. 健康检查和异常处理
    """

    def __init__(self, max_queue_size: int, poll_wait_time: float = 0.05):
        # 线程控制
        self.exiting = threading.Event()
        self.paused = threading.Event()

        # 队列管理
        self.input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

    def submit(self, async_fn, *args, task_id: int, **kwargs) -> int:
        """提交异步任务（线程安全）"""
        task_input = _TaskInput(async_fn=async_fn, args=args, kwargs=kwargs, task_id=task_id)
        self.input_queue.put_nowait(task_input)
        self._signal_new_input()
        return task_id

    def wait(self, count: int, timeout: float = None) -> list[T]:
        """等待任务完成"""
        results = []
        while len(results) < count:
            result = self.output_queue.get(timeout=wait_time)
            results.append(result.data)
        return results

    def pause(self):
        """暂停新任务（现有任务继续）"""
        self.paused.set()

    def resume(self):
        """恢复任务提交"""
        self.paused.clear()
        self._signal_new_input()
```

#### 工作流异步接口

```python
# areal/api/workflow_api.py
class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        完全异步的episode执行

        典型实现：
        1. 异步生成动作
        2. 异步执行环境步骤
        3. 异步计算奖励
        4. 所有I/O不阻塞GPU
        """
        raise NotImplementedError()
```

### 1.3 Verl的Ray分布式模型

#### Ray训练器

```python
# verl/trainer/ppo/ray_trainer.py
class PPOTrainer:
    """
    基于Ray的分布式训练器

    特点：
    1. 使用Ray远程actor
    2. 数据通过Ray对象存储传输
    3. 分布式调度和容错
    """

    def __init__(self, config: DictConfig):
        # 创建远程actor
        self.actor_rollout = ray.remote(ActorRollout).options(
            num_gpus=config.actor_rollout.gpu_per_actor
        )
        self.actor_train = ray.remote(ActorTrain).options(
            num_gpus=config.actor_train.gpu_per_actor
        )
        self.critic = ray.remote(Critic).options(
            num_gpus=config.critic.gpu_per_actor
        )

    def update(self, data: DataProto) -> DataProto:
        """分布式训练更新"""
        # 使用Ray的远程调用
        future = self.actor_train.update.remote(data)
        result = ray.get(future)
        return result
```

## 2. 移植工作量评估

### 2.1 总体估算

| 阶段 | 工作内容 | 预估时间 | 难度 |
|------|---------|---------|------|
| 阶段1 | 核心异步层移植 | 3-4周 | ⭐⭐⭐⭐ 高 |
| 阶段2 | 工作流接口适配 | 2-3周 | ⭐⭐⭐⭐⭐ 很高 |
| 阶段3 | 引擎接口适配 | 2-3周 | ⭐⭐⭐ 中 |
| 阶段4 | 环境接口适配 | 1-2周 | ⭐⭐⭐⭐ 高 |
| 阶段5 | 测试和优化 | 1-2周 | ⭐⭐⭐ 中 |
| **总计** | **完整移植** | **9-14周** | - |

### 2.2 阶段1：核心异步层移植（3-4周）

#### 任务：实现Ray版本的AsyncTaskRunner

```python
# verl/async_core/ray_async_runner.py
import asyncio
import ray
from ray.util.queue import Queue as RayQueue
from typing import Callable, Any

class RayAsyncTaskRunner:
    """
    基于Ray的异步任务执行器

    挑战：
    1. Ray的异步API与asyncio不同
    2. 需要适配Ray的对象存储
    3. 队列实现不同
    """

    def __init__(self, max_queue_size: int):
        # Ray的队列（与Python标准库不同）
        self.input_queue = RayQueue(maxsize=max_queue_size)
        self.output_queue = RayQueue(maxsize=max_queue_size)

        # Ray actor pool
        self.actor_pool = self._create_actor_pool()

    async def submit(self, fn: Callable, *args, **kwargs):
        """
        提交任务到Ray actor

        挑战：Ray的remote调用是同步的，需要手动异步化
        """
        # 选择一个actor
        actor = self.actor_pool.select_actor()

        # 提交任务（返回ObjectRef）
        future = actor.execute.remote(fn, *args, **kwargs)

        # 包装为asyncio Future
        return await self._ray_future_to_async(future)

    async def _ray_future_to_async(self, ray_object_ref):
        """
        将Ray的ObjectRef转换为asyncio Future

        这是核心挑战
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ray_object_ref)

    async def wait(self, count: int, timeout: float = None):
        """
        等待任务完成

        挑战：Ray没有原生的async等待
        """
        results = []
        for _ in range(count):
            result = await self._async_queue_get(self.output_queue, timeout)
            results.append(result)
        return results

    async def _async_queue_get(self, queue, timeout):
        """异步从队列获取"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, queue.get, timeout)

    def pause(self):
        """暂停新任务"""
        # Ray的暂停机制需要手动实现
        for actor in self.actor_pool.actors:
            ray.get(actor.pause.remote())

    def resume(self):
        """恢复任务"""
        for actor in self.actor_pool.actors:
            ray.get(actor.resume.remote())
```

**关键挑战：**

1. **异步语义转换**
   - Ray的`remote()`调用是同步的，返回ObjectRef
   - 需要手动转换为asyncio Future
   - `ray.get()`会阻塞，需要在executor中运行

2. **队列差异**
   - AReaL使用`queue.Queue`（线程安全）
   - Ray使用`ray.util.queue.Queue`（进程安全）
   - API不同，需要适配

3. **事件循环集成**
   - Ray有自己的事件循环
   - 与asyncio的事件循环需要协调

#### 测试需求

```python
# tests/test_ray_async_runner.py
import pytest
import asyncio
from verl.async_core.ray_async_runner import RayAsyncTaskRunner

@pytest.mark.asyncio
async def test_basic_submit_wait():
    """测试基本的提交和等待"""
    runner = RayAsyncTaskRunner(max_queue_size=10)
    await runner.initialize()

    async def sample_task(x):
        await asyncio.sleep(0.1)
        return x * 2

    # 提交任务
    runner.submit(sample_task, 5, task_id=1)
    runner.submit(sample_task, 10, task_id=2)

    # 等待结果
    results = await runner.wait(count=2)

    assert results[0] == 10
    assert results[1] == 20

    await runner.destroy()

@pytest.mark.asyncio
async def test_pause_resume():
    """测试暂停和恢复"""
    runner = RayAsyncTaskRunner(max_queue_size=10)
    await runner.initialize()

    async def long_task():
        await asyncio.sleep(1.0)
        return "done"

    # 提交多个任务
    for i in range(5):
        runner.submit(long_task, task_id=i)

    # 暂停
    runner.pause()

    # 等待当前任务完成
    await asyncio.sleep(1.5)

    # 恢复
    runner.resume()

    # 等待所有任务
    results = await runner.wait(count=5)

    assert len(results) == 5

    await runner.destroy()
```

### 2.3 阶段2：工作流接口适配（2-3周）

#### 任务：将AReaL的异步工作流转换为Ray actor

```python
# AReaL的工作流（异步）
class AReaLWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 1. 异步生成
        response = await engine.agenerate(ModelRequest(
            input_ids=data["input_ids"],
            images=data["images"],
        ))

        # 2. 异步环境交互
        next_obs, reward = await self.env.aexecute(
            "step",
            {"action": response.actions}
        )

        # 3. 异步奖励计算
        final_reward = await self.reward_fn.async_compute(
            trajectory,
            response
        )

        return {
            "observations": [data["obs"], next_obs],
            "actions": [response.actions],
            "rewards": [reward],
        }

# Verl的工作流（需要转换为Ray actor）
@ray.remote
class VerlWorkflowActor:
    """
    Ray actor版本的工作流

    挑战：
    1. 所有async调用需要转换为ray.remote调用
    2. 需要手动管理依赖关系
    3. 错误处理更复杂
    """

    def __init__(self, engine_actor, env_actor, reward_fn):
        self.engine = engine_actor
        self.env = env_actor
        self.reward_fn = reward_fn

    def run_episode(self, data):
        """
        同步版本的episode执行
        （Ray actor中不能直接用async）
        """
        # 1. 生成动作（ray.get会阻塞）
        response_future = self.engine.generate.remote(
            ModelRequest(
                input_ids=data["input_ids"],
                images=data["images"],
            )
        )
        response = ray.get(response_future)

        # 2. 环境交互（阻塞）
        step_future = self.env.execute.remote(
            "step",
            {"action": response.actions}
        )
        next_obs, reward = ray.get(step_future)

        # 3. 奖励计算（阻塞）
        reward_future = self.reward_fn.compute.remote(
            trajectory,
            response
        )
        final_reward = ray.get(reward_future)

        return {
            "observations": [data["obs"], next_obs],
            "actions": [response.actions],
            "rewards": [reward],
        }

    async def run_episode_async(self, data):
        """
        尝试保持异步语义

        限制：Ray actor的async方法支持有限
        """
        # 使用ray.experimental.async_api
        response = await self.engine.generate.async_call(
            ModelRequest(...)
        )
        # ... 但这需要所有组件都支持async
```

**关键挑战：**

1. **异步语义丢失**
   - AReaL的`await`变成`ray.get()`（阻塞）
   - 失去并发优势
   - GPU利用率下降

2. **依赖管理**
   - 需要显式管理ray.get()的依赖关系
   - 容易造成死锁
   - 调试困难

3. **错误处理**
   - Ray的异常传播机制与asyncio不同
   - 需要额外的错误处理代码

#### 测试需求

```python
# tests/test_workflow_adapter.py
import pytest
import ray
from verl.workflow.ray_workflow_adapter import VerlWorkflowActor

@pytest.fixture
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_workflow_basic(ray_init):
    """测试基本工作流"""
    # 创建actors
    engine = ray.remote(MockEngine).remote()
    env = ray.remote(MockEnv).remote()

    # 创建工作流actor
    workflow = VerlWorkflowActor.remote(engine, env, None)

    # 运行episode
    data = {"input_ids": [1, 2, 3], "images": None}
    result = ray.get(workflow.run_episode.remote(data))

    assert "observations" in result
    assert "actions" in result
    assert "rewards" in result

def test_workflow_concurrent(ray_init):
    """测试并发执行"""
    engine = ray.remote(MockEngine).remote()
    env = ray.remote(MockEnv).remote()
    workflow = VerlWorkflowActor.remote(engine, env, None)

    # 并发提交多个episodes
    futures = []
    for i in range(10):
        data = {"input_ids": [i, i+1, i+2], "images": None}
        future = workflow.run_episode.remote(data)
        futures.append(future)

    # 等待所有完成
    results = ray.get(futures)

    assert len(results) == 10
```

### 2.4 阶段3：引擎接口适配（2-3周）

#### 任务：包装AReaL的InferenceEngine为Ray actor

```python
# areal/engine/remote_inference.py
@ray.remote(num_gpus=1)
class RayInferenceEngineActor:
    """
    Ray actor包装的推理引擎

    挑战：
    1. 异步方法转同步
    2. 权重更新机制转换
    3. 批处理优化
    """

    def __init__(self, base_engine_config):
        # 在actor内部创建实际的推理引擎
        from areal.engine.sglang_engine import RemoteSGLangEngine
        self.base_engine = RemoteSGLangEngine(**base_engine_config)
        self.base_engine.initialize()

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        同步版本的generate

        原AReaL: async def agenerate(...)
        现在需要同步阻塞
        """
        # 使用asyncio.run在同步上下文中运行async代码
        response = asyncio.run(self.base_engine.agenerate(request))
        return response

    def generate_batch(self, requests: list[ModelRequest]) -> list[ModelResponse]:
        """
        批处理优化

        挑战：需要手动管理批处理
        """
        # 并发执行多个生成请求
        async def _generate_all():
            tasks = [self.base_engine.agenerate(req) for req in requests]
            return await asyncio.gather(*tasks)

        responses = asyncio.run(_generate_all())
        return responses

    def update_weights(self, weight_data):
        """
        权重更新

        挑战：AReaL使用异步RPC，Ray使用对象存储
        """
        # 从Ray对象存储获取权重
        # 更新本地模型
        self.base_engine.update_weights_from_distributed(weight_data)
```

**关键挑战：**

1. **异步转同步**
   - `asyncio.run()`有性能开销
   - 事件循环创建/销毁成本
   - 可能影响吞吐量

2. **批处理优化**
   - AReaL的AsyncTaskRunner自动批处理
   - Ray需要手动管理
   - 需要额外的调度逻辑

#### 测试需求

```python
# tests/test_engine_adapter.py
import pytest
import ray
from verl.engine.ray_engine_adapter import RayInferenceEngineActor

@pytest.fixture
def engine_actor(ray_init):
    config = {
        "model_path": "/path/to/model",
        "tensor_parallel_size": 1,
    }
    actor = RayInferenceEngineActor.remote(config)
    yield actor
    ray.kill(actor)

def test_single_generate(engine_actor):
    """测试单个生成"""
    request = ModelRequest(
        input_ids=[1, 2, 3],
        max_tokens=100,
    )

    response = ray.get(engine_actor.generate.remote(request))

    assert response.token_ids is not None
    assert len(response.token_ids) > 0

def test_batch_generate(engine_actor):
    """测试批处理生成"""
    requests = [
        ModelRequest(input_ids=[i, i+1, i+2], max_tokens=100)
        for i in range(10)
    ]

    responses = ray.get(engine_actor.generate_batch.remote(requests))

    assert len(responses) == 10
    for response in responses:
        assert response.token_ids is not None
```

### 2.5 阶段4：环境接口适配（1-2周）

#### 任务：将AReaL的环境接口转为Ray actor

```python
# verl/env/ray_env_adapter.py
@ray.remote
class RayEnvironmentActor:
    """
    Ray actor包装的环境

    挑战：
    1. 状态ful环境的状态管理
    2. 异步操作转同步
    3. 资源清理
    """

    def __init__(self, env_config):
        # 导入AReaL的环境
        from areal.envs.navsim_env import NavSimEnvironment
        self.env = NavSimEnvironment(**env_config)

        # 初始化环境
        self._initialized = False

    def initialize(self):
        """初始化环境"""
        if not self._initialized:
            asyncio.run(self.env.ainitialize())
            self._initialized = True

    def execute(self, tool_name: str, tool_args: dict):
        """
        执行环境操作

        原AReaL: async def aexecute(...)
        """
        self.initialize()
        result = asyncio.run(self.env.aexecute(tool_name, tool_args))
        return result

    def get_state(self) -> bytes:
        """获取环境状态"""
        return self.env.get_state()

    def load_state(self, state: bytes):
        """加载环境状态"""
        self.env.load_state(state)

    def close(self):
        """清理环境"""
        asyncio.run(self.env.aclose())
        self._initialized = False
```

**关键挑战：**

1. **状态管理**
   - AReaL支持环境状态保存/恢复
   - Ray actor重启后状态丢失
   - 需要额外的状态持久化

2. **资源清理**
   - Ray actor的生命周期管理
   - 需要正确关闭环境
   - 避免资源泄漏

#### 测试需求

```python
# tests/test_env_adapter.py
import pytest
import ray
from verl.env.ray_env_adapter import RayEnvironmentActor

@pytest.fixture
def env_actor(ray_init):
    config = {"env_type": "navsim", "scene_id": "test_scene"}
    actor = RayEnvironmentActor.remote(config)
    ray.get(actor.initialize.remote())
    yield actor
    ray.get(actor.close.remote())
    ray.kill(actor)

def test_env_execute(env_actor):
    """测试环境执行"""
    result = ray.get(env_actor.execute.remote(
        "reset",
        {}
    ))

    assert "observation" in result

def test_env_state(env_actor):
    """测试状态保存/恢复"""
    # 执行一些步骤
    ray.get(env_actor.execute.remote("step", {"action": [0.5, 0.0]}))

    # 保存状态
    state = ray.get(env_actor.get_state.remote())
    assert state is not None

    # 恢复状态
    ray.get(env_actor.load_state.remote(state))

    # 检查状态是否正确恢复
    result = ray.get(env_actor.execute.remote("get_state", {}))
    assert result is not None
```

### 2.6 阶段5：测试和优化（1-2周）

#### 性能测试

```python
# tests/benchmark/test_performance.py
import pytest
import time
import ray
from verl.async_core.ray_async_runner import RayAsyncTaskRunner
from areal.core.async_task_runner import AsyncTaskRunner

@pytest.mark.benchmark
def test_async_throughput_comparison():
    """
    对比AReaL和Ray版本的吞吐量

    预期：Ray版本可能更慢
    """
    num_tasks = 1000

    # AReaL版本
    areal_runner = AsyncTaskRunner(max_queue_size=1000)
    areal_runner.initialize()

    start = time.time()
    for i in range(num_tasks):
        areal_runner.submit(sample_task, i, task_id=i)
    areal_results = areal_runner.wait(count=num_tasks)
    areal_time = time.time() - start

    # Ray版本
    ray.init()
    ray_runner = RayAsyncTaskRunner(max_queue_size=1000)
    await ray_runner.initialize()

    start = time.time()
    for i in range(num_tasks):
        await ray_runner.submit(sample_task, i, task_id=i)
    ray_results = await ray_runner.wait(count=num_tasks)
    ray_time = time.time() - start

    ray.shutdown()

    print(f"AReaL: {areal_time:.2f}s, Ray: {ray_time:.2f}s")
    print(f"Slowdown: {ray_time / areal_time:.2f}x")

    # Ray版本通常慢1.5-3倍
```

#### 压力测试

```python
# tests/stress/test_stress.py
@pytest.mark.stress
async def test_high_concurrency():
    """高并发压力测试"""
    runner = RayAsyncTaskRunner(max_queue_size=10000)
    await runner.initialize()

    # 提交大量任务
    num_tasks = 10000
    for i in range(num_tasks):
        await runner.submit(long_running_task, i, task_id=i)

    # 测试内存使用
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # 等待完成
    results = await runner.wait(count=num_tasks)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Memory: {mem_before:.2f}MB -> {mem_after:.2f}MB")

    # 检查内存泄漏
    assert mem_after < mem_before * 1.5  # 增长不超过50%

    await runner.destroy()
```

## 3. 关键挑战总结

### 3.1 性能挑战

| 方面 | AReaL | Verl (Ray) | 影响 |
|------|-------|------------|------|
| 任务调度 | uvloop事件循环 | Ray调度器 | Ray慢1.5-2x |
| 数据传输 | 共享内存 | 对象存储+序列化 | Ray慢2-3x |
| 异步I/O | 原生asyncio | ray.get()阻塞 | Ray慢2-3x |
| 总体性能 | 基准 | 慢3-5倍 | ⚠️ 严重 |

### 3.2 语义挑战

```python
# AReaL: 自然的异步
async def arun_episode(self, engine, data):
    # 所有操作并发执行
    action_task = engine.agenerate(request)
    env_task = self.env.aexecute("step", {...})

    action, env_result = await asyncio.gather(action_task, env_task)

# Verl: 需要手动管理依赖
def run_episode(self, engine, env, data):
    # 顺序执行（失去并发）
    action = ray.get(engine.generate.remote(request))
    env_result = ray.get(env.execute.remote("step", {...}))
```

### 3.3 调试挑战

| 问题 | AReaL | Verl (Ray) |
|------|-------|------------|
| 异常传播 | 直接抛出 | 需要ray.get()才能看到 |
| 死锁检测 | asyncio工具 | Ray工具较少 |
| 日志追踪 | 单进程 | 分布式，复杂 |
| 性能分析 | 标准工具 | Ray profiler |

### 3.4 维护挑战

1. **代码同步**
   - AReaL更新时需要同步移植
   - 两套代码库维护

2. **测试覆盖**
   - 需要维护两套测试
   - Ray测试需要分布式环境

3. **文档维护**
   - 需要记录移植差异
   - 用户需要学习两种模式

## 4. 替代方案建议

### 4.1 混合架构

```
┌───────────────────────────────────────┐
│           训练流程                      │
├───────────────────────────────────────┤
│                                        │
│  ┌─────────┐      ┌─────────────────┐ │
│  │ AReaL  │ ───→ │ 共享存储        │ │
│  │Rollout │ 数据 │ (Redis/         │ │
│  │(在线)  │      │  NFS)           │ │
│  └─────────┘      └────────┬────────┘ │
│                            │          │
│                            ▼          │
│  ┌─────────────────────────────────┐  │
│  │  Verl                          │  │
│  │  Training (离线)                │  │
│  └─────────────────────────────────┘  │
│                                        │
└────────────────────────────────────────┘

优势：
- AReaL保持异步优势
- Verl利用分布式训练能力
- 通过共享存储解耦
```

### 4.2 接口适配层

```python
# hybrid/adapter.py
class AReaLToVerlAdapter:
    """
    AReaL数据到Verl DataProto的适配器

    无需修改核心代码，只做数据格式转换
    """

    @staticmethod
    def areal_to_verl(areal_trajectory: dict) -> DataProto:
        """转换AReaL的trajectory到Verl的DataProto"""
        return DataProto.from_dict(
            tensors={
                "input_ids": areal_trajectory["input_ids"],
                "pixel_values": areal_trajectory["images"],
                "actions": areal_trajectory["actions"],
            },
            non_tensors={
                "rewards": areal_trajectory["rewards"],
            }
        )

    @staticmethod
    def verl_to_areal(verl_batch: DataProto) -> dict:
        """转换Verl的DataProto到AReaL格式"""
        return {
            "input_ids": verl_batch.batch["input_ids"],
            "images": verl_batch.batch["pixel_values"],
            "actions": verl_batch.batch["actions"],
        }
```

### 4.3 不移植的建议

**理由：**
1. **架构差异太大** - 异步是AReaL的核心优势，移植会失去
2. **性能可能下降** - Ray的对象存储开销可能抵消优势
3. **维护成本高** - 需要持续维护两套代码
4. **调试复杂度** - 分布式调试比asyncio复杂得多

**建议：**
- 选择适合场景的框架，而不是强行移植
- 需要异步交互 → AReaL
- 需要大规模训练 → Verl
- 考虑混合架构，各取所长

## 5. 决策矩阵

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 在线RL + 高频环境交互 | AReaL | 异步I/O优势 |
| 大规模离线训练 | Verl | 分布式能力 |
| 复杂Agent工作流 | AReaL | 异步工作流支持 |
| 生产级部署 | Verl | 成熟工具链 |
| 快速原型开发 | RLinf | 配置驱动 |

## 6. 结论

### 移植可行性：⚠️ 不推荐

**技术可行性：** ✅ 可以实现
**经济可行性：** ❌ 性能损失和维护成本过高

### 建议

1. **保持AReaL用于在线rollout**
   - 利用异步I/O优势
   - 最大化GPU利用率

2. **使用Verl用于大规模训练**
   - 利用分布式能力
   - 成熟的工具链

3. **通过共享存储连接**
   - AReaL收集数据
   - Verl训练模型
   - 各取所长

4. **如果必须统一**
   - 考虑将AReaL的异步层移植到Verl
   - 但这需要深入理解两者架构
   - 时间成本：3-6个月
