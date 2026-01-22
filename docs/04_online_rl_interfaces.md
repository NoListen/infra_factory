# 在线强化学习环境接口分析

## 1. AReaL - 最完善的在线RL支持

### 1.1 核心环境接口

```python
# areal/api/env_api.py
import abc
from typing import Any

class Environment(abc.ABC):
    """
    AReaL的环境抽象接口

    特点：
    1. 完全异步的I/O操作
    2. 支持工具调用（Agent工作流）
    3. 支持状态ful环境
    4. 优雅的资源管理
    """

    @abc.abstractmethod
    async def ainitialize(self):
        """异步初始化环境"""

    def list_tools(self) -> list[dict[str, Any]]:
        """列出可用工具（用于Agent工作流）"""

    @abc.abstractmethod
    async def aexecute(self, tool_name: str, tool_args: dict) -> Any:
        """异步执行工具/动作"""

    @abc.abstractmethod
    async def aclose(self):
        """异步清理资源"""
```

### 1.2 在线RL工作流接口

```python
# areal/api/workflow_api.py
class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self,
        engine: "InferenceEngine",
        data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """运行一个完整的episode（完全异步）"""
        raise NotImplementedError()
```

---

## 2. RLinf - 针对Embodied AI优化

### 2.1 环境管理器接口

```python
# rlinf/envs/env_manager.py
class EnvOffloadMixin:
    """环境状态保存/恢复混入类"""

    def get_state(self) -> bytes:
        """保存环境状态到内存"""

    def load_state(self, state: bytes):
        """从内存恢复环境状态"""

class EnvManager:
    """环境管理器（进程级并行）"""

    def start_env(self):
        """启动环境进程（异进程模式）"""

    def stop_env(self):
        """停止环境进程（保存状态）"""

    def __getattr__(self, name):
        """代理环境方法"""
        def method_proxy(*args, **kwargs):
            self.command_queue.put({"method": name, "args": args, "kwargs": kwargs})
            result = self.result_queue.get()
            return result["data"]
        return method_proxy
```

---

## 3. Verl - 实验性支持

### 3.1 实验性环境接口

```python
# verl/experimental/vla/workers/env/env_manager.py
class EnvManager:
    """Verl的环境管理器（实验性）"""

    def start_simulator(self):
        """启动模拟器进程"""

    def reset(self):
        """重置环境"""

    def step(self, actions):
        """执行一步"""

    def stop_simulator(self):
        """停止模拟器"""
```

---

## 4. siiRL - DAG架构的异步VLA接口

### 4.1 VLA环境接口

```python
# siirl/environment/embodied/base.py
class BaseVLAEnvironment(abc.ABC):
    """
    VLA环境抽象基类

    特点：
    1. 原生异步接口设计（async def）
    2. 支持多模态观察（图像+文本）
    3. 返回标准Gym格式的5元组
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
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
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

### 4.2 向量化环境

```python
# siirl/environment/embodied/venv.py
class BaseVectorEnv(abc.ABC):
    """向量化环境基类"""

    @abc.abstractmethod
    async def reset(self) -> List[Dict[str, Any]]:
        """批量重置所有环境"""

    @abc.abstractmethod
    async def step(self, actions: List[Dict]) -> Tuple[List, List[float], List[bool], List[bool], List]:
        """批量执行步骤"""


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

### 4.3 环境适配器

```python
# siirl/environment/embodied/adapters/
class LiberoAdapter(BaseVLAEnvironment):
    """Libero机器人环境适配器"""

    def __init__(self, task_id):
        from libero.libero_env import LiberoEnv
        self.env = LiberoEnv(task_id=task_id)

    async def reset(self):
        """异步重置"""
        import asyncio
        obs = await asyncio.to_thread(self.env.reset)
        return {
            "image": obs["image"],
            "text": obs["instruction"]
        }

    async def step(self, action):
        """异步步骤"""
        import asyncio
        result = await asyncio.to_thread(
            self.env.step,
            action["continuous_action"]
        )
        return (
            result["observation"],
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"]
        )
```

### 4.4 在线RL训练示例

```python
# siiRL训练流程使用DAG定义

def embodied_online_rl_pipeline() -> TaskGraph:
    """
    Embodied AI在线RL训练管道
    """
    pipeline = Pipeline("embodied_online_rl", "Online RL for VLA")

    # 1. Rollout节点（与环境交互）
    pipeline.add_node(
        "rollout_actor",
        func="siirl.dag_worker.dagworker:DAGWorker.generate",
        deps=[],
        node_type=NodeType.MODEL_INFERENCE,
        node_role=NodeRole.ROLLOUT
    )

    # 2. 奖励计算节点
    pipeline.add_node(
        "function_reward",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        deps=["rollout_actor"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.REWARD
    )

    # 3. 优势计算节点
    pipeline.add_node(
        "calculate_advantages",
        func="siirl.dag_worker.dagworker:DAGWorker.compute_advantage",
        deps=["function_reward"],
        node_type=NodeType.COMPUTE,
        node_role=NodeRole.ADVANTAGE
    )

    # 4. 训练节点
    pipeline.add_node(
        "actor_train",
        func="siirl.dag_worker.dagworker:DAGWorker.train_actor",
        deps=["calculate_advantages"],
        node_type=NodeType.MODEL_TRAIN,
        node_role=NodeRole.ACTOR
    )

    return pipeline.build()
```

---

## 5. 接口对比总结

### 5.1 功能对比

| 功能 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **环境初始化** | `async def ainitialize()` | `__init__()` | `__init__()` | `async def reset()` |
| **动作执行** | `async def aexecute()` | `def step()` | `def step()` | `async def step()` |
| **资源清理** | `async def aclose()` | `def close()` | `def close()` | N/A |
| **状态保存** | 自定义 | `get_state()`/`load_state()` | `get_state()`/`load_state()` | 自定义 |
| **工具调用** | ✅ `list_tools()` | ❌ | ❌ | ❌ |
| **异步I/O** | ✅ 原生 | ❌ 进程级 | ❌ 同步 | ✅ 原生 |
| **并发EPISODE** | ✅ 支持 | ⚠️ 有限 | ❌ 不支持 | ✅ 支持（向量化） |
| **多模态观察** | ✅ 支持 | ✅ 支持 | ⚠️ 部分 | ✅ 原生支持 |

### 5.2 性能对比

| 指标 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **环境吞吐** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GPU利用率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **并发能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### 5.3 适用场景

#### AReaL最适合：
- ✅ 复杂的多轮交互（Web Agent、游戏AI）
- ✅ 需要异步奖励计算（代码执行、外部API）
- ✅ 高频环境交互（物理模拟、实时控制）
- ✅ 状态ful环境（浏览器会话、长连接）

#### RLinf最适合：
- ✅ 机器人学习（ManiSkill、Behavior）
- ✅ 快速实验（配置驱动）
- ✅ CPU密集型环境
- ✅ 需要环境offload

#### Verl最适合：
- ⚠️ 大规模离线训练（在线支持有限）
- ⚠️ 文本生成为主的任务
- ⚠️ 生产级部署（需要定制）

#### siiRL最适合：
- ✅ VLA模型训练（专门的异步VLA环境）
- ✅ 超大规模并发训练（向量化+异步推理）
- ✅ 复杂的多模态观察（图像+文本+传感器）
- ✅ 需要异步推理引擎（VLLM/SGLang）
- ✅ 机器人操作（Libero等环境支持）

---

## 6. 导航驾驶场景接口设计

### 6.1 四框架NavSim接口对比

#### AReaL NavSim适配器
```python
class AReaLNavSimAdapter(Environment):
    async def ainitialize(self):
        """异步初始化"""
        import asyncio
        await asyncio.to_thread(self.base_env.load_assets)

    def list_tools(self) -> list[dict]:
        return [
            {"name": "reset", "description": "Reset environment"},
            {"name": "step_with_waypoints", "description": "Execute driving"},
            {"name": "get_route_info", "description": "Get route"},
        ]

    async def aexecute(self, tool_name: str, tool_args: dict) -> Any:
        import asyncio
        if tool_name == "reset":
            return await asyncio.to_thread(self.base_env.reset)
        elif tool_name == "step_with_waypoints":
            return await asyncio.to_thread(
                self.base_env.step_with_waypoints,
                tool_args["waypoints"],
                tool_args.get("current_state"),
            )
```

#### RLinf NavSim适配器
```python
class RLinfNavSimAdapter(EnvOffloadMixin):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        from navsim import NavSimEnv
        self.envs = [
            NavSimEnv(scene_id=cfg.scene_id, seed=seed_offset + i)
            for i in range(num_envs)
        ]

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [env.step(act) for env, act in zip(self.envs, actions)]

    def get_state(self) -> bytes:
        import pickle
        return pickle.dumps([env.get_state() for env in self.envs])

    def load_state(self, state: bytes):
        import pickle
        for env, env_state in zip(self.envs, pickle.loads(state)):
            env.load_state(env_state)
```

#### Verl NavSim适配器
```python
class VerlNavSimAdapter:
    def __init__(self, cfg, rank, world_size):
        from navsim import NavSimEnv
        self.base_env = NavSimEnv(scene_id=cfg.scene_id)

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        return self.base_env.step(action)

    def get_all_state_ids(self):
        return [0]  # 简化实现

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        pass  # 简化实现
```

#### siiRL NavSim适配器
```python
# siirl/environment/embodied/adapters/navsim_adapter.py
class NavSimVLAAdapter(BaseVLAEnvironment):
    """siiRL NavSim VLA环境适配器"""

    def __init__(self, scene_id: str, config):
        from navsim import NavSimEnv
        self.base_env = NavSimEnv(scene_id=scene_id)
        self.config = config

    async def reset(self) -> Dict[str, Any]:
        """异步重置，返回多模态观察"""
        import asyncio
        obs = await asyncio.to_thread(self.base_env.reset)

        # 构造多模态观察
        return {
            "camera_images": obs["camera"],
            "lidar": obs.get("lidar"),
            "ego_state": obs["ego_state"],
            "route_description": self._format_route(obs["route"]),
            "goal": obs["goal"],
        }

    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """异步执行驾驶动作"""
        import asyncio

        # 支持多种动作格式
        if "waypoints" in action:
            # 路点格式
            result = await asyncio.to_thread(
                self.base_env.step_with_waypoints,
                action["waypoints"],
                action.get("current_state")
            )
        elif "control" in action:
            # 直接控制格式
            result = await asyncio.to_thread(
                self.base_env.step,
                action["control"]
            )
        else:
            raise ValueError(f"Unknown action format: {action.keys()}")

        return (
            result["observation"],
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"]
        )

    def _format_route(self, route: List[Dict]) -> str:
        """格式化路线信息为文本"""
        descriptions = []
        for i, point in enumerate(route):
            cmd = point["command"]
            pos = point["position"]
            descriptions.append(f"{i+1}. {cmd} to ({pos[0]:.1f}, {pos[1]:.1f})")
        return "Route: " + " -> ".join(descriptions)
```

### 6.2 Bench2Drive接口对比

#### AReaL Bench2Drive适配器
```python
class AReaLBench2DriveAdapter(Environment):
    def list_tools(self) -> list[dict]:
        return [
            {"name": "reset"},
            {"name": "drive", "parameters": {"throttle", "steer", "brake"}},
            {"name": "get_command"},
        ]

    async def aexecute(self, tool_name: str, tool_args: dict) -> Any:
        if tool_name == "drive":
            return await asyncio.to_thread(
                self.base_env.step,
                tool_args  # {"throttle", "steer", "brake"}
            )
        # ... 其他工具
```

#### siiRL Bench2Drive适配器
```python
class Bench2DriveVLAAdapter(BaseVLAEnvironment):
    """siiRL Bench2Drive VLA环境适配器"""

    async def reset(self) -> Dict[str, Any]:
        """异步重置，返回多模态观察"""
        import asyncio
        obs = await asyncio.to_thread(self.base_env.reset)

        return {
            "front_camera": obs["front_camera"],
            "left_camera": obs["left_camera"],
            "right_camera": obs["right_camera"],
            "ego_state": obs["ego_state"],
            "command": obs["command"],  # LEFT, RIGHT, STRAIGHT, LANE_FOLLOW
        }

    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """异步执行控制信号"""
        import asyncio

        # 提取控制信号
        control = action.get("control", {})
        throttle = control.get("throttle", 0.0)
        steer = control.get("steer", 0.0)
        brake = control.get("brake", 0.0)

        result = await asyncio.to_thread(
            self.base_env.step,
            {"throttle": throttle, "steer": steer, "brake": brake}
        )

        return (
            result["observation"],
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"]
        )
```

---

## 7. 总结建议

### 7.1 在线RL框架选择（更新）

| 需求 | 推荐框架 | 理由 |
|------|---------|------|
| **VLA模型在线训练** | **siiRL** | 原生异步VLA环境接口 |
| **最大GPU利用率** | **siiRL** 或 **AReaL** | 都支持异步I/O |
| **快速原型** | **RLinf** | 配置驱动，开箱即用 |
| **大规模生产** | **Verl** | 成熟工具链 |
| **复杂Agent** | **AReaL** | 工作流支持完善 |
| **多模态观察** | **siiRL** | 原生支持 |

### 7.2 环境接口实现建议

#### 统一基础接口

```python
# vla_models/environment/unified_interface.py
class UnifiedDrivingEnvironment(abc.ABC):
    """统一的驾驶环境接口"""

    @abc.abstractmethod
    async def reset(self) -> Dict[str, Any]:
        """异步重置"""

    @abc.abstractmethod
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """异步步骤"""

    @abc.abstractmethod
    def get_observation_space(self) -> Dict[str, Tuple]:
        """返回观察空间规范"""

    @abc.abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """返回动作空间规范"""
```

#### 框架适配器实现

```python
# vla_models/environment/adapters.py

# AReaL适配器
class AReaLNavSimAdapter(UnifiedDrivingEnvironment, Environment):
    # 实现Environment接口（AReaL）
    async def ainitialize(self): ...
    def list_tools(self): ...
    async def aexecute(self, tool_name, tool_args): ...

    # 实现UnifiedDrivingEnvironment接口
    async def reset(self): ...
    async def step(self, action): ...

# siiRL适配器
class SiirLNavSimAdapter(BaseVLAEnvironment):
    # 实现BaseVLAEnvironment接口（siiRL）
    async def reset(self): ...
    async def step(self, action): ...

# RLinf适配器
class RLinfNavSimAdapter(EnvOffloadMixin):
    # 实现EnvOffloadMixin接口（RLinf）
    def reset(self): ...
    def step(self, actions): ...
    def get_state(self): ...
    def load_state(self, state): ...
```

### 7.3 四框架环境接口速查表

| 特性 | AReaL | RLinf | Verl | siiRL |
|------|-------|-------|------|-------|
| **异步接口** | ✅ `async def` | ❌ | ❌ | ✅ `async def` |
| **工具系统** | ✅ `list_tools()` | ❌ | ❌ | ❌ |
| **状态保存** | 自定义 | ✅ `get_state()` | ✅ `get_state()` | 自定义 |
| **向量化** | ❌ | ✅ `SubprocVectorEnv` | ❌ | ✅ `SubprocVectorEnv` |
| **多模态** | ✅ 支持 | ✅ 支持 | ⚠️ 部分 | ✅ 原生 |
| **VLA优化** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **NavSim支持** | 适配器 | 适配器 | 实验性 | 适配器 |
| **Bench2Drive支持** | 适配器 | 适配器 | 实验性 | 适配器 |
