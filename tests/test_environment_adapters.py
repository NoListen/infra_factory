# tests/test_environment_adapters.py
"""
测试环境适配器

测试NavSim和Bench2Drive的环境适配器实现
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Tuple, Any

from vla_models.interface import VLAForDriving
from vla_models.model_registry import VLAModelRegistry


# ==================== Mock环境 ====================

class MockNavSimEnv:
    """Mock NavSim环境"""

    def __init__(self, scene_id: str, **kwargs):
        self.scene_id = scene_id
        self._initialized = False
        self._step_count = 0

    def load_assets(self):
        """加载资源"""
        self._initialized = True

    def warmup(self):
        """预热"""
        pass

    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self._step_count = 0
        return {
            "camera": [
                Mock(shape=(224, 224, 3)),
                Mock(shape=(224, 224, 3)),
                Mock(shape=(224, 224, 3)),
            ],
            "lidar": np.random.randn(1000, 3),
            "ego_state": {
                "position": np.array([0.0, 0.0, 0.0]),
                "heading": 0.0,
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "route": [
                {
                    "position": np.array([10.0, 0.0, 0.0]),
                    "command": "LANE_FOLLOW",
                }
            ],
            "goal": {"position": np.array([100.0, 0.0, 0.0])},
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        self._step_count += 1

        # 模拟状态更新
        new_position = np.array([float(self._step_count), 0.0, 0.0])

        observation = {
            "camera": [Mock(), Mock(), Mock()],
            "ego_state": {
                "position": new_position,
                "heading": 0.1 * self._step_count,
                "velocity": np.array([1.0, 0.0, 0.0]),
            },
        }

        reward = 1.0 if self._step_count < 10 else -1.0
        done = self._step_count >= 100
        info = {"step": self._step_count}

        return observation, reward, done, info

    def step_with_waypoints(
        self, waypoints: np.ndarray, current_state: Dict
    ) -> Tuple[Dict, float, bool, Dict]:
        """使用路点执行步骤"""
        return self.step(waypoints.flatten())

    def get_route_info(self) -> list:
        """获取路线信息"""
        return [
            {"position": np.array([10.0, 0.0, 0.0]), "command": "LANE_FOLLOW"}
        ]

    def close(self):
        """关闭环境"""
        self._initialized = False

    def get_state(self) -> bytes:
        """获取状态"""
        import pickle
        return pickle.dumps(self.__dict__)

    def load_state(self, state: bytes):
        """加载状态"""
        import pickle
        self.__dict__.update(pickle.loads(state))


# ==================== AReaL NavSim适配器 ====================

class AReaLNavSimAdapter:
    """
    AReaL NavSim适配器

    实现AReaL的Environment接口
    """

    def __init__(self, scene_id: str, **kwargs):
        from navsim import NavSimEnv
        self.base_env = NavSimEnv(scene_id=scene_id, **kwargs)
        self._initialized = False

    async def ainitialize(self):
        """异步初始化"""
        import asyncio

        # 异步加载资源
        await asyncio.to_thread(self.base_env.load_assets)

        # 预热模型
        await asyncio.to_thread(self.base_env.warmup)

        self._initialized = True

    def list_tools(self) -> list[dict]:
        """列出可用工具"""
        return [
            {
                "name": "reset",
                "description": "Reset the environment",
                "parameters": {},
            },
            {
                "name": "step_with_waypoints",
                "description": "Execute driving with waypoints",
                "parameters": {
                    "waypoints": {
                        "type": "array",
                        "description": "Array of waypoints [N, 3]",
                    },
                    "current_state": {
                        "type": "object",
                        "description": "Current ego vehicle state",
                    },
                },
            },
            {
                "name": "get_route_info",
                "description": "Get current route information",
                "parameters": {},
            },
        ]

    async def aexecute(self, tool_name: str, tool_args: dict) -> Any:
        """异步执行工具"""
        import asyncio

        if tool_name == "reset":
            result = await asyncio.to_thread(self.base_env.reset)
        elif tool_name == "step_with_waypoints":
            result = await asyncio.to_thread(
                self.base_env.step_with_waypoints,
                tool_args["waypoints"],
                tool_args.get("current_state"),
            )
        elif tool_name == "get_route_info":
            result = await asyncio.to_thread(self.base_env.get_route_info)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        return result

    async def aclose(self):
        """异步关闭"""
        import asyncio
        await asyncio.to_thread(self.base_env.close)
        self._initialized = False

    # 同步接口（用于兼容）
    def reset(self) -> Dict[str, Any]:
        return self.base_env.reset()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        return self.base_env.step(action)

    def close(self):
        self.base_env.close()


# ==================== RLinf NavSim适配器 ====================

class RLinfNavSimAdapter:
    """
    RLinf NavSim适配器

    实现RLinf的EnvOffloadMixin接口
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.scene_id = cfg.get("scene_id", "scene_001")
        self.num_envs = num_envs
        self.seed_offset = seed_offset

        # 创建多个环境实例（向量化）
        from navsim import NavSimEnv
        self.envs = [
            NavSimEnv(self.scene_id, seed=seed_offset + i)
            for i in range(num_envs)
        ]

    def reset(self):
        """重置所有环境"""
        return [env.reset() for env in self.envs]

    def step(self, actions: np.ndarray):
        """
        批量执行步骤

        参数:
            actions: [num_envs, action_dim]
        """
        results = []
        for env, action in zip(self.envs, actions):
            results.append(env.step(action))
        return results

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

    def get_state(self) -> bytes:
        """保存状态"""
        import pickle
        return pickle.dumps({
            "scene_id": self.scene_id,
            "num_envs": self.num_envs,
            "seed_offset": self.seed_offset,
            "envs": [env.get_state() for env in self.envs],
        })

    def load_state(self, state: bytes):
        """加载状态"""
        import pickle
        state_dict = pickle.loads(state)

        self.scene_id = state_dict["scene_id"]
        self.num_envs = state_dict["num_envs"]
        self.seed_offset = state_dict["seed_offset"]

        for env, env_state in zip(self.envs, state_dict["envs"]):
            env.load_state(env_state)


# ==================== Verl NavSim适配器 ====================

class VerlNavSimAdapter:
    """
    Verl NavSim适配器（实验性）
    """

    def __init__(self, cfg, rank, world_size):
        self.scene_id = cfg.get("scene_id", "scene_001")
        self.rank = rank
        self.world_size = world_size

        from navsim import NavSimEnv
        self.base_env = NavSimEnv(self.scene_id)

    def reset(self) -> Dict[str, Any]:
        return self.base_env.reset()

    def step(self, action: np.ndarray):
        return self.base_env.step(action)

    def close(self):
        self.base_env.close()

    # Verl可能需要额外的方法
    def get_all_state_ids(self):
        """获取所有可用状态ID（Verl特有）"""
        return [0]  # 简化实现

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        """重置到特定状态（Verl特有）"""
        # 简化实现
        pass


# ==================== 测试 ====================

class TestAReaLNavSimAdapter:
    """测试AReaL NavSim适配器"""

    @pytest.mark.asyncio
    async def test_ainitialize(self):
        """测试异步初始化"""
        adapter = AReaLNavSimAdapter("scene_001")

        await adapter.ainitialize()

        assert adapter._initialized == True

    def test_list_tools(self):
        """测试列出工具"""
        adapter = AReaLNavSimAdapter("scene_001")

        tools = adapter.list_tools()

        assert len(tools) == 3
        tool_names = [tool["name"] for tool in tools]
        assert "reset" in tool_names
        assert "step_with_waypoints" in tool_names
        assert "get_route_info" in tool_names

    @pytest.mark.asyncio
    async def test_aexecute_reset(self):
        """测试异步执行reset"""
        adapter = AReaLNavSimAdapter("scene_001")
        await adapter.ainitialize()

        result = await adapter.aexecute("reset", {})

        assert "camera" in result
        assert "ego_state" in result

    @pytest.mark.asyncio
    async def test_aexecute_step_with_waypoints(self, sample_waypoints, sample_driving_state):
        """测试异步执行路点步骤"""
        adapter = AReaLNavSimAdapter("scene_001")
        await adapter.ainitialize()

        result = await adapter.aexecute(
            "step_with_waypoints",
            {
                "waypoints": sample_waypoints,
                "current_state": sample_driving_state,
            }
        )

        assert isinstance(result, tuple) or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_aexecute_unknown_tool(self):
        """测试异步执行未知工具"""
        adapter = AReaLNavSimAdapter("scene_001")

        with pytest.raises(ValueError, match="Unknown tool"):
            await adapter.aexecute("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_aclose(self):
        """测试异步关闭"""
        adapter = AReaLNavSimAdapter("scene_001")
        await adapter.ainitialize()

        await adapter.aclose()

        assert adapter._initialized == False

    def test_sync_reset(self):
        """测试同步reset"""
        adapter = AReaLNavSimAdapter("scene_001")

        obs = adapter.reset()

        assert "ego_state" in obs

    def test_sync_step(self):
        """测试同步step"""
        adapter = AReaLNavSimAdapter("scene_001")

        obs = adapter.reset()
        action = np.random.randn(15)  # 随机动作

        next_obs, reward, done, info = adapter.step(action)

        assert "ego_state" in next_obs
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)


class TestRLinfNavSimAdapter:
    """测试RLinf NavSim适配器"""

    def test_init(self):
        """测试初始化"""
        cfg = {"scene_id": "scene_001"}
        adapter = RLinfNavSimAdapter(cfg, num_envs=4, seed_offset=0, total_num_processes=1, worker_info=None)

        assert adapter.num_envs == 4
        assert len(adapter.envs) == 4

    def test_reset(self):
        """测试批量reset"""
        cfg = {"scene_id": "scene_001"}
        adapter = RLinfNavSimAdapter(cfg, num_envs=3, seed_offset=0, total_num_processes=1, worker_info=None)

        observations = adapter.reset()

        assert len(observations) == 3

    def test_step(self):
        """测试批量step"""
        cfg = {"scene_id": "scene_001"}
        adapter = RLinfNavSimAdapter(cfg, num_envs=3, seed_offset=0, total_num_processes=1, worker_info=None)

        actions = np.random.randn(3, 15)
        results = adapter.step(actions)

        assert len(results) == 3

    def test_get_state(self):
        """测试保存状态"""
        cfg = {"scene_id": "scene_001"}
        adapter = RLinfNavSimAdapter(cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)

        state = adapter.get_state()

        assert isinstance(state, bytes)

    def test_load_state(self):
        """测试加载状态"""
        cfg = {"scene_id": "scene_001"}
        adapter = RLinfNavSimAdapter(cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)

        # 保存状态
        state = adapter.get_state()

        # 创建新适配器并恢复状态
        new_adapter = RLinfNavSimAdapter(cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)
        new_adapter.load_state(state)

        assert new_adapter.scene_id == adapter.scene_id


class TestVerlNavSimAdapter:
    """测试Verl NavSim适配器"""

    def test_init(self):
        """测试初始化"""
        cfg = {"scene_id": "scene_001"}
        adapter = VerlNavSimAdapter(cfg, rank=0, world_size=1)

        assert adapter.scene_id == "scene_001"
        assert adapter.rank == 0

    def test_reset(self):
        """测试reset"""
        cfg = {"scene_id": "scene_001"}
        adapter = VerlNavSimAdapter(cfg, rank=0, world_size=1)

        obs = adapter.reset()

        assert "ego_state" in obs

    def test_step(self):
        """测试step"""
        cfg = {"scene_id": "scene_001"}
        adapter = VerlNavSimAdapter(cfg, rank=0, world_size=1)

        obs = adapter.reset()
        action = np.random.randn(15)

        next_obs, reward, done, info = adapter.step(action)

        assert "ego_state" in next_obs

    def test_get_all_state_ids(self):
        """测试获取所有状态ID"""
        cfg = {"scene_id": "scene_001"}
        adapter = VerlNavSimAdapter(cfg, rank=0, world_size=1)

        state_ids = adapter.get_all_state_ids()

        assert state_ids == [0]

    def test_reset_envs_to_state_ids(self):
        """测试重置到特定状态"""
        cfg = {"scene_id": "scene_001"}
        adapter = VerlNavSimAdapter(cfg, rank=0, world_size=1)

        # 应该不抛出异常
        adapter.reset_envs_to_state_ids([0, 1, 2], [0, 1, 2])


# ==================== 环境注册测试 ====================

class TestEnvironmentAdapterRegistration:
    """测试环境适配器注册"""

    def setup_method(self):
        """每个测试前清空环境适配器注册"""
        VLAModelRegistry._env_adapters = {
            "navsim": {"areal": None, "rlinf": None, "verl": None},
            "bench2drive": {"areal": None, "rlinf": None, "verl": None},
        }

    def test_register_navsim_adapters(self):
        """测试注册NavSim适配器"""
        VLAModelRegistry.register_env_adapter("navsim", "areal", AReaLNavSimAdapter)
        VLAModelRegistry.register_env_adapter("navsim", "rlinf", RLinfNavSimAdapter)
        VLAModelRegistry.register_env_adapter("navsim", "verl", VerlNavSimAdapter)

        # 验证注册
        areal_adapter = VLAModelRegistry.get_env_adapter("navsim", "areal")
        rlinf_adapter = VLAModelRegistry.get_env_adapter("navsim", "rlinf")
        verl_adapter = VLAModelRegistry.get_env_adapter("navsim", "verl")

        assert areal_adapter == AReaLNavSimAdapter
        assert rlinf_adapter == RLinfNavSimAdapter
        assert verl_adapter == VerlNavSimAdapter

    def test_list_environments(self):
        """测试列出支持的环境"""
        envs = VLAModelRegistry.list_environments()

        assert "navsim" in envs
        assert "bench2drive" in envs


# ==================== 模拟环境测试 ====================

class TestMockNavSimEnv:
    """测试Mock NavSim环境"""

    def test_init(self):
        """测试初始化"""
        env = MockNavSimEnv("scene_001")

        assert env.scene_id == "scene_001"
        assert env._initialized == False

    def test_reset(self):
        """测试reset"""
        env = MockNavSimEnv("scene_001")

        obs = env.reset()

        assert "camera" in obs
        assert "ego_state" in obs
        assert "route" in obs
        assert "goal" in obs
        assert len(obs["camera"]) == 3

    def test_step(self):
        """测试step"""
        env = MockNavSimEnv("scene_001")

        obs = env.reset()
        action = np.random.randn(15)

        next_obs, reward, done, info = env.step(action)

        assert "ego_state" in next_obs
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self):
        """测试多步执行"""
        env = MockNavSimEnv("scene_001")

        obs = env.reset()

        for i in range(10):
            action = np.random.randn(15)
            obs, reward, done, info = env.step(action)

            if done:
                break

        assert env._step_count >= 10

    def test_state_save_load(self):
        """测试状态保存和加载"""
        env = MockNavSimEnv("scene_001")

        env.reset()
        env.step(np.random.randn(15))

        # 保存状态
        state = env.get_state()
        assert isinstance(state, bytes)

        # 创建新环境并恢复状态
        new_env = MockNavSimEnv("scene_001")
        new_env.load_state(state)

        assert new_env._step_count == env._step_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
