""" Wrapper to convert a standard Gym environment into a GoalEnv-like environment. """

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

    
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_envs.gymnasium_utils.sur_env import SurRoLEnv

class GoalEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        print(f"--- INITIALIZING THE CORRECT GoalEnvWrapper! It is wrapping an env of type: {type(env)} ---")
        super().__init__(env)

        original_observation_space = env.unwrapped.observation_space
        goal_shape = env.unwrapped.goal.shape

        self.observation_space = spaces.Dict({
            'observation': original_observation_space,
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
        })

    def _create_goal_observation(self, obs):
        # 创建 goal-based 观测
        observation = obs
        achieved_goal, _ = self.env.unwrapped._get_task_obs()   # 物体位置
        desired_goal = self.env.unwrapped.goal         # 目标位置

        goal_observation = {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        return goal_observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.unwrapped.step(action)
        # 将 observation 包装成字典格式
        goal_obs = self._create_goal_observation(observation)
        return goal_obs, reward, terminated, truncated, info
    
    # =======================================================================================================================
    # def reset(self, **kwargs):
    #     observation, info = self.env.unwrapped.reset(**kwargs)

    #     goal_observation = self._create_goal_observation(observation)

    #     return goal_observation, info
    # =======================================================================================================================
    
    def reset(self, seed=None, options=None):  # 🛠️ 添加标准参数
        """
        包装器的 reset 方法
        """
        # 🛠️ 确保传递正确的参数
        observation, info = self.env.unwrapped.reset(seed=seed, options=options)

        goal_observation = self._create_goal_observation(observation)

        return goal_observation, info  # ✅ 返回格式正确

# 在重置时强制进行合规性检查，确保观察空间是合法的 GoalEnv。
class SurRoLGoalEnv(SurRoLEnv):
    """
    A gym GoalEnv wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    # def reset(self):
    #     # Enforce that each GoalEnv uses a Goal-compatible observation space.
    #     if not isinstance(self.observation_space, gym.spaces.Dict):
    #         raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
    #     for key in ['observation', 'achieved_goal', 'desired_goal']:
    #         if key not in self.observation_space.spaces:
    #             raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
    #     return super().reset()
    

    def reset(self, seed=None, options=None):
        """
        重置环境，开始新的回合
        """
        # 1. 重要：调用父类（通常是 gym.Env）的 reset 来设置随机种子
        # 这行代码确保了环境内部的随机数生成器被正确初始化
        super().reset(seed=seed)  # 这行是修复的核心

        # 2. 您原有的 GoalEnv 合规性检查逻辑（通常不需要改动）
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))

        # 3. 您原有的环境重置逻辑（例如：重置机器人、物体位置等）
        # ... 您原来的重置代码 ...

        # 4. 获取初始观测和信息
        observation = self._get_obs()
        info = self._get_info()  # 确保这个方法返回一个字典（即使是空字典）

        # 5. 返回符合 Gymnasium API 的元组 (observation, info)
        return observation, info
    