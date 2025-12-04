import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import time
import numpy as np
import random
import pybullet as p
import pybullet_data

from psm_env import PsmsEnv, goal_distance
from sim_envs.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from sim_envs.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
from sim_envs.const import ASSET_DIR_PATH


class BiPegBoard(PsmsEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS1 = ((0.47, 0.66), (-0., 0.10), (0.606, 0.785))
    WORKSPACE_LIMITS2 = ((0.47, 0.66), (-0.15, 0.), (0.606, 0.785))
    SCALING = 5.
    POSE_PSM1 = ((0.05, 0.25, 0.8224), (0, 0, -(90 + 20) / 180 * np.pi))
    POSE_PSM2 = ((0.05, -0.26, 0.8524), (0, 0, -(90 - 20) / 180 * np.pi))

    def __init__(self, render_mode=None, cid=-1):
        # ==================== 方案二接口：可修改初始化参数 ====================
        # 如需自定义环境参数，可在此添加，例如：
        # self.max_episode_steps = 1000  # 最大回合步数
        # self.distance_threshold = 0.-5  # 成功阈值
        super().__init__(render_mode, cid)
        
        # ==================== 方案二接口：可添加自定义观测空间 ====================
        # 如需扩展观测空间，可在此修改，例如：
        # self.observation_space = self._get_expanded_observation_space()

    def _env_setup(self):
        super()._env_setup()  # 调用父类设置
        # 您的现有_env_setup代码保持不变...
        # [原有代码保持不动]

    def _set_action(self, action: np.ndarray):
        # 您的现有_set_action代码保持不变...
        # [原有代码保持不动]

    def _sample_goal(self) -> np.ndarray:
        # 您的现有_sample_goal代码保持不变...
        # [原有代码保持不动]

    def _sample_goal_callback(self):
        # 您的现有_sample_goal_callback代码保持不变...
        # [原有代码保持不动]

    def _get_obs(self) -> dict:
        """获取环境观测"""
        # ==================== 方案二接口：可自定义观测信息 ====================
        # 如需添加额外观测信息，可在此修改，例如：
        # obs = super()._get_obs()
        # # 添加自定义观测
        # custom_obs = self._get_custom_observation()
        # obs['custom_observation'] = custom_obs
        # return obs
        
        # 当前使用父类的观测
        return super()._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        """计算奖励函数"""
        # ==================== 方案二接口：可自定义奖励函数 ====================
        # 如需实现自定义奖励函数，可重写此方法，例如：
        # 
        # # 稀疏奖励示例
        # distance = goal_distance(achieved_goal, desired_goal)
        # if distance < self.distance_threshold:
        #     return 0.0  # 成功
        # else:
        #     return -1.0  # 失败
        # 
        # # 密集奖励示例
        # distance = goal_distance(achieved_goal, desired_goal)
        # base_reward = -distance  # 基础奖励：负距离
        # 
        # # 添加时间惩罚
        # time_penalty = -0.01
        # 
        # # 添加成功奖励
        # success_bonus = 10.0 if distance < self.distance_threshold else 0.0
        # 
        # return base_reward + time_penalty + success_bonus
        
        # 当前使用父类的奖励函数
        return super().compute_reward(achieved_goal, desired_goal, info)

    def _is_success(self, achieved_goal, desired_goal) -> bool:
        """判断是否成功"""
        # ==================== 方案二接口：可自定义成功条件 ====================
        # 如需修改成功条件，可重写此方法，例如：
        # distance = goal_distance(achieved_goal, desired_goal)
        # return distance < self.custom_distance_threshold
        
        # 当前使用父类的成功条件
        return super()._is_success(achieved_goal, desired_goal)

    def step(self, action):
        """环境步进"""
        # ==================== 方案二接口：可自定义步进逻辑 ====================
        # 如需在每一步添加自定义逻辑，可重写此方法，例如：
        # obs, reward, done, info = super().step(action)
        # 
        # # 添加自定义奖励组件
        # custom_reward = self._compute_custom_reward(obs, action)
        # reward += custom_reward
        # 
        # # 添加自定义终止条件
        # if self._check_custom_termination():
        #     done = True
        # 
        # return obs, reward, done, info
        
        # 当前使用父类的步进逻辑
        return super().step(action)

    def reset(self):
        """重置环境"""
        # ==================== 方案二接口：可自定义重置逻辑 ====================
        # 如需在重置时添加自定义逻辑，可重写此方法，例如：
        # obs = super().reset()
        # 
        # # 重置自定义变量
        # self.custom_variable = 0
        # 
        # # 修改初始状态
        # self._set_custom_initial_state()
        # 
        # return obs
        
        # 当前使用父类的重置逻辑
        return super().reset()

    # ==================== 方案二接口：可添加自定义方法 ====================
    # 如需添加环境特有的方法，可在此定义，例如：
    # 
    # def _compute_custom_reward(self, obs, action):
    #     """计算自定义奖励组件"""
    #     # 实现您的自定义奖励逻辑
    #     return 0.0
    # 
    # def _check_custom_termination(self):
    #     """检查自定义终止条件"""
    #     # 实现您的自定义终止逻辑
    #     return False
    # 
    # def _get_expanded_observation_space(self):
    #     """获取扩展的观测空间"""
    #     # 实现您的自定义观测空间
    #     return self.observation_space

    # 您的其他现有方法保持不变...
    def _meet_contact_constraint_requirement(self):
        # 现有代码保持不变...
        pass

    def get_oracle_action(self, obs) -> np.ndarray:
        # 现有代码保持不变...
        pass

    @property
    def waypoints(self):
        # 现有代码保持不变...
        pass


if __name__ == "__main__":
    # 测试代码保持不变...
    print(ASSET_DIR_PATH)
    env = BiPegBoard(render_mode='human')
    env.test()
    time.sleep(2000)