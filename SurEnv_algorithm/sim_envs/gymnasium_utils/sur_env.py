""" A gym Env class for SurEnv"""
# 提供了一个完整的、符合标准规范的接口，可以用于部署和测试强化学习算法。

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

    
import time,os
import socket
import warnings

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import pybullet as p
import pybullet_data
import pkgutil
from sim_envs.utils.pybullet_utils import (
    p_step,
    render_image,
)
import numpy as np
from sim_envs.const import ROOT_DIR_PATH, ASSET_DIR_PATH


RENDER_HEIGHT = 480  # train
RENDER_WIDTH = 640
# RENDER_HEIGHT = 1080  # record
# RENDER_WIDTH = 1920


class SurRoLEnv(gym.Env):
    """
    A gym Env wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'img_array']}

    def __init__(self, render_mode: str = None, cid: int = -1):
        # rendering and connection options  若人类观察模式,加载GUI界面
        self._render_mode = render_mode
        # render_mode = 'human'
        # if render_mode == 'human':
        #     self.cid = p.connect(p.SHARED_MEMORY)
        #     if self.cid < 0:
        if render_mode == 'human':
            if cid < 0:
                self.cid = p.connect(p.GUI)
            else:
                self.cid = cid
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            if cid < 0:
                self.cid = p.connect(p.DIRECT)
            else:
                self.cid = cid
            # See PyBullet Quickstart Guide Synthetic Camera Rendering
            # TODO: no light when using direct without egl
            if socket.gethostname().startswith('pc') or True:
                # TODO: not able to run on remote server
                egl = pkgutil.get_loader('eglRenderer')
                plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        # camera related setting  相机以及投影初始化
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0, 0, 0.2),
                                                                distance=1.5,
                                                                yaw=90,
                                                                pitch=-36,
                                                                roll=0,
                                                                upAxisIndex=2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov=45,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=20.0)
        # additional settings  加载物理世界,光源/重力/地板/木料纹路//物体在世界中存在方式(固定/浮动)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        p.setGravity(0, 0, -9.81)
        plane=p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"), (0, 0, -0.001))
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        p.changeVisualShape(plane, -1, textureUniqueId=wood)

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # self.seed()
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================
        # 训练参数种子,保证实验可重复===========================================================================================================================

        if hasattr(self, 'seed'):
            # 保持向后兼容性，但推荐使用新的reset(seed=...)方式
            warnings.warn("The 'seed' method is deprecated. Use 'reset(seed=...)' instead.", DeprecationWarning)
        else:
            # 确保环境有基本的随机数生成器
            self.np_random, _ = gym.utils.seeding.np_random()

# 训练参数种子,保证实验可重复===========================================================================================================================

        # self.actions = []  # only for demo
        self._env_setup()
        self.goal = self._sample_goal()  # tasks are all implicitly goal-based
        self._sample_goal_callback()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(self.action_size,), dtype='float32')
        if isinstance(obs, np.ndarray):
            # gym.Env
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        elif isinstance(obs, dict):
            # gym.GoalEnv
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        else:
            raise NotImplementedError

        # @taohuang
        self._duration = 0.2  # important for mini-steps

    # 强化学习的核心函数,进行一次完整的交互
    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # time0 = time.time()
        self._set_action(action)
        # time1 = time.time()
        # TODO: check the best way to step simulation
        p_step(self._duration)

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()

        done = False
        truncated = False   # 新增：表示因时间限制等外部条件被截断
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        if isinstance(obs, dict):
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, truncated, info

    # def reset(self):        # 重置
    #     # reset scene in the corresponding file
    #     p.resetSimulation()
    #     p.setGravity(0, 0, -9.81)
    #     p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

    #     # Temporarily disable rendering to load scene faster.
    #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    #     # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

    #     plane=p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"), (0, 0, -0.001))
    #     wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
    #     p.changeVisualShape(plane, -1, textureUniqueId=wood)
    #     self._env_setup()
    #     p_step(0.25)
    #     self.goal = self._sample_goal().copy()
    #     self._sample_goal_callback()

    #     # Re-enable rendering.
    #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    #     obs = self._get_obs()
    #     return obs


    def reset(self, seed=None, options=None):  # 🛠️ 修改点1：添加标准参数签名
        """
        重置环境，开始新的回合
        Args:
            seed (int, optional): 用于环境随机数生成器的种子，确保实验可复现
            options (dict, optional): 额外的重置选项（可自定义初始状态等）
        Returns:
            tuple: 包含 (observation, info) 的元组，符合 Gymnasium 标准
        """
        # 🛠️ 修改点2：调用父类 reset(seed=seed) 初始化随机数生成器（至关重要！）
        # 这行代码确保环境内部的随机数生成器（self.np_random）被正确设置
        super().reset(seed=seed)

        # === 您原有的重置逻辑保持不变 ===
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        plane = p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"), (0, 0, -0.001))
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        p.changeVisualShape(plane, -1, textureUniqueId=wood)
        self._env_setup()
        p_step(0.25)
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # === 原有逻辑结束 ===

        # 🛠️ 修改点3：返回符合 Gymnasium API 的元组 (observation, info)
        obs = self._get_obs()
        info = {}  # 可在此字典中添加额外的重置信息（如自定义初始状态参数等）
        return obs, info
    

    def close(self):
        if self.cid >= 0:
            p.disconnect()
            self.cid = -1

    def render(self, mode='rgb_array'):
        self._render_callback(mode)
        if mode == "human":
            return np.array([])
        # TODO: check the way to render image
        rgb_array, mask = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       self._view_matrix, self._proj_matrix)
        if mode == 'rgb_array':
            return rgb_array
        else:
            return rgb_array, mask

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def _env_setup(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _set_action(self, action):
        """ Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError

    def _sample_goal(self):
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _sample_goal_callback(self):
        """ For goal visualization, etc.
        """
        pass

    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    @property
    def action_size(self):
        raise NotImplementedError

    # 一个预设动作
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError

    # def test(self, horizon=100):
    # original 100 steps; change to 200 for bimanual need to be reverted
    def test(self, horizon=200):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        steps, done = 0, False
        obs = self.reset()
        while not done and steps <= 9999900:
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
            # print('action:', action)
            obs, reward, done, info = self.step(action)
            if isinstance(obs, dict):
                print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
            else:
                print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            toc = time.time()
            print(" -> step time: {:.4f}".format(toc - tic))
            time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))

    def __del__(self):
        self.close()
