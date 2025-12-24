import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from gymnasium import error
from surrol.gym.surrol_env import SurRoLEnv


class SurRoLGoalEnv(SurRoLEnv):
    """
    A gym GoalEnv wrapper for SurRoL.
    Compatible with gymnasium API: reset(self, *, seed=None, options=None) -> (obs, info)
    """

    def reset(self, *, seed: int = None, options: dict = None):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))

        # Call parent reset using gymnasium style keyword args.
        # Parent may return either obs (old gym) or (obs, info) (gymnasium). Return whatever parent returns.
        return super().reset(seed=seed, options=options)