""" 展示采集到的轨迹数据
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import os
import gymnasium as gym
import numpy as np
from sim_envs.const import DATA_DIR_PATH

def main(env_name:str):
    env = gym.make(env_name, render_mode='human')
    env.reset()

    # TODO: not implemented


if __name__ == "__main__":
    main()