import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import state_based.surrol.gym
import torch.multiprocessing as mp
import hydra
from trainers.rl_trainer import RLTrainer

@hydra.main(version_base=None, config_path="./configs", config_name="train_ddpg")
def main(cfg):
    exp = RLTrainer(cfg)
    exp.train()
    # exp.resume_train()



if __name__ == "__main__":
    # For multiprocessing with CUDA, 'spawn' is often safer than 'fork'
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    main()

    #python /home/ma/桌面/SurRoL-code-explainning/Benchmark/state_based/rl/train_rl.py device=cpu