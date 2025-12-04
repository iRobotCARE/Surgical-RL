import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==================== 方案一修改点：导入强化学习库 ====================
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# ===================================================================

from peg_board import BiPegBoard  # 导入您的环境类
import numpy as np

def main():
    # ==================== 方案一修改点：创建强化学习环境 ====================
    print("初始化双机械臂钉板环境...")
    env = BiPegBoard(render_mode=None)  # 训练时关闭渲染以提高性能
    
    # 检查环境是否符合Gymnasium标准
    try:
        check_env(env)
        print("环境检查通过!")
    except Exception as e:
        print(f"环境检查警告: {e}")
    
    # 将环境包装为向量化环境（可选，但推荐）
    env = DummyVecEnv([lambda: env])
    
    # ==================== 方案一修改点：初始化PPO算法 ====================
    print("初始化PPO算法...")
    model = PPO(
        "MultiInputPolicy",           # 使用多层感知机策略
        env,                   # 训练环境
        learning_rate=3e-4,    # 学习率
        n_steps=2048,          # 每次迭代的步数
        batch_size=64,         # 批次大小
        gamma=0.99,            # 折扣因子
        verbose=1,             # 输出训练信息
        tensorboard_log="./tensorboard_logs/",  # TensorBoard日志目录
        device="auto"          # 自动选择GPU/CPU
    )
    
    # ==================== 方案一修改点：设置评估回调 ====================
    print("设置评估回调...")
    eval_env = BiPegBoard(render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs/",
        eval_freq=1000,        # 每1000步评估一次
        deterministic=True,
        render=False
    )
    
    # ==================== 方案一修改点：开始训练 ====================
    print("开始训练PPO算法...")
    total_timesteps = 100000  # 总训练步数
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,  # 使用评估回调
        tb_log_name="ppo_bi_peg_board"  # TensorBoard实验名称
    )
    
    # ==================== 方案一修改点：保存训练好的模型 ====================
    print("训练完成，保存模型...")
    model.save("ppo_bi_peg_board_final")
    
    # ==================== 方案一修改点：演示训练结果 ====================
    print("开始演示训练结果...")
    demo_env = BiPegBoard(render_mode="human")  # 演示时开启渲染
    obs = demo_env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = demo_env.step(action)
        demo_env.render()
        
        if done:
            obs = demo_env.reset()
            print(f"Episode {i} 完成, 奖励: {reward}")
    
    demo_env.close()
    print("演示结束!")

if __name__ == "__main__":
    main()