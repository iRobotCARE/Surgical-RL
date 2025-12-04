#!/home/zack/anaconda3/envs/Sur/bin/python
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from peg_board import BiPegBoard

def diagnose_reset():
    print("创建环境...")
    env = BiPegBoard(render_mode=None)
    
    print("测试 reset 方法...")
    # 先不进行解包，直接查看返回值
    result = env.reset(seed=42)
    print(f"reset() 返回值: {result}")
    print(f"reset() 返回值类型: {type(result)}")
    
    if result is None:
        print("❌ reset() 返回了 None")
        # 检查环境类的 reset 方法
        print("检查环境类结构...")
        print("BiPegBoard MRO:", BiPegBoard.__mro__)
        
        # 检查每个类的 reset 方法
        for cls in BiPegBoard.__mro__:
            if hasattr(cls, 'reset'):
                print(f"{cls.__name__} 有 reset 方法: {cls.reset}")
            else:
                print(f"{cls.__name__} 没有 reset 方法")
    
    elif isinstance(result, tuple) and len(result) == 2:
        print("✅ reset() 返回正确的元组格式")
        obs, info = result
        print(f"观测类型: {type(obs)}")
        print(f"信息类型: {type(info)}")
    else:
        print(f"❌ reset() 返回了不正确的格式: {result}")
    
    env.close()

if __name__ == "__main__":
    diagnose_reset()