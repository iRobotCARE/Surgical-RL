import surrol.gym
import gym
import traceback

# 关键：导入你项目中的环境定义文件，
# 这一步会自动向 Gym 注册你的自定义环境。
# 根据你的报错信息，可能是下面这个文件或类似的文件。
# 请根据你的项目结构进行调整。
try:
    # 尝试导入包含环境注册逻辑的模块
    # 根据你的报错路径，这个导入应该是正确的
    from surrol.tasks import psm_env_RL
except ImportError as e:
    print("导入自定义环境失败！请确保：")
    print("1. 你是从项目的根目录运行此脚本。")
    print("2. surrol 模块在你的 PYTHONPATH 中。")
    print(f"原始错误: {e}")
    exit()


# --- 请在这里修改为你要测试的环境名称 ---
# 这个名称通常是在你的项目中定义的，例如 'NeedlePick-v0', 'PegTransfer-v0' 等
# 'NeedlePick-v0' 是一个基于你路径的猜测，请务必确认！
ENV_NAME = 'NeedlePickRL-v0'
RENDER_MODE = 'rgb_array'  # 使用和你报错代码中完全相同的渲染模式

print("-" * 50)
print(f"正在尝试创建环境: '{ENV_NAME}'")
print(f"渲染模式: '{RENDER_MODE}'")
print("-" * 50)

try:
    # 这是测试的核心：调用 gym.make()
    # 我们将完全复现你报错的那行代码
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)

    # 如果代码能成功执行到这里，说明环境创建成功
    print("\n✅ ✅ ✅ 恭喜！环境创建成功！ ✅ ✅ ✅")
    print("现在将尝试重置环境并关闭。")

    # 作为良好实践，我们重置并关闭环境
    env.reset()
    env.close()

    print("\n环境已成功重置和关闭。测试通过！")
    print("-" * 50)


except Exception as e:
    # 捕获标准的Python异常（但无法捕获 Segmentation Fault）
    print("\n❌ ❌ ❌ 环境创建失败！发生了一个Python异常！ ❌ ❌ ❌")
    print("这不是一个段错误（Segmentation Fault），但仍然是一个问题。")
    print("详细错误信息如下:")
    traceback.print_exc()
    print("-" * 50)