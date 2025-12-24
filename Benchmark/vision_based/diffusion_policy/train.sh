#!/usr/bin/env bash

# 生成演示数据
# 在根目录下运行

set -e      # 脚本遇到错误时终止

base_path=$(pwd)      # 保存当前路径

FILE_DIR="${base_path}/Benchmark/vision_based/diffusion_policy"
DATA_DIR="${base_path}/Benchmark/state_based/surrol/data/experiment_data"
CONFIG_DIR="${base_path}/Benchmark/vision_based/diffusion_policy"

SCRIPT_NAME="${FILE_DIR}/train.py"
# CONFIG_NAME=state_surrol_needlepick_bet.yaml
CONFIG_NAME="state_surrol_needlepick_diffusion_policy_cnn.yaml"

python "${SCRIPT_NAME}" \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  training.seed=42 training.device=cuda:0 \
  task.dataset.data_root="${DATA_DIR}/needlepick" \
  hydra.run.dir="${base_path}/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}"