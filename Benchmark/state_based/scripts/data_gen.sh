#!/usr/bin/env bash

# 生成演示数据
# 在根目录下运行

set -e      # 脚本遇到错误时终止

base_path=$(pwd)      # 保存当前路径

DATA_DIR="${base_path}/Benchmark/state_based/surrol/data"

SCRIPT_NAME="data_generation.py"

python "${DATA_DIR}/${SCRIPT_NAME}" --env NeedlePickDataRL-v0