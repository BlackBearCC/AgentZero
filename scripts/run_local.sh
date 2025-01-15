#!/bin/bash

# 设置环境变量
export ENV=dev
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 启动服务
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload 