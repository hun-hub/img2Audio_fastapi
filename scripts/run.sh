#!/bin/bash

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate api

# 로그 디렉토리 확인 및 생성
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# PID 파일 디렉토리 확인 및 생성
PID_DIR="pids"
if [ ! -d "$PID_DIR" ]; then
  mkdir -p "$PID_DIR"
fi

# API 서버 실행 및 PID 저장
nohup python main_api.py > logs/api_log.out 2>&1 &
echo $! > "$PID_DIR/api.pid"

# Demo 서버 실행 및 PID 저장
nohup python main_demo.py > logs/demo_log.out 2>&1 &
echo $! > "$PID_DIR/demo.pid"

echo "Servers are running in the background."