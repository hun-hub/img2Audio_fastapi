#!/bin/bash

# PID 파일 디렉토리 설정
PID_DIR="../pids"

# API 서버 종료
if [ -f "$PID_DIR/api.pid" ]; then
  API_PID=$(cat "$PID_DIR/api.pid")
  if kill -0 $API_PID > /dev/null 2>&1; then
    kill $API_PID
    echo "API server (PID: $API_PID) has been stopped."
  else
    echo "API server (PID: $API_PID) is not running."
  fi
  rm "$PID_DIR/api.pid"
else
  echo "No PID file for API server."
fi

# Demo 서버 종료
if [ -f "$PID_DIR/demo.pid" ]; then
  DEMO_PID=$(cat "$PID_DIR/demo.pid")
  if kill -0 $DEMO_PID > /dev/null 2>&1; then
    kill $DEMO_PID
    echo "Demo server (PID: $DEMO_PID) has been stopped."
  else
    echo "Demo server (PID: $DEMO_PID) is not running."
  fi
  rm "$PID_DIR/demo.pid"
else
  echo "No PID file for Demo server."
fi