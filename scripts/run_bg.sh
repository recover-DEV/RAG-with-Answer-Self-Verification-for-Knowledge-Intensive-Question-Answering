#!/usr/bin/env bash
# 后台跑任意命令，日志按"标签+时间"落盘，pid 写文件方便 kill。
#
# 用法：
#   bash scripts/run_bg.sh gen_e01 python scripts/03_generate.py --split test
#   bash scripts/run_bg.sh nli python scripts/05_verify.py
#
# 收工三件事：
#   tail -f logs/gen_e01_2026xxxx.log     # 看进度
#   cat logs/gen_e01.pid                  # 看 pid
#   kill $(cat logs/gen_e01.pid)          # 停任务（脚本自己带断点续跑，停了重跑即可）
set -euo pipefail

TAG=${1:?用法: bash scripts/run_bg.sh <标签> <命令...>}
shift

mkdir -p logs
TS=$(date +%Y%m%d-%H%M)
LOG="logs/${TAG}_${TS}.log"

nohup "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "logs/${TAG}.pid"

echo "已启动：$*"
echo "  pid : $PID  （logs/${TAG}.pid）"
echo "  log : $LOG"
echo "  跟踪: tail -f $LOG"
echo "  停止: kill \$(cat logs/${TAG}.pid)"
