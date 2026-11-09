#!/usr/bin/env bash
# 启动 vLLM 的 OpenAI 兼容服务。
#
# 用法（在项目根目录）：
#   bash scripts/serve_vllm.sh                     # 默认 Qwen2.5-7B-Instruct, 端口 8000
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ QUANT=awq bash scripts/serve_vllm.sh
#   PORT=8001 GPU_UTIL=0.85 bash scripts/serve_vllm.sh
#
# 必须在 tmux/screen 里跑，或配合 scripts/run_bg.sh：
#   tmux new -s vllm
#   bash scripts/serve_vllm.sh
#   # Ctrl+B 然后 D 脱离；回来用 tmux a -t vllm
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
GPU_UTIL=${GPU_UTIL:-0.88}
MAX_LEN=${MAX_LEN:-4096}
MAX_BATCH=${MAX_BATCH:-16}
QUANT=${QUANT:-}
DTYPE=${DTYPE:-bfloat16}

mkdir -p logs
LOG="logs/vllm_$(date +%Y%m%d-%H%M).log"

echo "== 启动参数 =="
echo "  model      : $MODEL"
echo "  port       : $HOST:$PORT"
echo "  dtype      : $DTYPE"
echo "  gpu-util   : $GPU_UTIL   (24GB 卡留 12% 给 CUDA context 和碎片)"
echo "  max-len    : $MAX_LEN"
echo "  log        : $LOG"

ARGS=(
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_BATCH"
  --enable-prefix-caching
)
if [[ -n "$QUANT" ]]; then
  ARGS+=(--quantization "$QUANT")
  echo "  quant      : $QUANT"
fi

# 显存不够时按顺序加这三个（见 docs/服务器部署指南.md §7.1）：
#   --enforce-eager            关掉 CUDA graph，省几百 MB
#   --swap-space 4             降 CPU 交换空间（默认 4-8GB）
#   --max-model-len 2048       降上下文长度

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" 2>&1 | tee "$LOG"
