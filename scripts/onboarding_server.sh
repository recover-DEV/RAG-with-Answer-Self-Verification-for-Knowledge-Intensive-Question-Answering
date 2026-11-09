#!/usr/bin/env bash
# 服务器上第一次进来时跑这个：只做体检，不改任何东西。
#
#   bash scripts/onboarding_server.sh
#
# 它会告诉你"现在卡在哪一步"，以及下一步该看文档的哪一节。
set -uo pipefail

ok=0; bad=0
pass() { echo "  [OK]   $1"; ok=$((ok+1)); }
fail() { echo "  [缺]   $1"; bad=$((bad+1)); }
note() { echo "         -> $1"; }

echo "======================================================"
echo " 0. 我在哪"
echo "======================================================"
echo "  user : $(whoami)@$(hostname)"
echo "  cwd  : $(pwd)"
echo "  os   : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -a)"
echo "  磁盘 : $(df -h . | awk 'NR==2 {print $4" 可用 / "$2}')"

echo
echo "======================================================"
echo " 1. GPU"
echo "======================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
             --format=csv,noheader | sed 's/^/  /'
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [[ "${FREE_MB:-0}" -gt 20000 ]]; then
    pass "显存 ${FREE_MB} MiB 空闲：7B bf16 够用"
  elif [[ "${FREE_MB:-0}" -gt 8000 ]]; then
    note "只有 ${FREE_MB} MiB 空闲：可能有人在用这张卡，或需要量化版模型"
  else
    fail "显存几乎被占满（${FREE_MB} MiB）：先 nvidia-smi 看是谁在用"
  fi
else
  fail "没有 nvidia-smi"
  note "CPU 机器也能做：数据准备、BM25、指标；生成需换 API（backend=openai）"
fi

echo
echo "======================================================"
echo " 2. Python 环境"
echo "======================================================"
if command -v python >/dev/null 2>&1; then
  echo "  python : $(python -V 2>&1)  ($(command -v python))"
  if python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    pass "Python >= 3.10"
  else
    fail "Python 太旧（vLLM 新版要求 >= 3.9，建议 3.10/3.11）"
  fi
else
  fail "没有 python"
  note "conda create -n mvs python=3.10 && conda activate mvs"
fi
if command -v conda >/dev/null 2>&1; then
  pass "conda $(conda --version 2>/dev/null | awk '{print $2}')"
else
  note "没装 conda：可用 uv / venv，见 docs/服务器部署指南.md §2"
fi

echo
echo "======================================================"
echo " 3. 关键包"
echo "======================================================"
for pkg in numpy pandas scipy sklearn yaml requests; do
  if python -c "import $pkg" >/dev/null 2>&1; then pass "$pkg"; else fail "$pkg"; fi
done
if python -c "import torch" >/dev/null 2>&1; then
  pass "torch $(python -c 'import torch;print(torch.__version__)') cuda=$(python -c 'import torch;print(torch.cuda.is_available())')"
else
  note "torch 未装（只在要做 NLI / 向量检索时才必须）"
fi
if python -c "import vllm" >/dev/null 2>&1; then
  pass "vllm $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
else
  note "vllm 未装：pip install vllm（要与 torch/CUDA 匹配，见 §3.3）"
fi

echo
echo "======================================================"
echo " 4. 网络与缓存"
echo "======================================================"
echo "  HF_ENDPOINT : ${HF_ENDPOINT:-(未设置，国内建议 export HF_ENDPOINT=https://hf-mirror.com)}"
echo "  HF_HOME     : ${HF_HOME:-(未设置，建议指到数据盘)}"
if python - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("https://hf-mirror.com", timeout=8)
PY
then
  pass "能连 hf-mirror"
else
  note "连不上 hf-mirror：模型要提前用离线包传上去"
fi

echo
echo "======================================================"
echo " 5. 项目"
echo "======================================================"
if [[ -f pyproject.toml && -d src/mvs ]]; then
  pass "在项目根目录"
  python -c "import mvs" 2>/dev/null && pass "mvs 可导入" || note "pip install -e . 后再试"
else
  fail "不在项目根目录（应能看到 pyproject.toml 和 src/mvs）"
fi

echo
echo "======================================================"
echo " 结论：$ok 项通过，$bad 项待处理"
echo "======================================================"
if [[ $bad -eq 0 ]]; then
  echo " 直接开始：python scripts/00_check_env.py"
else
  echo " 按 docs/服务器部署指南.md 的对应小节补齐，再跑一次本脚本。"
fi
