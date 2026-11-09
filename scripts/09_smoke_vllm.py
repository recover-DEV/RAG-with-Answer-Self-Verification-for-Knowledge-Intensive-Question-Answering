#!/usr/bin/env python
"""探活：确认 vLLM 服务真的能用，再开始跑长任务。

一次 ssh 只跑这一个脚本，30 秒内得到三个答案：
    1. 端口通不通（/v1/models）
    2. 模型能不能生成（一次 chat）
    3. 中文会不会乱码、延迟大概多少

用法：
    python scripts/09_smoke_vllm.py --base-url http://127.0.0.1:8000/v1
    python scripts/09_smoke_vllm.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

from mvs.generate.llm import LLMClient, LLMConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    client = LLMClient(LLMConfig(
        backend="vllm_http", base_url=args.base_url, model=args.model,
        api_key=args.api_key, temperature=0.0, max_tokens=64, timeout_s=60, max_retries=1,
    ))

    print(f"[1/3] 探活 {args.base_url}/models ...", end=" ", flush=True)
    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        print("失败")
        print(f"      {type(exc).__name__}: {exc}")
        print("      排查：服务起没起？端口对不对？ssh 隧道有没有开？")
        return 1
    print("OK")
    print(f"      可用模型：{models}")
    if args.model not in models and models:
        print(f"      [WARN] 请求的模型 {args.model} 不在列表里，检查 --model 或 vLLM 启动参数")

    print("[2/3] 生成测试（中文 + 数字）...", end=" ", flush=True)
    started = time.time()
    try:
        answer = client.chat([
            {"role": "user", "content": "用一句中文回答：3 加 4 等于几？只输出数字。"},
        ])
    except Exception as exc:  # noqa: BLE001
        print("失败")
        print(f"      {type(exc).__name__}: {exc}")
        return 2
    elapsed = time.time() - started
    print(f"OK（{elapsed:.1f}s）")
    print(f"      输出：{answer.strip()[:200]}")

    print("[3/3] 统计")
    print(f"      {client.summary()}")
    print("\n结论：可以开始跑长任务了。")
    print("      下一步：nohup/tmux 里跑 scripts/03_generate.py，别直接放在 ssh 前台。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
