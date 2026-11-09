#!/usr/bin/env python
"""环境体检：本地和服务器上都先跑这个，一眼看出缺什么。

用法：
    python scripts/00_check_env.py
    python scripts/00_check_env.py --config configs/exp/e0_smoke.yaml
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

CORE = ["numpy", "pandas", "sklearn", "scipy", "yaml", "requests", "tqdm", "jieba"]
OPTIONAL = ["bm25s", "torch", "transformers", "sentence_transformers", "faiss", "vllm"]


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001 - 缺包的原因五花八门，统一显示
        return f"缺失（{type(exc).__name__}）"
    return getattr(mod, "__version__", "已安装（无 __version__）")


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"不可用：{exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/e0_smoke.yaml")
    args = ap.parse_args()

    print("=" * 68)
    print("1. 解释器")
    print("=" * 68)
    print(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform : {platform.platform()}")
    print(f"cwd      : {Path.cwd()}")

    print("\n" + "=" * 68)
    print("2. 依赖")
    print("=" * 68)
    missing_core = []
    for name in CORE:
        ver = _version(name)
        print(f"  [核心] {name:<22} {ver}")
        if ver.startswith("缺失"):
            missing_core.append(name)
    for name in OPTIONAL:
        print(f"  [可选] {name:<22} {_version(name)}")

    print("\n" + "=" * 68)
    print("3. GPU / 磁盘")
    print("=" * 68)
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    print("nvidia-smi :", smi if smi else "无 GPU 或未装驱动（本地正常）")
    try:
        import torch

        print("torch.cuda :", torch.cuda.is_available(),
              "| 设备数:", torch.cuda.device_count(),
              "| CUDA:", getattr(torch.version, "cuda", None))
    except ImportError:
        print("torch.cuda : torch 未安装（本地分析任务不需要）")

    root = Path(os.environ.get("MVS_ROOT", Path(__file__).resolve().parents[1])).resolve()
    total, used, free = shutil.disk_usage(root)
    print(f"磁盘 {root.drive or root} : 剩余 {free / 2**30:.1f} GiB / 共 {total / 2**30:.1f} GiB")
    if free < 20 * 2**30:
        print("  [WARN] 剩余空间不足 20 GiB：模型权重 + 索引可能放不下")

    print("\n" + "=" * 68)
    print("4. 项目配置")
    print("=" * 68)
    try:
        from mvs.config import load_config
        from mvs.paths import PROJECT_ROOT, RUNS_DIR

        cfg = load_config(args.config)
        print(f"项目根     : {PROJECT_ROOT}")
        print(f"run 目录   : {RUNS_DIR}")
        print(f"LLM 后端   : {cfg.generator.backend} / {cfg.generator.model}")
        print(f"数据划分   : {cfg.splits}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置加载失败：{type(exc).__name__}: {exc}")
        return 2

    data_ok = (PROJECT_ROOT / "data" / "raw").exists()
    print(f"data/raw   : {'存在' if data_ok else '不存在（先按 data/README.md 下载）'}")

    print("\n" + "=" * 68)
    if missing_core:
        print(f"结论：缺核心依赖 {missing_core}")
        print("      pip install -r requirements.txt")
        return 1
    print("结论：核心环境 OK。")
    if not smi:
        print("      无 GPU —— 仍可完成：数据准备、BM25 建索引、prompt 联调、指标与画图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
