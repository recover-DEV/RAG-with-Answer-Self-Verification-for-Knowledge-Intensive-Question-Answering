"""路径常量与 run 目录管理。

所有脚本都不要手写相对路径，一律从这里取 —— 这样本地（E:\\...\\实验）
和服务器（/home/xxx/mvs）跑同一份代码，不需要改任何代码或配置。
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR", "INDEX_DIR",
    "RESULTS_DIR", "RUNS_DIR", "FEATURES_DIR", "TABLES_DIR", "FIGURES_DIR",
    "LOGS_DIR", "CONFIGS_DIR", "ensure_dirs", "run_dir", "new_run_id",
]

# src/mvs/paths.py -> parents[2] 就是仓库根（实验/）
PROJECT_ROOT = Path(os.environ.get("MVS_ROOT", Path(__file__).resolve().parents[2])).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FEATURES_DIR = RESULTS_DIR / "features"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

_ALL_DIRS = (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INDEX_DIR,
    RUNS_DIR, FEATURES_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR,
)

_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def ensure_dirs() -> None:
    """建齐所有约定目录（幂等，随便调用）。"""
    for path in _ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def new_run_id(prefix: str, tag: str = "", when: dt.datetime | None = None) -> str:
    """生成可读的 run 标识，例如 E03_bm25top5_20261116-1430。

    命名规则：<前缀>_<可选标签>_<YYYYMMDD-HHMM>
    这样按名字排序就是按时间排序，找"那天的实验"不用翻台账。
    """
    stamp = (when or dt.datetime.now()).strftime("%Y%m%d-%H%M")
    parts = [prefix.strip()]
    if tag:
        parts.append(_ID_SAFE.sub("", tag.strip().replace(" ", "")))
    parts.append(stamp)
    return "_".join(p for p in parts if p)


def run_dir(run_id: str, create: bool = True) -> Path:
    """一次实验一个目录：results/runs/<run_id>/"""
    path = RUNS_DIR / _ID_SAFE.sub("_", run_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
