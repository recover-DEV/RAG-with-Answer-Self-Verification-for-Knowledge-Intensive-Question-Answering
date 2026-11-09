"""日志：控制台 + 文件双写。

服务器上跑长任务时，控制台的输出会随 ssh 断开而丢失，
所以文件日志是唯一可靠的现场。路径统一在 logs/ 下。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mvs.paths import LOGS_DIR

__all__ = ["get_logger"]

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE = "%H:%M:%S"
_configured: dict[str, logging.Logger] = {}


def get_logger(name: str = "mvs", log_file: str | Path | None = None,
               level: int | str = logging.INFO) -> logging.Logger:
    """取一个同时写控制台和文件的 logger（同名重复调用不会重复加 handler）。"""
    key = f"{name}|{log_file}"
    if key in _configured:
        return _configured[key]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FMT, datefmt=_DATE)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    target = Path(log_file) if log_file else LOGS_DIR / f"{name}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured[key] = logger
    return logger
