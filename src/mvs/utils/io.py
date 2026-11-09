"""JSONL / CSV / JSON 读写。

服务器上的长任务必须能"断点续跑"：崩溃或者被踢下线后重跑，已完成的题不再重算。
这里用最朴素也最可靠的方式实现 —— 追加写 JSONL，每写一行就 flush。
不搞数据库、不用 parquet 边写边改，因为那些在中断时更容易留下坏文件。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "read_jsonl", "write_jsonl", "append_jsonl", "iter_jsonl",
    "done_keys", "write_csv", "read_csv", "save_json", "load_json",
]


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # 中断时最后一行常常是半截的，跳过比整个任务作废好
                print(f"[warn] {p.name}:{lineno} 不是合法 JSON，已跳过（{exc.msg}）")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    """追加一行并立即 flush —— 断电最多丢最后一行。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def done_keys(path: str | Path, key: str = "qid") -> set[str]:
    """读已有结果文件里已完成的 id 集合，用于跳过。"""
    out: set[str] = set()
    for row in iter_jsonl(path):
        val = row.get(key)
        if val is not None:
            out.add(str(val))
    return out


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> int:
    """写宽表（features.csv 这类一张表支撑全部分析的文件）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return 0
    names = fieldnames or list(rows[0].keys())
    with p.open("w", encoding="utf-8-sig", newline="") as fh:  # utf-8-sig: Excel 打开不乱码
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_csv(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def save_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return p


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)
