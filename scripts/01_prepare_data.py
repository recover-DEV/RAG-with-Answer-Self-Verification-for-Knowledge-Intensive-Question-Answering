#!/usr/bin/env python
"""第 1 步：原始数据 -> 规整题目 + 三段划分 + 切块。

产物（全部可重建，删了不影响原始数据）：
    data/processed/questions.jsonl   带 split 标记的题目
    data/processed/splits.json       qid 列表 + 数量（提交这份，证明没事后重切）
    data/interim/chunks.jsonl        切块结果，供建索引使用
    data/processed/stats.json        数据画像，写论文"实验设置"时直接用

用法：
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --limit 50      # 只处理 50 题，本地快速验证
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

from mvs.config import load_config, save_config  # noqa: E402
from mvs.data.chunking import iter_page_chunks  # noqa: E402
from mvs.data.crag import make_splits, write_splits  # noqa: E402
from mvs.paths import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402
from mvs.utils.io import save_json, write_jsonl  # noqa: E402
from mvs.utils.logging import get_logger  # noqa: E402

log = get_logger("mvs.prepare")


def find_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(p for p in raw_dir.glob("*.jsonl"))
    files += sorted(p for p in raw_dir.glob("*.json"))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/crag.yaml")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 题（0 = 全部）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = PROJECT_ROOT / cfg.raw_dir
    files = find_raw_files(raw_dir)
    if not files:
        log.error("在 %s 没找到 *.jsonl / *.json", raw_dir)
        log.error("请按 data/README.md 下载 CRAG 数据后放到该目录（raw 目录只放原始文件）")
        return 1

    from mvs.data.crag import load_questions

    questions: dict[str, dict] = {}
    for path in files:
        log.info("读取 %s", path.name)
        for rec in load_questions(path):
            if rec["qid"] in questions:
                log.warning("qid 重复，后一条覆盖前一条: %s", rec["qid"])
            questions[rec["qid"]] = rec

    items = list(questions.values())
    if args.limit:
        items = items[: args.limit]
        log.warning("--limit %d 生效：只处理前 %d 题（统计数字不可用于论文）", args.limit, len(items))

    splits = make_splits(items, {k: v for k, v in vars(cfg.splits).items() if k != "seed"},
                         seed=cfg.splits.seed)
    split_of = {qid: name for name, ids in splits.items() for qid in ids}

    rows = []
    chunk_rows = []
    for rec in items:
        rec = dict(rec)
        rec["split"] = split_of[rec["qid"]]
        rows.append(rec)
        for chunk in iter_page_chunks(
            rec["pages"], doc_prefix=rec["qid"],
            max_chars=cfg.chunk.max_chars, overlap_chars=cfg.chunk.overlap_chars,
            min_chars=cfg.chunk.min_chars,
        ):
            chunk["qid"] = rec["qid"]
            chunk["split"] = rec["split"]
            chunk_rows.append(chunk)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROCESSED_DIR / "questions.jsonl", rows)
    write_splits(splits, PROCESSED_DIR / "splits.json")
    write_jsonl(INTERIM_DIR / "chunks.jsonl", chunk_rows)
    save_config(cfg, PROCESSED_DIR / "data_config.yaml")

    n_pages = sum(len(r["pages"]) for r in rows)
    n_support = sum(1 for r in rows for p in r["pages"] if p.get("support"))
    lengths = np.array([c["n_chars"] for c in chunk_rows]) if chunk_rows else np.zeros(1)
    tokens = lengths.sum() / 1.5  # 中文粗估：1 token ≈ 1.5 字

    stats = {
        "n_questions": len(rows),
        "n_pages": n_pages,
        "pages_per_question": round(n_pages / max(len(rows), 1), 2),
        "support_page_ratio": round(n_support / max(n_pages, 1), 3),
        "n_chunks": len(chunk_rows),
        "chunk_chars_mean": int(lengths.mean()),
        "chunk_chars_p95": int(np.percentile(lengths, 95)),
        "est_tokens_total": int(tokens),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "chunk": {"max_chars": cfg.chunk.max_chars, "overlap_chars": cfg.chunk.overlap_chars},
    }
    save_json(PROCESSED_DIR / "stats.json", stats)

    log.info("题目 %d | 页面 %d | 块 %d | 切分 %s",
             stats["n_questions"], n_pages, stats["n_chunks"], stats["split_counts"])
    log.info("块长度 均值 %d 字 / p95 %d 字 | 总 token 粗估 %d",
             stats["chunk_chars_mean"], stats["chunk_chars_p95"], stats["est_tokens_total"])
    if stats["support_page_ratio"] < 0.05:
        log.warning("support 标记比例很低（%.3f）：可能字段名不对，这会直接影响 oracle recall",
                    stats["support_page_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
