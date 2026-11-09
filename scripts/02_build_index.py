#!/usr/bin/env python
"""第 2 步：切块 -> BM25 索引（纯 CPU，2 万块秒级完成，笔记本即可）。

产物：
    data/index/crag/bm25.pkl        索引本体（可重建，别提交到 git）
    data/index/crag/meta.json       参数与统计
    data/index/crag/qid_to_idx.json 题号 -> 块下标，用于"只在该题自己的页集里检索"

为什么按题目隔离检索：CRAG 的语料是"每题配套的一组 mock 网页"，
如果用一个全局大库跨题检索，相当于让模型偷看别的题的资料，指标会虚高。
这一点必须在论文里写清楚，否则复现的人对不上数。

用法：
    python scripts/02_build_index.py
    python scripts/02_build_index.py --eval-n 50 --eval-k 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

from mvs.config import load_config, save_config  # noqa: E402
from mvs.paths import PROJECT_ROOT  # noqa: E402
from mvs.retrieve.bm25 import BM25  # noqa: E402
from mvs.utils.io import read_jsonl, save_json  # noqa: E402
from mvs.utils.logging import get_logger  # noqa: E402

log = get_logger("mvs.index")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/e0_smoke.yaml")
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out-dir", default="data/index/crag")
    ap.add_argument("--eval-n", type=int, default=0, help="用前 N 道 dev 题算 recall@k")
    ap.add_argument("--eval-k", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    chunks_path = PROJECT_ROOT / args.chunks
    if not chunks_path.is_file():
        log.error("找不到 %s，请先跑 scripts/01_prepare_data.py", chunks_path)
        return 1

    chunks = read_jsonl(chunks_path)
    if not chunks:
        log.error("块文件为空：%s", chunks_path)
        return 1

    doc_ids = [c["doc_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    qids = [c["qid"] for c in chunks]
    supports = np.array([bool(c.get("support")) for c in chunks], dtype=bool)

    started = time.time()
    bm25 = BM25(k1=cfg.retrieval.bm25.k1, b=cfg.retrieval.bm25.b).fit(doc_ids, texts)
    build_s = time.time() - started

    out_dir = PROJECT_ROOT / args.out_dir
    bm25.save(out_dir / "bm25.pkl")

    qid_to_idx: dict[str, list[int]] = {}
    for idx, qid in enumerate(qids):
        qid_to_idx.setdefault(qid, []).append(idx)
    save_json(out_dir / "qid_to_idx.json", qid_to_idx)
    save_config(cfg, out_dir / "index_config.yaml")

    meta = {
        "n_chunks": len(chunks),
        "n_questions": len(qid_to_idx),
        "avgdl": round(bm25.avgdl, 2),
        "k1": bm25.k1,
        "b": bm25.b,
        "build_seconds": round(build_s, 2),
        "n_support_chunks": int(supports.sum()),
        "chunks_file": str(chunks_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    # ---- 可选：oracle recall@k（检索侧上界）----
    if args.eval_n > 0:
        questions = read_jsonl(PROJECT_ROOT / "data/processed/questions.jsonl")
        dev = [q for q in questions if q.get("split") == "dev"][: args.eval_n]
        hits, support_sizes = 0, []
        for q in dev:
            idxs = np.asarray(qid_to_idx.get(q["qid"], []), dtype=np.int64)
            if idxs.size == 0:
                continue
            scores = bm25.scores(q["question"])
            local = scores[idxs]
            order = np.argsort(-local)[: args.eval_k]
            picked = idxs[order]
            sup = idxs[supports[idxs]]
            support_sizes.append(int(sup.size))
            if sup.size and np.isin(picked, sup).any():
                hits += 1
        recall = hits / max(len(dev), 1)
        meta["oracle_recall_at_k"] = round(recall, 4)
        meta["eval_k"] = args.eval_k
        meta["eval_n"] = len(dev)
        meta["support_chunks_per_question"] = round(float(np.mean(support_sizes)), 2) if support_sizes else 0
        log.info("oracle recall@%d = %.3f（%d 题）", args.eval_k, recall, len(dev))
        if recall < 0.7:
            log.warning("召回上界偏低：先确认 support 字段解析正确，再考虑换稠密检索")

    save_json(out_dir / "meta.json", meta)
    log.info("索引完成：%d 块 / %d 题，用时 %.1fs -> %s",
             meta["n_chunks"], meta["n_questions"], build_s, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
