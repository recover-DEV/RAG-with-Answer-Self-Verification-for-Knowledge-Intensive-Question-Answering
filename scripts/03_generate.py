#!/usr/bin/env python
"""第 3 步：检索 + 拼 prompt + 生成，结果逐条落盘（可断点续跑）。

本地没有卡时把后端设成 mock：验证 prompt 长度、JSON 解析、结果文件结构 ——
这三样占总出错原因的八成。服务器上换成 vllm_http，命令完全一样。

用法：
    python scripts/03_generate.py --config configs/exp/e0_smoke.yaml --limit 5
    python scripts/03_generate.py --config configs/exp/e0_smoke.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

from mvs.config import load_config, save_config  # noqa: E402
from mvs.generate.llm import LLMClient, LLMConfig  # noqa: E402
from mvs.generate.prompts import build_qa_messages  # noqa: E402
from mvs.paths import PROJECT_ROOT, new_run_id, run_dir  # noqa: E402
from mvs.retrieve.bm25 import BM25  # noqa: E402
from mvs.utils.io import append_jsonl, done_keys, load_json, read_jsonl, save_json  # noqa: E402
from mvs.utils.logging import get_logger  # noqa: E402
from mvs.utils.seed import set_seed  # noqa: E402

log = get_logger("mvs.generate")

QA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "integer"}},
                "sufficient": {"type": "boolean"},
                "missing": {"type": "string"},
            },
            "required": ["answer", "citations", "sufficient", "missing"],
            "additionalProperties": False,
        },
    },
}


def parse_json_block(text: str) -> dict | None:
    """解析首个 JSON 对象，允许前后带解释或代码围栏。"""
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, idx=start)
    except json.JSONDecodeError:
        return None
    return parsed


def parse_qa_output(text: str) -> tuple[dict | None, bool]:
    """只有完整字段仅缺外层右括号时才修复，保留原始输出。"""
    parsed = parse_json_block(text)
    if parsed is not None:
        return parsed, False
    start = text.find("{")
    if start < 0:
        return None, False
    try:
        repaired = json.loads(text[start:].rstrip() + "}")
    except json.JSONDecodeError:
        return None, False
    if (
        repaired.keys() == {"answer", "citations", "sufficient", "missing"}
        and isinstance(repaired["answer"], str)
        and isinstance(repaired["missing"], str)
        and isinstance(repaired["sufficient"], bool)
        and isinstance(repaired["citations"], list)
        and all(type(citation) is int for citation in repaired["citations"])
    ):
        return repaired, True
    return None, False


def retrieve_pages(bm25: BM25, qid: str, question: str, qid_to_idx: dict,
                   top_k: int, text_of: dict[str, str]) -> tuple[list[dict], list[str], list[float]]:
    """在该题自己的页集里检索 top_k 块，返回 (上下文页, 命中块 id, 分数)。"""
    idxs = np.asarray(qid_to_idx.get(qid, []), dtype=np.int64)
    if idxs.size == 0:
        return [], [], []
    scores = bm25.scores(question)
    order = np.argsort(-scores[idxs])[:top_k]
    picked = idxs[order]
    ids = [bm25.doc_ids[i] for i in picked]
    pages = [{"title": doc_id, "text": text_of.get(doc_id, "")} for doc_id in ids]
    return pages, ids, [round(float(scores[i]), 4) for i in picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/e0_smoke.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "cal", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    questions = read_jsonl(PROJECT_ROOT / "data/processed/questions.jsonl")
    if not questions:
        log.error("没有规整后的题目，请先跑 scripts/01_prepare_data.py")
        return 1
    if args.split != "all":
        questions = [q for q in questions if q.get("split") == args.split]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        log.error("split=%s 下没有题目", args.split)
        return 1

    index_dir = PROJECT_ROOT / "data/index/crag"
    if not (index_dir / "bm25.pkl").is_file():
        log.error("没有索引，请先跑 scripts/02_build_index.py")
        return 1
    bm25 = BM25.load(index_dir / "bm25.pkl")
    qid_to_idx = load_json(index_dir / "qid_to_idx.json", default={})
    text_of = {c["doc_id"]: c["text"] for c in read_jsonl(PROJECT_ROOT / "data/interim/chunks.jsonl")}

    tag = cfg.generator.model.split("/")[-1]
    response_format = QA_RESPONSE_FORMAT if cfg.generator.backend == "vllm_http" else None
    cfg.generator.response_format = response_format
    run_id = args.run_id or new_run_id("E00" if cfg.generator.backend == "mock" else "E01", tag)
    out_path = run_dir(run_id) / "gen.jsonl"
    save_config(cfg, out_path.parent / "config.yaml")

    finished = done_keys(out_path, key="qid")
    todo = [q for q in questions if q["qid"] not in finished]
    log.info("run=%s | 待跑 %d 题（已完成 %d）", run_id, len(todo), len(finished))
    if not todo:
        log.info("全部已完成：%s", out_path)
        return 0

    client = LLMClient(LLMConfig(
        backend=cfg.generator.backend, base_url=cfg.generator.base_url,
        model=cfg.generator.model, api_key=cfg.generator.api_key,
        temperature=cfg.generator.temperature, top_p=cfg.generator.top_p,
        max_tokens=cfg.generator.max_tokens, timeout_s=cfg.generator.timeout_s,
        max_retries=cfg.generator.max_retries, concurrency=cfg.generator.concurrency,
        price_per_1m_input=cfg.generator.price_per_1m_input,
        price_per_1m_output=cfg.generator.price_per_1m_output,
    ))
    if cfg.generator.backend != "mock":
        log.info("探活成功，可用模型：%s", client.list_models())

    started = time.time()
    n_done = 0
    batch_size = max(1, cfg.generator.concurrency)

    for offset in range(0, len(todo), batch_size):
        batch = todo[offset: offset + batch_size]
        payloads, metas = [], []
        for q in batch:
            pages, ids, scores = retrieve_pages(
                bm25, q["qid"], q["question"], qid_to_idx,
                cfg.retrieval.top_k, text_of,
            )
            if not ids:
                log.warning("%s 没有可检索的块，跳过", q["qid"])
                continue
            payloads.append(build_qa_messages(q["question"], pages[: cfg.retrieval.rerank_top_n]))
            metas.append({"q": q, "ids": ids[: cfg.retrieval.rerank_top_n],
                          "scores": scores[: cfg.retrieval.rerank_top_n]})

        if not payloads:
            continue
        replies = client.chat_many(
            payloads, concurrency=cfg.generator.concurrency, response_format=response_format,
        )

        for meta, reply in zip(metas, replies):
            q = meta["q"]
            parsed, repaired = parse_qa_output(reply or "")
            append_jsonl(out_path, {
                "qid": q["qid"],
                "question": q["question"],
                "gold_answer": q.get("answer", ""),
                "split": q.get("split"),
                "retrieved": meta["ids"],
                "retrieval_scores": meta["scores"],
                "raw_output": reply,
                "answer": (parsed or {}).get("answer", ""),
                "citations": (parsed or {}).get("citations", []),
                "model_sufficient": (parsed or {}).get("sufficient"),
                "model_missing": (parsed or {}).get("missing", ""),
                "parse_ok": parsed is not None,
                "json_repaired": repaired,
            })
            n_done += 1
        log.info("进度 %d/%d", min(offset + batch_size, len(todo)), len(todo))

    elapsed = time.time() - started
    rows = read_jsonl(out_path)
    parse_fail = sum(1 for r in rows if not r.get("parse_ok"))
    save_json(out_path.parent / "stats.json", {
        "run_id": run_id, "split": args.split, "n_new": n_done,
        "n_total": len(rows), "parse_fail": parse_fail,
        "json_repaired": sum(bool(r.get("json_repaired")) for r in rows),
        "seconds": round(elapsed, 1),
        "sec_per_question": round(elapsed / max(n_done, 1), 3),
        **client.summary(),
    })
    log.info("完成 %d 条（本次 %d）| 解析失败 %d | %.1fs -> %s",
             len(rows), n_done, parse_fail, elapsed, out_path.parent)
    if parse_fail > len(rows) * 0.05:
        log.warning("解析失败率 > 5%%：检查 prompt 的 JSON 约定，或确认 temperature=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
