"""数据读取、字段规整、三段划分。

设计原则：官方字段名一定会有出入（json / jsonl / parquet，answers 还是 answer），
所以这里只做"容错规整"，把一切差异收敛到 data/README.md 里写的那个 schema。
一旦规整完，后面的模块就再也不用关心原始格式。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from mvs.utils.io import iter_jsonl, save_json

__all__ = ["load_questions", "normalize_question", "assign_split", "make_splits", "write_splits"]


def load_questions(path: str | Path) -> list[dict]:
    """读题目文件（.jsonl 或 .json 数组），返回规整后的列表。"""
    p = Path(path)
    if p.suffix == ".jsonl":
        raw = iter_jsonl(p)
    elif p.suffix == ".json":
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data if isinstance(data, list) else data.get("questions", [])
    else:
        raise ValueError(f"不支持的文件类型: {p.suffix}（请先转成 jsonl 或 json）")

    out = []
    for i, rec in enumerate(raw):
        try:
            out.append(normalize_question(rec, fallback_idx=i))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[warn] 跳过第 {i} 条（{exc}）")
    return out


def normalize_question(rec: dict, fallback_idx: int = 0) -> dict:
    """把一条原始记录规整成统一字段。字段缺失时给出显式默认值，不静默吞掉。"""
    if not isinstance(rec, dict):
        raise TypeError(f"记录不是 dict: {type(rec)}")

    qid = (
        rec.get("interaction_id")
        or rec.get("qid")
        or rec.get("id")
        or rec.get("question_id")
    )
    if qid is None:
        qid = f"auto-{fallback_idx:05d}"
    question = rec.get("question") or rec.get("query") or rec.get("question_text")
    if not question:
        raise ValueError("缺少 question 字段")

    answer = rec.get("answer") or rec.get("answers") or rec.get("gold_answer") or ""
    if isinstance(answer, (list, tuple)):
        aliases = [str(a) for a in answer]
        answer = aliases[0] if aliases else ""
    else:
        aliases = [str(a) for a in (rec.get("alt_ans") or rec.get("answer_aliases") or [])]

    pages = rec.get("pages") or rec.get("search_results") or rec.get("contexts") or []
    if not isinstance(pages, list):
        pages = [pages]

    return {
        "qid": str(qid),
        "question": str(question),
        "answer": str(answer),
        "answer_aliases": aliases,
        "domain": str(rec.get("domain") or rec.get("category") or "unknown"),
        "question_type": str(rec.get("question_type") or rec.get("type") or "unknown"),
        "pages": [_normalize_page(pg, i) for i, pg in enumerate(pages)],
        "official_split": rec.get("split"),
        "query_time": rec.get("query_time"),
    }



def _normalize_page(page, idx: int) -> dict:
    if isinstance(page, str):
        return {"title": f"page-{idx}", "url": "", "text": page, "support": None}
    if not isinstance(page, dict):
        return {"title": f"page-{idx}", "url": "", "text": str(page), "support": None}
    text = page.get("text") or page.get("content") or page.get("page_snippet") or ""
    support = page.get("support")
    if isinstance(support, str):
        support = support.strip().lower() in {"true", "yes", "1", "support"}
    return {
        "title": str(page.get("title") or page.get("page_name") or f"page-{idx}"),
        "url": str(page.get("url") or page.get("page_url") or ""),
        "text": str(text),
        "support": support,
    }


def _hash01(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def assign_split(qid: str, ratios: dict[str, float], seed: int = 2027) -> str:
    """按 (seed, qid) 哈希落到 dev/cal/test。

    必须是哈希而不是 random.shuffle —— 后者只要题目顺序变了，切分就全变，
    而"事后重切"是审阅时最容易被抓的作弊点之一。
    """
    names = list(ratios.keys())
    total = sum(ratios.values())
    point = _hash01(f"{seed}:{qid}")
    acc = 0.0
    for name in names:
        acc += ratios[name] / total
        if point <= acc:
            return name
    return names[-1]


def make_splits(questions: Iterable[dict], ratios: dict[str, float],
                seed: int = 2027) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {name: [] for name in ratios}
    for rec in questions:
        out[assign_split(rec["qid"], ratios, seed)].append(rec["qid"])
    return out


def write_splits(splits: dict[str, list[str]], out_path: str | Path) -> Path:
    payload = {name: sorted(ids) for name, ids in splits.items()}
    payload["_counts"] = {name: len(ids) for name, ids in splits.items()}
    return save_json(out_path, payload)
