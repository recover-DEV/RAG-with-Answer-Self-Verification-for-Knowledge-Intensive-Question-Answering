"""检索侧：BM25（纯 numpy 倒排实现，零 GPU、零网络依赖）。

为什么不直接用 rank_bm25：百万级语料它要吃 10—20 GB 内存，服务器上会直接把内存打满。
为什么不直接用 bm25s：它的 API 在不同版本间变动过，而且本仓库要保持"随便一台机器
clone 下来就能跑通冒烟实验"。所以这里自己写一个 60 行的倒排版本——
够快（2 万块毫秒级）、可控、可序列化。要到百万级再换 bm25s，接口保持一致。
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = ["BM25", "default_tokenizer"]

_ASCII_RE = re.compile(r"[a-z0-9]+")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]+")


def default_tokenizer(text: str) -> list[str]:
    """有 jieba 用 jieba，没有就用"英文词 + 中文二元组"兜底。

    二元组不是最优，但对"能不能检索到"这件事足够，且完全无依赖、可复现。
    """
    try:
        import jieba

        return [w.strip().lower() for w in jieba.lcut(text or "") if w.strip()]
    except ImportError:
        low = (text or "").lower()
        terms = _ASCII_RE.findall(low)
        for seg in _ZH_RE.findall(low):
            terms.extend(seg[i:i + 2] for i in range(max(len(seg) - 1, 1)))
        return terms


class BM25:
    """Okapi BM25：score = Σ idf(t) · tf·(k1+1) / (tf + k1·(1-b+b·dl/avgdl))"""

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or default_tokenizer
        self.doc_ids: list[str] = []
        self.doc_len = np.zeros(0, dtype=np.float64)
        self.avgdl = 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    # ---------- 构建 ----------

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25":
        if len(doc_ids) != len(texts):
            raise ValueError(f"doc_ids 与 texts 长度不一致: {len(doc_ids)} vs {len(texts)}")
        self.doc_ids = [str(d) for d in doc_ids]

        post_docs: dict[str, list[int]] = {}
        post_tf: dict[str, list[int]] = {}
        lengths = np.zeros(len(texts), dtype=np.float64)

        for idx, text in enumerate(texts):
            toks = self.tokenizer(text or "")
            lengths[idx] = len(toks)
            for term, count in Counter(toks).items():
                post_docs.setdefault(term, []).append(idx)
                post_tf.setdefault(term, []).append(count)

        n_docs = len(texts)
        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if n_docs and lengths.mean() > 0 else 1.0

        for term, docs in post_docs.items():
            df = len(docs)
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.postings[term] = (
                np.asarray(docs, dtype=np.int64),
                np.asarray(post_tf[term], dtype=np.float64),
            )
        return self

    # ---------- 查询 ----------

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_ids), dtype=np.float64)
        for term, q_count in Counter(self.tokenizer(query or "")).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, tfs = posting
            dl = self.doc_len[docs]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            out[docs] += q_count * self.idf[term] * tfs * (self.k1 + 1.0) / denom
        return out

    def search(self, query: str, k: int = 20, positive_only: bool = True) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        all_scores = self.scores(query)
        if positive_only:
            candidates = np.flatnonzero(all_scores > 0)
        else:
            candidates = np.arange(all_scores.shape[0])
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(-all_scores[candidates])][:k]
        return [(self.doc_ids[i], float(all_scores[i])) for i in order]

    # ---------- 持久化 ----------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1, "b": self.b, "doc_ids": self.doc_ids,
            "doc_len": self.doc_len, "avgdl": self.avgdl,
            "postings": self.postings, "idf": self.idf,
        }
        with p.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @classmethod
    def load(cls, path: str | Path, tokenizer: Callable[[str], list[str]] | None = None) -> "BM25":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        obj = cls(k1=payload["k1"], b=payload["b"], tokenizer=tokenizer)
        obj.doc_ids = payload["doc_ids"]
        obj.doc_len = payload["doc_len"]
        obj.avgdl = payload["avgdl"]
        obj.postings = payload["postings"]
        obj.idf = payload["idf"]
        return obj

    def __len__(self) -> int:
        return len(self.doc_ids)
