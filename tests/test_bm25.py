"""BM25 的单元测试：检索排序对不对、存档能不能原样读回来。"""

from __future__ import annotations

import numpy as np
import pytest

from mvs.retrieve.bm25 import BM25


DOCS = [
    "苹果公司的创始人是谁 乔布斯 沃兹尼亚克",
    "苹果 手机 iPhone 发布会 时间",
    "香蕉 的 营养 成分 和 热量",
    "量子 计算 的 基本 原理 是 叠加 与 纠缠",
]


@pytest.fixture()
def engine() -> BM25:
    return BM25(k1=1.5, b=0.75).fit(
        [f"d{i}" for i in range(len(DOCS))], DOCS,
    )


def test_most_relevant_first(engine: BM25):
    hits = engine.search("苹果 手机 发布会", k=2)
    assert hits, "没有召回任何结果"
    assert hits[0][0] == "d1"


def test_unrelated_query_returns_nothing(engine: BM25):
    assert engine.search("完全无关的词汇组合", k=3) == []


def test_empty_corpus_is_safe():
    engine = BM25().fit([], [])
    assert engine.search("任意查询", k=3) == []
    assert len(engine) == 0


def test_save_load_roundtrip(tmp_path, engine: BM25):
    path = engine.save(tmp_path / "bm25.pkl")
    loaded = BM25.load(path)
    assert loaded.doc_ids == engine.doc_ids
    assert np.allclose(loaded.doc_len, engine.doc_len)
    assert loaded.avgdl == pytest.approx(engine.avgdl)
    q = "量子 计算 原理"
    assert loaded.search(q, k=3) == engine.search(q, k=3)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        BM25().fit(["a", "b"], ["only one text"])


def test_scores_are_non_negative(engine: BM25):
    scores = engine.scores("苹果 香蕉 量子")
    assert np.all(scores >= 0)
