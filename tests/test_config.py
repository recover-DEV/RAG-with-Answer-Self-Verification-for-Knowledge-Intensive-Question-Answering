"""配置与数据切分的单元测试。"""

from __future__ import annotations

import textwrap

import yaml

from mvs.config import load_config, load_raw
from mvs.data.crag import assign_split, make_splits, normalize_question


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_extends_merge_and_override(tmp_path):
    _write(tmp_path, "base.yaml", """
        project: mvs
        seed: 2027
        generator:
          backend: mock
          max_tokens: 128
    """)
    child = _write(tmp_path, "exp/e0.yaml", """
        extends: ../base.yaml
        generator:
          max_tokens: 64
    """)
    cfg = load_config(child)
    assert cfg.project == "mvs"
    assert cfg.generator.backend == "mock"
    assert cfg.generator.max_tokens == 64      # 子配置覆盖父配置


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_URL", "http://example.test/v1")
    path = _write(tmp_path, "c.yaml", """
        generator:
          base_url: ${MY_URL}
          model: ${NOT_SET:-fallback-model}
    """)
    cfg = load_config(path)
    assert cfg.generator.base_url == "http://example.test/v1"
    assert cfg.generator.model == "fallback-model"


def test_cli_overrides(tmp_path):
    path = _write(tmp_path, "c.yaml", "run:\n  n: 1\n")
    cfg = load_config(path, ["run.n=5"])
    assert cfg.run.n == 5


def test_extends_cycle_detected(tmp_path):
    a = _write(tmp_path, "a.yaml", "extends: b.yaml\nx: 1\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\ny: 2\n")
    try:
        load_raw(a)
    except ValueError as exc:
        assert "循环" in str(exc)
    else:
        raise AssertionError("应当检测到循环引用")


def test_split_is_deterministic_and_partitions():
    qids = [f"q{i:04d}" for i in range(200)]
    ratios = {"dev": 0.2, "cal": 0.3, "test": 0.5}
    first = make_splits([{"qid": q} for q in qids], ratios, seed=2027)
    second = make_splits([{"qid": q} for q in qids], ratios, seed=2027)
    assert first == second                                   # 可复现
    flat = [q for ids in first.values() for q in ids]
    assert sorted(flat) == sorted(qids)                       # 不重不漏
    assert abs(len(first["test"]) / len(qids) - 0.5) < 0.12    # 比例大致对


def test_split_changes_with_seed():
    qid = "q0001"
    ratios = {"dev": 0.5, "test": 0.5}
    assert assign_split(qid, ratios, seed=1) in ratios


def test_normalize_question_tolerates_aliases():
    rec = {
        "id": "x1",
        "query": "问题文本",
        "answers": ["答案一", "答案二"],
        "search_results": [{"page_name": "标题", "page_snippet": "正文", "page_url": "u"}],
    }
    out = normalize_question(rec)
    assert out["qid"] == "x1"
    assert out["question"] == "问题文本"
    assert out["answer"] == "答案一"
    assert out["answer_aliases"] == ["答案一", "答案二"]
    assert out["pages"][0]["text"] == "正文"


def test_yaml_safe_dump_roundtrip_of_config(tmp_path):
    path = _write(tmp_path, "c.yaml", "a:\n  b: [1, 2]\n")
    data = load_raw(path)
    assert yaml.safe_load(yaml.safe_dump(data)) == data
