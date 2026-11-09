"""指标模块的单元测试：用能心算出答案的小例子，防止"指标算错但曲线很好看"。

跑法：
    pytest -q
    python -m pytest tests/test_metrics.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from mvs.eval.metrics import (
    aurc, brier, cp_upper_bound, ece, paired_bootstrap_pvalue,
    risk_coverage, select_threshold,
)


def test_risk_coverage_perfect_order():
    # 置信度高的全对、低的错：任何覆盖率下风险都是 0
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 1, 1]
    cov, risk = risk_coverage(conf, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, 0.0)


def test_risk_coverage_first_accepted_is_wrong():
    conf = [0.9, 0.8]
    correct = [0, 1]
    cov, risk = risk_coverage(conf, correct)
    assert risk[0] == pytest.approx(1.0)   # 只接纳 1 条，且错了
    assert risk[1] == pytest.approx(0.5)   # 全接纳，错 1/2


def test_aurc_bounds():
    conf = np.linspace(0.99, 0.01, 100)
    perfect = aurc(conf, np.ones(100, dtype=bool))
    worst = aurc(conf, np.zeros(100, dtype=bool))
    assert perfect == pytest.approx(0.0, abs=1e-9)
    assert worst == pytest.approx(1.0, abs=1e-9)


def test_ece_of_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=20000)
    correct = rng.random(20000) < conf
    assert ece(conf, correct) < 0.02


def test_brier_endpoints():
    conf = np.array([1.0, 0.0])
    correct = np.array([1, 0])
    assert brier(conf, correct) == pytest.approx(0.0)


def test_cp_upper_bound_shrinks_with_more_samples():
    # 同样的错误率 10%，样本越多上界越紧
    small = cp_upper_bound(10, 100, alpha=0.05)
    large = cp_upper_bound(100, 1000, alpha=0.05)
    assert large < small
    assert 0.10 < large < 0.13
    assert 0.10 < small < 0.20


def test_cp_upper_bound_all_correct():
    assert cp_upper_bound(0, 100, alpha=0.05) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-3)


def test_select_threshold_respects_alpha():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, size=3000)
    correct = rng.random(3000) < conf          # 校准良好的置信度
    best = select_threshold(conf, correct, alpha=0.15)
    assert best["n_accepted"] > 0
    assert best["cp_upper"] <= 0.15
    assert best["risk"] <= 0.15


def test_select_threshold_impossible_alpha_returns_empty():
    conf = np.linspace(1, 0, 50)
    correct = np.zeros(50, dtype=bool)          # 全错
    best = select_threshold(conf, correct, alpha=0.05)
    assert best["n_accepted"] == 0


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.1, size=200)
    b = rng.normal(0.0, 0.1, size=200)
    assert paired_bootstrap_pvalue(a, b) < 0.01


def test_paired_bootstrap_same_distribution_not_significant():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)
    assert paired_bootstrap_pvalue(a, b) > 0.05
