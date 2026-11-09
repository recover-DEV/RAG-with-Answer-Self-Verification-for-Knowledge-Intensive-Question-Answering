"""评测侧：风险-覆盖曲线、AURC、ECE、Clopper-Pearson 风险上界、bootstrap 置信区间。

这一层是整个毕设的"裁判"。它必须能独立于模型运行（输入就是一张 CSV），
所以论文里的每一张图、每一个数，都应该能从这里对一个 features.csv 重算出来。
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "risk_coverage", "aurc", "ece", "brier", "cp_upper_bound",
    "select_threshold", "bootstrap_ci", "paired_bootstrap_pvalue",
]


def _as_arrays(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(conf, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=bool).ravel()
    if c.shape != y.shape:
        raise ValueError(f"conf 与 correct 长度不一致: {c.shape} vs {y.shape}")
    return c, y


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """按置信度从高到低逐个接纳，返回 (coverage, risk)。

    risk = 已接纳样本里的错误率。曲线上每个点对应一个阈值 τ。
    """
    c, y = _as_arrays(conf, correct)
    if c.size == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(-c, kind="mergesort")
    hits = np.cumsum(y[order])
    n = np.arange(1, c.size + 1, dtype=np.float64)
    return n / c.size, 1.0 - hits / n


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area Under Risk-Coverage curve：越小越好。随机排序时约等于总体错误率。"""
    cov, risk = risk_coverage(conf, correct)
    if cov.size == 0:
        return float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 / numpy<2
    return float(trapezoid(risk, cov))


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected Calibration Error：分箱后 |平均置信度 − 实际正确率| 的加权和。"""
    c, y = _as_arrays(conf, correct)
    if c.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if not mask.any():
            continue
        total += mask.sum() / c.size * abs(c[mask].mean() - y[mask].mean())
    return float(total)


def brier(conf: Sequence[float], correct: Sequence[bool]) -> float:
    c, y = _as_arrays(conf, correct)
    if c.size == 0:
        return float("nan")
    return float(np.mean((c - y.astype(np.float64)) ** 2))


def cp_upper_bound(errors: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson 单侧上界：在 1−alpha 置信度下，真实错误率不超过这个值。

    这是"风险受控"三个字在统计上的落地方式 —— 不是"我看着风险挺低"，
    而是"错得再多也就这么多，有 95% 把握"。
    """
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    try:
        from scipy.stats import beta

        return float(beta.ppf(1.0 - alpha, errors + 1, n - errors))
    except ImportError:  # 没有 scipy 时退化为正态近似
        p = errors / n
        se = np.sqrt(max(p * (1 - p), 1e-9) / n)
        return float(min(1.0, p + 1.64 * se + 1.0 / n))


def select_threshold(conf: Sequence[float], correct: Sequence[bool], alpha: float = 0.1,
                     min_coverage: float = 0.05, alpha_ci: float = 0.05) -> dict:
    """在 conf 上选阈值 τ：在 CP 上界 ≤ alpha 的前提下，覆盖尽可能高。

    注意：这个函数只应该在 cal 划分上调用。在 test 上选阈值就是阈值泄漏，
    是这类论文最容易被质疑的地方（也是你这套方法唯一真正容易做错的地方）。
    """
    c, y = _as_arrays(conf, correct)
    if c.size == 0:
        raise ValueError("conf 为空")
    order = np.argsort(-c, kind="mergesort")
    c_sorted, y_sorted = c[order], y[order]
    errors = np.cumsum(~y_sorted)
    n = np.arange(1, c.size + 1)

    best = {"tau": float("inf"), "coverage": 0.0, "risk": 0.0,
            "n_accepted": 0, "cp_upper": 0.0, "alpha": alpha}
    for i in range(c.size - 1, -1, -1):
        cov = n[i] / c.size
        if cov < min_coverage:
            break
        upper = cp_upper_bound(int(errors[i]), int(n[i]), alpha_ci)
        if upper <= alpha:
            best = {
                "tau": float(c_sorted[i]),
                "coverage": float(cov),
                "risk": float(errors[i] / n[i]),
                "n_accepted": int(n[i]),
                "cp_upper": float(upper),
                "alpha": alpha,
            }
            break
    return best


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """bootstrap 置信区间，返回 (点估计, 下界, 上界)。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[i] = stat(sample)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(stat(arr)), float(lo), float(hi)


def paired_bootstrap_pvalue(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                           seed: int = 0) -> float:
    """配对 bootstrap 检验两个方法是否有显著差异（单侧取 min 再 ×2）。"""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"配对样本长度必须相同: {x.shape} vs {y.shape}")
    if x.size == 0:
        return float("nan")
    diff = x - y
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    p_low = float(np.mean(boot <= 0.0))
    p_high = float(np.mean(boot >= 0.0))
    return float(min(1.0, 2.0 * min(p_low, p_high)))
