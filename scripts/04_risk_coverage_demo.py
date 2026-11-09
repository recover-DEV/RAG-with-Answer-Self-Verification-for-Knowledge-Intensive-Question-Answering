#!/usr/bin/env python
"""第 4 步（不需要 GPU、不需要数据）：用合成数据跑通指标与画图。

这一步的意义：论文里最关键的那张"风险-覆盖曲线"，
它的代码现在就能写完、跑通、看到形状。等到有真数据时，
只是把 conf/correct 两列从 features.csv 读进来 —— 不用再改画图逻辑。

用法：
    python scripts/04_risk_coverage_demo.py
产物：
    results/figures/fig01_risk_coverage_demo.png
    results/tables/demo_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvs.utils.console import init_console  # noqa: E402

init_console()

from mvs.eval.metrics import aurc, ece, risk_coverage, select_threshold  # noqa: E402
from mvs.paths import FIGURES_DIR, TABLES_DIR  # noqa: E402
from mvs.utils.io import write_csv  # noqa: E402
from mvs.utils.logging import get_logger  # noqa: E402

log = get_logger("mvs.demo")


def simulate(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """造三组样本：随机基线 / 未校准 / 已校准。

    真值生成方式：每题有个隐藏难度 d，难度越高越容易答错；
    置信度 = 难度的单调函数 + 噪声。校准版只是把置信度重映射到 [0,1]，
    所以"排序不变、曲线更准"—— 这正是校准该起到的作用。
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.beta(2.0, 3.0, size=n)
    correct = rng.random(n) > difficulty

    raw_conf = np.clip(1.0 - difficulty + rng.normal(0, 0.12, size=n), 0.0, 1.0)
    # 未校准：过度自信，把 0.5 附近的分数往两端拉
    overconf = np.clip(raw_conf ** 0.6, 0.0, 1.0)
    # 已校准：按分位数重排成均匀分布
    ranks = np.argsort(np.argsort(raw_conf)) / (n - 1)
    calibrated = 0.05 + 0.9 * ranks
    baseline = rng.random(n)
    return np.stack([baseline, overconf, calibrated]), correct, baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10, help="允许的最大错误率")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    confs, correct, _ = simulate(args.n)
    names = ["random", "overconfident", "calibrated"]
    rows = []
    curves = {}
    for name, conf in zip(names, confs):
        cov, risk = risk_coverage(conf, correct)
        curves[name] = (cov, risk)
        best = select_threshold(conf, correct, alpha=args.alpha)
        rows.append({
            "method": name,
            "aurc": round(aurc(conf, correct), 4),
            "ece": round(ece(conf, correct), 4),
            "coverage_at_alpha": round(best["coverage"], 4),
            "risk_at_alpha": round(best["risk"], 4),
            "cp_upper": round(best["cp_upper"], 4),
            "tau": round(best["tau"], 4),
            "n_accepted": best["n_accepted"],
        })

    table_path = TABLES_DIR / "demo_metrics.csv"
    n_rows = write_csv(table_path, rows)
    print(f"\n写入 {table_path}（{n_rows} 行）")
    header = f"{'method':<16}{'AURC':>8}{'ECE':>8}{'cov@a':>9}{'risk@a':>9}{'CP上界':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['method']:<16}{r['aurc']:>8.4f}{r['ece']:>8.4f}"
              f"{r['coverage_at_alpha']:>9.4f}{r['risk_at_alpha']:>9.4f}{r['cp_upper']:>9.4f}")

    try:
        import matplotlib

        matplotlib.use("Agg")  # 服务器上没有显示器，必须用无 GUI 后端
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, (cov, risk) in curves.items():
            axes[0].plot(cov, risk, label=name, linewidth=1.6)
        axes[0].axhline(args.alpha, color="crimson", linestyle="--", linewidth=1.0,
                        label=f"α={args.alpha}")
        axes[0].set_xlabel("coverage（回答比例）")
        axes[0].set_ylabel("risk（回答里的错误率）")
        axes[0].set_title("Risk–Coverage 曲线（合成数据）")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 0.6)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        for name, conf in zip(names, confs):
            axes[1].hist(conf[correct], bins=20, alpha=0.5, label=f"{name} 答对")
            axes[1].hist(conf[~correct], bins=20, alpha=0.5, label=f"{name} 答错")
        axes[1].set_xlabel("置信度 S")
        axes[1].set_ylabel("题目数")
        axes[1].set_title("置信度分布（校准前后的差别）")
        axes[1].legend(fontsize=6)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_png = FIGURES_DIR / "fig01_risk_coverage_demo.png"
        fig.savefig(out_png, dpi=180)
        print(f"图：{out_png}")
    except ImportError:
        log.warning("没装 matplotlib，跳过画图（pip install matplotlib）")

    log.info("指标代码与画图链路 OK。换成真数据时只需把 conf/correct 从 features.csv 读进来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
