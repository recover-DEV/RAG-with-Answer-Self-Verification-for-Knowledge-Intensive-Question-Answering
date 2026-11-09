"""流水线编排：一步一个脚本，每步的产物落盘，下一步只读上一步的文件。

    build_index.py        原始数据 -> chunks.jsonl -> bm25.pkl
    run_generation.py     chunks + questions -> gen.jsonl
    run_verify.py         gen.jsonl -> claims.jsonl
    make_features.py      以上 -> features.csv（唯一分析入口）
    run_decision.py       features.csv -> 决策与指标

拆成多步而不是一个大 main，是为了能在服务器上"跑到一半断了就接着跑"，
也为了能单独重跑被改掉的那一环（比如只换 NLI 模型）。
"""
