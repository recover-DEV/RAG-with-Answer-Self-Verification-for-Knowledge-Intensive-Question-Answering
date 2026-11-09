"""断言级校验：把答案拆成原子断言，逐条判 支持 / 中立 / 矛盾。

计划（第 6 周，需 GPU）：
    decompose.py  规则版（本地可跑）→ LLM 版（服务器）
    nli.py        DeBERTa-v3-large-mnli 批量推理，输出每条断言的
                  p_support / p_contradiction
    claims.py     聚合成两个特征：coverage = 被支持断言占比，
                  contradiction = 存在矛盾的断言占比

本目录当前为空是有意的：它依赖模型推理，本地无法完成，
先不要写假实现骗过流水线 —— 那会让后面所有指标建立在空气上。
"""
