"""mvs —— 面向知识密集型问答的证据充分性判定与风险受控拒答。

模块划分对应论文的技术路线六步：
    data      读题、切块、切分
    retrieve  检索（bm25 / dense / hybrid / rerank）
    generate  生成（vLLM 兼容接口）
    verify    断言分解 + NLI 校验
    decide    特征、校准、阈值决策
    eval      指标、风险-覆盖曲线、bootstrap
"""

__version__ = "0.1.0"
