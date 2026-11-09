"""决策层：从特征到"答还是拒"，以及可解释的拒答理由。

计划（第 10—12 周，纯 CPU）：
    features.py    把 verify / retrieve 的中间量拼成一张宽表 features.csv
    calibrate.py   温度缩放 / 等渗回归（在 dev 上拟合）
    risk_control.py 在 cal 上按 Clopper-Pearson 上界选 τ；在 test 上只做应用
    explain.py     生成拒答理由："缺少关于 X 的证据" / "资料与答案矛盾"

这一层是全课题唯一真正的"方法"，但要记住：它的合法性来自
"阈值只在 cal 上选、test 只报数"这条纪律，而不是模型多复杂。
"""
