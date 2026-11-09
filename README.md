# MVS：证据充分性判定与风险受控拒答（本科毕设实验仓库）

课题：《面向知识密集型问答的证据充分性判定与风险受控拒答方法研究与系统实现》

这个仓库只放**代码、配置、日志和实验产物**，不放数据本体和论文正文。
目标是在任何一台机器（笔记本 / 服务器）上用同一条命令复现同一批数字。

---

## 快速开始（5 步，本地即可，不需要 GPU）

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python scripts/00_check_env.py                     # 环境自检，最后一行是结论
python scripts/01_prepare_data.py --limit 50       # 规整 + 切分 + 切块
python scripts/02_build_index.py --eval-n 50       # 建 BM25 索引 + 算 oracle recall
python scripts/03_generate.py --limit 5            # 检索→prompt→（mock）生成→解析
```

要上服务器，看 [`docs/服务器部署指南.md`](docs/服务器部署指南.md)。第一次用服务器尤其建议按它第 1 节的顺序来。

---

## 目录结构

```
实验/
├── configs/                  # 配置（实验 = 配置的组合，不改代码）
│   ├── base.yaml             #   全局：随机种子、路径、环境记录
│   ├── data/crag.yaml        #   数据集：切分比例、切块参数、检索参数
│   ├── model/*.yaml          #   模型：生成器 / 校验器 / 断言分解
│   └── exp/*.yaml            #   具体实验：extends 上面的，只写差异
├── data/                     # 数据（不进 git）—— 详见 data/README.md
│   ├── raw/                  #   原始下载文件，只读，一个字节都不改
│   ├── interim/              #   中间产物：切块结果
│   ├── processed/            #   可直接用的：questions.jsonl / splits.json / stats.json
│   └── index/                #   索引文件：bm25.pkl + meta.json
├── src/mvs/                  # 源代码（packaged，可 import）
│   ├── config.py             #   配置加载：extends 继承 + ${ENV} 展开
│   ├── paths.py              #   所有路径的唯一来源 + run_id 生成
│   ├── data/                 #   读数据、规整、切分、切块
│   ├── retrieve/             #   BM25（自实现）+ 后续的稠密检索/重排
│   ├── generate/             #   prompt 模板 + LLM 客户端（vLLM / API / mock）
│   ├── verify/               #   断言分解 + NLI 校验（第 10—12 周实现）
│   ├── decide/               #   校准 + 风险受控决策（第 12—13 周实现）
│   ├── eval/                 #   指标：risk-coverage、AURC、ECE、保形上界
│   └── utils/                #   日志、随机种子、jsonl/CSV 读写
├── scripts/                  # 入口脚本，编号即执行顺序
│   ├── 00_check_env.py       #   环境自检
│   ├── 01_prepare_data.py    #   规整 + 切分 + 切块
│   ├── 02_build_index.py     #   建索引 + oracle recall
│   ├── 03_generate.py        #   生成（检索→prompt→解析→落盘）
│   ├── 04_risk_coverage_demo.py  # 指标与画图的干跑
│   ├── 09_smoke_vllm.py      #   vLLM 服务冒烟测试
│   ├── serve_vllm.sh         #   起服务
│   ├── run_bg.sh             #   后台跑长任务
│   └── onboarding_server.sh  #   服务器体检（不改环境）
├── results/                  # 实验产物（不进 git，但要备份）
│   ├── runs/<run_id>/        #   一次实验一个目录：gen.jsonl + config.yaml + stats.json
│   ├── features/features.csv #   ★ 所有方法共用的宽表，唯一分析入口
│   ├── tables/               #   论文里的表（CSV）
│   └── figures/              #   论文里的图（PNG）
├── logs/                     # 运行日志（不进 git）
├── tests/                    # 单元测试：指标公式、BM25、配置与切分
├── docs/                     # 部署指南、实验台账
└── pyproject.toml            # 打包与工具配置
```

---

## 三条铁律

这三条不是形式主义，每一条都对应一个会毁掉实验结论的具体错误。

**① `data/raw/` 只读，永不修改。**
原始文件一旦被改，你就再也无法证明"数据是这样处理的"。
所有清洗、规整都写到 `interim/` 或 `processed/`，原始文件旁边留 `SOURCE.md` 记下载日期、URL、md5。

**② 一次实验一个 `results/runs/<run_id>/` 目录，永不覆盖。**
`run_id` 由 `mvs.paths.new_run_id(prefix, tag)` 生成，形如 `E03_bm25top5_20261116-1430`
（`前缀_标签_时间戳`，按名字排序就是按时间排序）。
目录里必须有 `config.yaml`（本次实验的完整配置快照）和 `stats.json`（题数、耗时、解析成功率）。
**没有配置快照的 run 目录等于废数据**——一个月后你无法解释"这个数字是哪来的"。

**③ `results/features/features.csv` 是唯一分析入口。**
一行一题，列是：`qid, split, method, conf, correct, coverage_deficit, contradiction, retrieval_score, ...`
所有方法的对比、所有消融、所有图，都从这一张宽表算出来。
**不要**在各自的 gen.jsonl 上分别算指标——那会让"同一批题"这个前提悄悄失效。

---

## run_id 命名规范

```
<阶段><序号>_<方法简写>_<YYYYMMDD-HHMM>
E01_bm25top5_20261116-1430
A03_nocalib_20270210-0930
G02_qwen8b_20270218-1400
```

- 阶段：`P` 预研 / `E` 主实验 / `A` 消融 / `G` 换生成模型 / `S` 敏感性与种子
- 只用小写字母、数字、下划线；**不要**用中文和空格（rsync 和 shell 会折腾你）

---

## 常用命令

```bash
python scripts/00_check_env.py                 # 开工先跑这个
pytest -q                                      # 改完指标代码跑这个
python scripts/03_generate.py --config configs/exp/e0_smoke.yaml --limit 5
bash scripts/run_bg.sh gen_full python scripts/03_generate.py --split all   # 服务器上
tail -f logs/gen_full_*.log                    # 看进度
```

---

## 实验记录

每次实验在 [`docs/实验台账.md`](docs/实验台账.md) 加一行。
台账是写论文第 5 章时的唯一依据，也是导师问"你做过什么"时的答案。
