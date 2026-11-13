# MVS · 面向知识密集型问答的答案自验证 RAG

本仓库用于本科毕设实验。以 Meta CRAG Task 1 为主要评测场景，研究如何提高检索证据的相关性、判断答案是否有证据支持，以及在证据不足时选择重试或拒答。

## 研究范围与实现进度

| 研究内容 | 实验设计 | 仓库现状 |
|---|---|---|
| 检索基线与重排序 | 比较 BM25、向量检索和混合检索。分别比较文本切分策略、召回数及重排序方法 | 已有切块、BM25 建索引与检索。向量、混合检索和重排序待实现。当前 `rerank_top_n` 只截取 BM25 排序结果 |
| 答案–证据一致性评分 | 融合检索相关性、断言受证据支持程度、证据对问题的充分性 | 已有提示词、生成客户端和输出解析。三维评分待实现 |
| 自适应决策 | 一致性达标时回答。按证据和答案状态选择重检索、重新生成或拒答。重试后复验 | 重检索、重新生成、复验和拒答策略待实现。已有部分风险–覆盖指标函数 |
| 对比、消融与错误分析 | 在官方公开测试集报告答案准确率、拒答准确率与端到端推理延迟，复核检索失效、答案冲突、误拒及重试失败的案例 | 数据准备已保留官方测试集。生成时仍会跳过无检索块的题，逐题完整评测和实验汇总待完成 |

检索实验固定生成模型与评判器。验证与决策实验固定首轮检索和生成配置，比较普通 RAG、仅验证后拒答与完整自适应方法。重检索更新证据后再生成，重新生成沿用当前证据。首版把额外重试预算设为每题一次。无可用证据且无法重检索时可直接拒答，重试后依据复验结果回答或拒答。

预期交付可复现的实验框架、带答案自验证和自适应动作的 RAG 方法，以及包含基线对比、消融与错误案例的实验报告。

研究设计见[技术路线](docs/01_毕设完成路线与技术调研.md)，分周交付与实验编号见[实验排期](docs/02_学习与实验计划_按周排期.md)，数据口径见[数据说明](data/README.md)。

---

## 本地 CPU 自检与 BM25 冒烟

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/00_check_env.py                     # 环境自检，最后一行是结论
python scripts/01_prepare_data.py --limit 50       # 规整 + 切分 + 切块
python scripts/02_build_index.py                   # 建 BM25 索引
```

准备数据前，将解包后的官方 CRAG JSONL 放在 `data/raw/crag/`。`--limit` 会将本地派生产物重建为小样本，正式实验前需要重新处理全量数据。当前 `configs/exp/e0_smoke.yaml` 默认调用 vLLM HTTP 服务。本地模拟生成需要在实验配置中把 `generator.backend` 覆盖为 `mock`。服务可用或配置好模拟后，运行下方的生成命令。[服务器部署指南](docs/服务器部署指南.md)

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
│   ├── retrieve/             #   已有 BM25，计划接入向量、混合检索与重排
│   ├── generate/             #   prompt 模板 + LLM 客户端（vLLM / API / mock）
│   ├── verify/               #   预留：断言分解、证据校验与一致性评分
│   ├── decide/               #   预留：阈值校准与动作选择
│   ├── pipeline/             #   预留：重检索、重新生成与复验的流程编排
│   ├── eval/                 #   已有风险–覆盖、AURC、ECE 等指标函数
│   └── utils/                #   日志、随机种子、jsonl/CSV 读写
├── scripts/                  # 入口脚本，编号即执行顺序
│   ├── 00_check_env.py       #   环境自检
│   ├── 01_prepare_data.py    #   规整 + 切分 + 切块
│   ├── 02_build_index.py     #   建 BM25 索引
│   ├── 03_generate.py        #   生成（检索→prompt→解析→落盘）
│   ├── 04_risk_coverage_demo.py  # 指标与画图的干跑
│   ├── 09_smoke_vllm.py      #   vLLM 服务冒烟测试
│   ├── serve_vllm.sh         #   起服务
│   ├── run_bg.sh             #   后台跑长任务
│   └── onboarding_server.sh  #   服务器体检（不改环境）
├── results/                  # 实验产物（不进 git，但要备份）
│   ├── runs/<run_id>/        #   一次实验一个目录：gen.jsonl + config.yaml + stats.json
│   ├── features/features.csv #   计划产物：逐题对齐的实验汇总表
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

**③ 用同一题集对齐各方法的结果。**
计划用 `results/features/features.csv` 汇总数据，每个 run 的每道题一行，记录题目 ID、检索配置、一致性分项、初始与最终判定、触发动作、拒答标记及分阶段时延。保留初始和复验的证据、答案及分数，以便复查。缺证据、拒答和执行失败的样本也要计入对应测试集。

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
bash scripts/run_bg.sh gen_full python scripts/03_generate.py --split all   # 服务器上，遍历所有划分
tail -f logs/gen_full_*.log                    # 看进度
```

---

## 实验记录

每次实验在 [`docs/实验台账.md`](docs/实验台账.md) 加一行。
台账是写论文第 5 章时的唯一依据，也是导师问"你做过什么"时的答案。
