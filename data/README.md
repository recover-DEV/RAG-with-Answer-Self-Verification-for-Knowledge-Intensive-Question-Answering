# 数据目录说明

## 1. 目录约定

| 目录 | 存放内容 | 能不能改 | 入不入库 |
|---|---|---|---|
| `raw/` | 官方下载的原始文件，一个字节都不要动 | **只读**，永不手改 | 否 |
| `interim/` | 清洗/切块中间产物（chunks.jsonl） | 可随时删了重建 | 否 |
| `processed/` | 规整后带 split 标记的题目（questions.jsonl） | 可重建 | 否 |
| `index/` | 当前为 BM25 索引与 `meta.json`，后续加入向量索引 | 可重建 | 否 |

**铁律：`raw` 只读。** 一旦手改原始数据，后面所有实验都无法复现，查重/答辩时也没法证明"结果来自公开数据"。

## 2. 数据来源

- **CRAG Task 1** 是主数据集。使用官网提供的 Task 1 QA 与检索内容，覆盖五个领域，每题最多五个候选网页。项目使用的 v4 公开文件共 2,706 题，含验证集和公开测试集。[官方数据说明与下载](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md#download-crag-data)
- 原课题的数据集列表还列有 **RAG**，其全名、来源和版本待确认。确认后记录下载方式、数据划分和评测协议。
- **RGB** 的负例子集用于可选的拒答补充实验。[RGB 项目](https://github.com/chen700564/RGB)

解包后将 JSONL 放在 `data/raw/crag/`。`01_prepare_data.py` 会扫描这个目录的 `.jsonl` 和 `.json` 文件，同一次运行只放计划使用的数据版本。

```
data/raw/crag/
└─ crag_task_1_and_2_dev_v4.jsonl
```

规整脚本保留官方 `split=1` 的 1,335 道公开测试题。仅对 `split=0` 的验证集按题目 ID 哈希划分 `dev` 和 `cal`，目标比例为 40% 和 60%，实际题数以 `data/processed/stats.json` 为准。官方 `interaction_id` 用作 `qid`，脚本同时保留 `query_time`、答案别名和网页原始顺序。[官方字段说明](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md#data-schema) · [CRAG 论文实验设置](https://arxiv.org/html/2406.04744#S5.SS1)

数据准备的源配置是 `configs/data/crag.yaml`。`data/processed/data_config.yaml` 由 `01_prepare_data.py` 生成，记录本次准备数据使用的切块、切分和输出路径。

## 3. 规整后的 schema（`processed/questions.jsonl`，每行一条）

```json
{
  "qid": "crag-fin-0001",
  "question": "……",
  "answer": "参考答案（金标）",
  "answer_aliases": ["别名1", "别名2"],
  "domain": "finance",
  "question_type": "simple",
  "split": "dev|cal|test",
  "pages": [
    {"title": "...", "url": "...", "text": "...", "support": null}
  ]
}
```

当前 `text` 取自 `page_snippet`，用作切块、检索与重排序的文本来源。CRAG Task 1 的公开网页字段没有逐页 `support`，因此 `null` 表示未标注。`support_page_ratio=0` 反映没有可用的正支持标签。证据 Recall@k 等指标仅在另行标注的子集上报告，问答准确率可以用官方参考答案评测。拒答准确率另需基于固定候选证据池标注应答或应拒答。[官方网页字段](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md#search-results-detail)

## 4. 必须记录的三件事（写进论文第 5 章"实验设置"）

1. 下载日期与来源 URL；
2. 原始文件大小和 SHA-256（`Get-FileHash` 或 `sha256sum`）。
3. `data/processed/splits.json` 的 SHA-256、各划分题数和官方题目 ID 对齐情况。
