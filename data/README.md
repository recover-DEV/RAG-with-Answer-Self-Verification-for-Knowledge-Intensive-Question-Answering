# 数据目录说明

## 1. 目录约定

| 目录 | 存放内容 | 能不能改 | 入不入库 |
|---|---|---|---|
| `raw/` | 官方下载的原始文件，一个字节都不要动 | **只读**，永不手改 | 否 |
| `interim/` | 清洗/切块中间产物（chunks.jsonl） | 可随时删了重建 | 否 |
| `processed/` | 规整后带 split 标记的题目（questions.jsonl） | 可重建 | 否 |
| `index/` | BM25 / 向量索引（`*.pkl`、`*.faiss`）、`meta.json` | 可重建 | 否 |

**铁律：`raw` 只读。** 一旦手改原始数据，后面所有实验都无法复现，查重/答辩时也没法证明"结果来自公开数据"。

## 2. 数据来源

- **CRAG**：Meta 的 Comprehensive RAG Benchmark，NeurIPS 2024 Datasets & Benchmarks。
  官方代码仓库 `https://github.com/facebookresearch/CRAG`，数据获取方式以该仓库 README 为准（通常经由下载链接或 Hugging Face 数据集页）。
  规模：4409 道题，5 个领域（Finance / Sports / Music / Movie / Open），每题配套一组 mock web pages。
- **RGB**：仅用它的"负例拒答"子集，AAAI 2024，`https://github.com/chen700564/RGB`。

下载后在 `raw/crag/` 下应看到（文件名以官方为准，若有差异请改 `01_check_data.py` 里的列表）：

```
raw/crag/
├─ questions.jsonl     # 或 *.json / *.parquet
└─ pages.jsonl         # 页面池（若有）
```

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
    {"title": "...", "url": "...", "text": "...", "support": true}
  ]
}
```

`support` 字段为官方标注的"这一页是否支持答案"，用来算**检索侧上界（oracle recall）**——没有它就没法区分"检索没找对"还是"模型没用对"。

## 4. 必须记录的三件事（写进论文第 5 章"实验设置"）

1. 下载日期与来源 URL；
2. 文件字数与 md5（`Get-FileHash` / `sha256sum`）；
3. 切分文件的 md5（`processed/splits.json` 一并提交），证明 dev/cal/test 没有事后重切。
