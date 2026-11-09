"""切块：页面 -> 句子 -> 带重叠的块。

为什么按句子打包而不是按固定字数硬切：
硬切会把"因为…所以…"拦腰砍断，NLI 判定直接失效，矛盾度指标会凭空升高。
按句子打包 + 尾部重叠，能让跨句的推理至少在一个块里完整出现。
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

__all__ = ["split_sentences", "chunk_text", "iter_page_chunks"]

_SENT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _hard_split(sent: str, max_chars: int) -> list[str]:
    """单句超长（表格、代码、无标点长段）时退化为等长硬切。"""
    return [sent[i:i + max_chars] for i in range(0, len(sent), max_chars)]


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 150,
               min_chars: int = 80) -> list[str]:
    """把一段文本切成若干块，块间保留尾部重叠。"""
    sentences: list[str] = []
    for sent in split_sentences(text):
        if len(sent) > max_chars:
            sentences.extend(_hard_split(sent, max_chars))
        else:
            sentences.append(sent)
    if not sentences:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        if buf and buf_len + len(sent) > max_chars:
            chunks.append("".join(buf))
            keep: list[str] = []
            keep_len = 0
            for prev in reversed(buf):
                if keep_len + len(prev) > overlap_chars:
                    break
                keep.insert(0, prev)
                keep_len += len(prev)
            buf, buf_len = keep, keep_len
        buf.append(sent)
        buf_len += len(sent)

    if buf:
        chunks.append("".join(buf))

    keep_chunks = [c for c in chunks if len(c) >= min_chars]
    return keep_chunks or [c for c in chunks if c.strip()]


def iter_page_chunks(pages: Iterable[dict], doc_prefix: str = "doc",
                     max_chars: int = 800, overlap_chars: int = 150,
                     min_chars: int = 80) -> Iterator[dict]:
    """逐页产出块。doc_id 稳定可复现：<前缀>-<页序号>-<块序号>。"""
    for page_idx, page in enumerate(pages):
        text = page.get("text") or ""
        for chunk_idx, chunk in enumerate(
            chunk_text(text, max_chars, overlap_chars, min_chars)
        ):
            yield {
                "doc_id": f"{doc_prefix}-{page_idx:05d}-{chunk_idx:03d}",
                "page_idx": page_idx,
                "chunk_idx": chunk_idx,
                "title": page.get("title", ""),
                "url": page.get("url", ""),
                "support": page.get("support"),
                "text": chunk,
                "n_chars": len(chunk),
            }
