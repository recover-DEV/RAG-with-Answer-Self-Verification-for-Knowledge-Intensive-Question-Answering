"""Prompt 模板：所有实验共用一个模板，改模板等于换实验，必须走配置。

模板里的 {context} 用 [1] [2] ... 编号，生成的答案必须带引用编号，
后面的"断言级校验"才能把每条断言对回具体页码 —— 这是整个方法的地基。
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["SYSTEM_PROMPT", "build_context", "build_qa_messages", "DECOMPOSE_PROMPT",
           "OUTPUT_SCHEMA_HINT"]

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能依据给定资料回答，不得使用资料之外的知识。"
    "若资料不足以回答，必须明确说明缺少什么信息，不要猜测。"
    "回答必须给出引用编号，格式如 [1]、[2]。"
)

OUTPUT_SCHEMA_HINT = (
    "按要求输出 JSON，不要输出任何多余文字：\n"
    '{"answer": "答案正文，含 [n] 引用", "citations": [1, 2], '
    '"sufficient": true, "missing": "若 insufficient，写缺少什么；否则空字符串"}'
)

DECOMPOSE_PROMPT = (
    "把下面的答案拆成若干条不可再分的原子断言，每条只陈述一个事实，"
    "保留原句中的引用编号。输出 JSON 数组，形如 "
    '[{"claim": "断言文本", "citations": [1]}]\n\n答案：\n{answer}'
)


def build_context(pages: Sequence[dict], max_chars: int = 6000) -> str:
    """把检索到的页拼成带编号的上下文，超长则截断（宁可少给，不要爆上下文）。"""
    blocks: list[str] = []
    used = 0
    for idx, page in enumerate(pages, 1):
        title = (page.get("title") or "").strip() or f"page-{idx}"
        text = (page.get("text") or "").strip()
        block = f"[{idx}] {title}\n{text}"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 200:
                blocks.append(block[:remain] + "…（截断）")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_qa_messages(question: str, pages: Sequence[dict],
                      max_context_chars: int = 6000) -> list[dict]:
    context = build_context(pages, max_context_chars)
    user = (
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def iter_batches(items: Iterable, size: int) -> Iterable[list]:
    buf: list = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
