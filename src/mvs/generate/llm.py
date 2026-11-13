"""生成侧：统一的 LLM 客户端。

三种后端，同一套调用方式，切换只改配置：
    vllm_http  连服务器上 vLLM 暴露的 OpenAI 兼容接口（本课题主用）
    openai     任何付费 API（兜底，按 token 计费）
    mock       不调模型，返回固定结构 —— 本地没卡时用来联调解析、切分、指标

为什么要自己写这一层而不用 openai SDK：SDK 的版本兼容问题比 urllib 麻烦得多，
而且 vLLM 的兼容接口就是普通 HTTP，requests 足够。
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence

import requests

from mvs.utils.logging import get_logger

__all__ = ["LLMClient", "LLMConfig", "MockBackend"]

log = get_logger("mvs.llm")


@dataclass
class LLMConfig:
    backend: str = "vllm_http"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    timeout_s: float = 120.0
    max_retries: int = 3
    concurrency: int = 8
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0


@dataclass
class _Stats:
    calls: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    _lock: Any = field(default=None, repr=False)


class MockBackend:
    """确定性假模型：同样的输入永远给同样的输出，方便断言校验逻辑本身写对没有。"""

    def chat(self, messages: Sequence[dict]) -> dict:
        user = messages[-1]["content"] if messages else ""
        preview = user[:40].replace("\n", " ")
        content = json.dumps(
            {
                "answer": f"（mock）根据资料 [1] 可回答问题：{preview}",
                "citations": [1],
                "sufficient": True,
                "missing": "",
            },
            ensure_ascii=False,
        )
        return {
            "content": content,
            "usage": {"prompt_tokens": len(user) // 2, "completion_tokens": len(content) // 2},
        }


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.stats = _Stats()
        self._mock = MockBackend() if cfg.backend == "mock" else None
        self._url = cfg.base_url.rstrip("/")
        self._session = requests.Session()
        if cfg.backend != "mock":
            self._session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            })

    # ---------- 单次调用 ----------

    def chat(self, messages: Sequence[dict], *, response_format: dict | None = None) -> str:
        if self._mock is not None:
            resp = self._mock.chat(messages)
            self._record(resp["usage"], 0.0)
            return resp["content"]

        payload = {
            "model": self.cfg.model,
            "messages": list(messages),
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            started = time.time()
            try:
                resp = self._session.post(
                    f"{self._url}/chat/completions", data=json.dumps(payload),
                    timeout=self.cfg.timeout_s,
                )
                resp.raise_for_status()
                body = resp.json()
                text = body["choices"][0]["message"]["content"]
                self._record(body.get("usage", {}), time.time() - started)
                return text
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                self.stats.errors += 1
                wait = min(2 ** attempt + random.uniform(0, 1), 30)
                log.warning("调用失败（%d/%d）: %s —— %.1fs 后重试",
                            attempt, self.cfg.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM 调用连续失败 {self.cfg.max_retries} 次: {last_exc}")

    # ---------- 批量调用 ----------

    def chat_many(self, batch: Sequence[Sequence[dict]], concurrency: int | None = None,
                  *, response_format: dict | None = None) -> list[str | None]:
        """并发跑一批，顺序与输入一致；单条失败返回 None 而不炸掉整个批次。"""
        workers = max(1, min(concurrency or self.cfg.concurrency, len(batch) or 1))
        results: list[str | None] = [None] * len(batch)
        if workers == 1:
            for i, msgs in enumerate(batch):
                try:
                    results[i] = self.chat(msgs, response_format=response_format)
                except RuntimeError as exc:
                    log.error("第 %d 条失败: %s", i, exc)
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.chat, msgs, response_format=response_format): i
                for i, msgs in enumerate(batch)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except RuntimeError as exc:
                    log.error("第 %d 条失败: %s", idx, exc)
        return results

    # ---------- 运维 ----------

    def list_models(self) -> list[str]:
        """探活：vLLM 起来后 /v1/models 会返回模型名。"""
        if self._mock is not None:
            return ["mock"]
        resp = self._session.get(f"{self._url}/models", timeout=min(self.cfg.timeout_s, 20))
        resp.raise_for_status()
        return [item.get("id", "") for item in resp.json().get("data", [])]

    def _record(self, usage: dict, seconds: float) -> None:
        self.stats.calls += 1
        self.stats.seconds += seconds
        self.stats.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.stats.tokens_out += int(usage.get("completion_tokens") or 0)

    def cost(self) -> float:
        """按配置单价估算花费（本地部署为 0）。"""
        return (self.stats.tokens_in / 1e6 * self.cfg.price_per_1m_input
                + self.stats.tokens_out / 1e6 * self.cfg.price_per_1m_output)

    def summary(self) -> dict:
        avg = self.stats.seconds / self.stats.calls if self.stats.calls else 0.0
        return {
            "backend": self.cfg.backend,
            "model": self.cfg.model,
            "calls": self.stats.calls,
            "errors": self.stats.errors,
            "tokens_in": self.stats.tokens_in,
            "tokens_out": self.stats.tokens_out,
            "avg_seconds": round(avg, 3),
            "cost_rmb": round(self.cost(), 4),
        }
