"""配置加载：YAML + extends 继承 + ${ENV} 环境变量展开。

为什么要这一层：毕设会跑十几个实验，如果每个实验都复制一份完整配置，
改一个模型名就要改十几处，且事后说不清"当时到底跑的是哪个配置"。

用法：
    from mvs.config import load_config
    cfg = load_config("configs/exp/e0_smoke.yaml")
    cfg.generator.model            # 属性式访问
    load_config(p, ["generator.temperature=0.7"])   # 命令行临时覆盖
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["load_config", "load_raw", "apply_overrides", "to_dict", "save_config"]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """把 ${VAR} / ${VAR:-default} 换成环境变量值。"""
    if isinstance(value, str):

        def _sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, "" if default is None else default)

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, Mapping):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_expand(v) for v in value]
    return value


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是映射（key: value）: {path}")
    return data


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(out.get(key), dict) and isinstance(val, Mapping):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    """递归解析 extends，父配置先合并、子配置后覆盖。"""
    if path in stack:
        raise ValueError(f"extends 形成循环引用: {' -> '.join(str(p) for p in stack + (path,))}")
    own = _read_yaml(path)
    parents = own.pop("extends", None)
    if not parents:
        return own
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for item in parents:
        parent_path = (path.parent / str(item)).resolve()
        merged = _deep_merge(merged, _resolve(parent_path, stack + (path,)))
    return _deep_merge(merged, own)


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"覆盖项必须是 key=value 形式: {item!r}")
    key, _, raw = item.partition("=")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key.strip().split("."), value


def apply_overrides(data: dict, overrides: Iterable[str] | Mapping[str, Any] | None) -> dict:
    """支持 {"a.b": 1} 和 ["a.b=1"] 两种写法。"""
    if not overrides:
        return data
    out = copy.deepcopy(data)
    items = overrides.items() if isinstance(overrides, Mapping) else (
        _parse_override(o) for o in overrides)
    for keys, value in items:
        cursor = out
        for key in keys[:-1]:
            node = cursor.get(key)
            if not isinstance(node, dict):
                node = {}
                cursor[key] = node
            cursor = node
        cursor[keys[-1]] = value
    return out


def load_raw(path: str | Path, overrides=None) -> dict:
    """读出合并后的纯 dict（不转 Namespace），适合存档、写日志。"""
    resolved = _resolve(Path(path).resolve())
    return _expand(apply_overrides(resolved, overrides))


def load_config(path: str | Path, overrides=None) -> SimpleNamespace:
    """读出配置并转成属性式对象。"""
    return _to_ns(load_raw(path, overrides))


def to_dict(obj: Any) -> Any:
    """Namespace -> dict，用于 yaml.dump 存档。"""
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def save_config(obj: Any, path: str | Path) -> Path:
    """把实际生效的配置落盘到 run 目录，作为"这次实验到底跑了什么"的证据。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(obj), fh, allow_unicode=True, sort_keys=False)
    return out
