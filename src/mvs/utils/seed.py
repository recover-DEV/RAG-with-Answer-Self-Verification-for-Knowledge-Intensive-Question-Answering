"""随机种子：一次设齐，保证同一配置重跑结果一致（至少同一台机器上一致）。

论文里凡是报"多次重复的均值±标准差"，就必须先有这一层。
"""

from __future__ import annotations

import os
import random

__all__ = ["set_seed"]

_SETS: set[int] = set()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy 没装也不该炸
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 会略微降低速度，但换来可复现；只在需要严格对齐时开
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    _SETS.add(seed)
