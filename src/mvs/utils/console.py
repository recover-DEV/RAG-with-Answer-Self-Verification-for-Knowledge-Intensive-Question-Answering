"""控制台输出修正：让脚本在 Windows 的 GBK 控制台里也不崩。

问题：Windows 中文系统的控制台编码是 cp936(GBK)。脚本里只要打印一个
GBK 编不出的字符（✓ ✗ → 之类），Python 就抛 UnicodeEncodeError，
整个脚本在"打印一句提示"这一步崩掉——跟研究内容毫无关系，
但能让一个晚上白费。

做法：不改变控制台编码（改了中文会变乱码），只把错误处理改成 replace，
编不出的字符变成 ? 而不是异常。同时脚本里统一用 [OK]/[WARN]/[FAIL]
这类 ASCII 标记，这样即使被 replace 也仍然是可读的字。

用法：脚本开头 import 后立刻调用一次。
    from mvs.utils.console import init_console
    init_console()
"""

from __future__ import annotations

import sys


def init_console() -> None:
    """把 stdout/stderr 的错误处理改成 replace。可重复调用，无副作用。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            # 被重定向成不可重配置的对象（如 pytest 捕获）时静默跳过
            pass
