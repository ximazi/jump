"""通用工具：日志、颜色、时间、数值安全转换。"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------- 颜色

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


_COLORS = {
    "DEBUG": C.GREY,
    "INFO": C.CYAN,
    "WARNING": C.YELLOW,
    "ERROR": C.RED,
    "CRITICAL": C.BOLD + C.RED,
}


class ColorFormatter(logging.Formatter):
    def __init__(self, color: bool = True):
        super().__init__("%(message)s")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        lvl = record.levelname
        if self.color:
            c = _COLORS.get(lvl, "")
            return f"{C.GREY}{ts}{C.RESET} {c}{lvl:<8}{C.RESET} {record.getMessage()}"
        return f"{ts} {lvl:<8} {record.getMessage()}"


def setup_logging(log_dir: str = "logs", level: str = "INFO", to_file: bool = True) -> logging.Logger:
    """控制台彩色 + 文件日志（文件带完整时间戳与行号）。"""
    logger = logging.getLogger("meme")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(color=sys.stdout.isatty()))
    logger.addHandler(console)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        fname = os.path.join(log_dir, f"meme_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(fname, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------- 数值

def to_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if f != f else f  # NaN 检查
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pct(a: float, b: float) -> float:
    """百分比变化，b 为基数。"""
    if b == 0:
        return 0.0
    return (a - b) / abs(b)


def fmt_usd(v: float) -> str:
    """把美元金额压成可读短串：$12.3M / $4.56K。"""
    av = abs(v)
    if av >= 1e9:
        return f"${v / 1e9:.2f}B"
    if av >= 1e6:
        return f"${v / 1e6:.2f}M"
    if av >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:.2f}"


def fmt_qty(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}K"
    if abs(v) >= 1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.6f}".rstrip("0").rstrip(".")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def utc_str(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def usdt_round_step(price: float, step: float) -> float:
    """按交易对的数量精度向下取整。"""
    if step <= 0:
        return price
    import math

    return math.floor(price / step + 1e-9) * step


def round_to_step(value: float, step: float, mode: str = "down") -> float:
    import math

    if step <= 0:
        return value
    if mode == "up":
        return math.ceil(value / step - 1e-9) * step
    return math.floor(value / step + 1e-9) * step
