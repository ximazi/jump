"""运行状态：持仓、日内盈亏、权益，JSON 持久化。

放在 state/ 目录下，程序重启后可恢复未平仓的纸面持仓。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger("meme.state")


@dataclass
class Position:
    symbol: str
    side: str = "SHORT"
    qty: float = 0.0
    remaining_qty: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profits: List[List[float]] = field(default_factory=list)  # [price, fraction, filled]
    opened_at: float = 0.0
    risk_amount: float = 0.0
    leverage: float = 0.0
    slippage: float = 0.003          # 建仓时估算的滑点，平仓时复用
    status: str = "OPEN"            # OPEN / CLOSED
    close_reason: str = ""
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    peak_profit_pct: float = 0.0    # 浮盈最高点，用来判断移动止损/保本
    stop_moved_to_breakeven: bool = False
    last_price: float = 0.0
    order_ids: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    mode: str = "PAPER"
    phase: str = ""             # 所属阶段：bounce（反弹腿）/ short（做空腿）
    event_day: str = ""         # 触发事件日 T0

    # -------------------------------------------------- 序列化

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    # -------------------------------------------------- 便捷属性

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "SHORT":
            return (self.entry - self.last_price) * self.remaining_qty
        return (self.last_price - self.entry) * self.remaining_qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry <= 0 or self.remaining_qty <= 0:
            return 0.0
        return self.unrealized_pnl / (self.entry * self.remaining_qty)

    @property
    def notional(self) -> float:
        return self.entry * self.remaining_qty

    def next_tp(self) -> Optional[List[float]]:
        for tp in self.take_profits:
            if not tp[2]:
                return tp
        return None


class StateStore:
    def __init__(self, directory: str = "state"):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self.positions_file = os.path.join(self.dir, "positions.json")
        self.account_file = os.path.join(self.dir, "account.json")

    # -------------------------------------------------- 持仓

    def load_positions(self) -> List[Position]:
        if not os.path.exists(self.positions_file):
            return []
        try:
            with open(self.positions_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [Position.from_dict(d) for d in raw]
        except Exception as e:
            log.error("读取持仓失败（%s），按空仓启动", e)
            return []

    def save_positions(self, positions: List[Position]) -> None:
        tmp = self.positions_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([p.to_dict() for p in positions], fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.positions_file)

    # -------------------------------------------------- 账户

    def load_account(self) -> Dict[str, Any]:
        default = {
            "equity": 0.0,
            "initial_equity": 0.0,
            "realized_pnl_today": 0.0,
            "day": "",
            "total_realized_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
        }
        if not os.path.exists(self.account_file):
            return default
        try:
            with open(self.account_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            default.update(data or {})
            return default
        except Exception:
            return default

    def save_account(self, acct: Dict[str, Any]) -> None:
        tmp = self.account_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(acct, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.account_file)

    def rollover_day(self, acct: Dict[str, Any]) -> Dict[str, Any]:
        """跨日重置日内盈亏与日内熔断状态。"""
        today = time.strftime("%Y-%m-%d")
        if acct.get("day") != today:
            acct["day"] = today
            acct["realized_pnl_today"] = 0.0
            acct["halted"] = False
        return acct
