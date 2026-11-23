"""仓位计算与止损止盈设计。

核心约束：
  - 单笔风险 = 权益 * risk_per_trade（默认 1%），仓位由止损距离反推
  - 杠杆硬上限 2~3x（用户策略建议），永远不超过 5x
  - 止损放在关键阻力位（近期前高）上方，而不是拍脑袋的固定百分比
  - 止损距离夹在 [1.5%, 12%]，过窄会被插针扫掉，过宽则仓位太小
  - 盈亏比 < min_rr 的标的直接放弃
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import BrokerConfig, RiskConfig
from .exchange import BinanceFutures
from .indicators import atr
from .screener import MemeCandidate
from .utils import round_to_step

log = logging.getLogger("meme.risk")


@dataclass
class TradePlan:
    symbol: str
    side: str = "SHORT"
    qty: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profits: List[Tuple[float, float]] = field(default_factory=list)  # (价格, 平仓比例)
    leverage: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0
    expected_profit: float = 0.0
    rr: float = 0.0
    slippage: float = 0.003
    notes: List[str] = field(default_factory=list)

    @property
    def tp1(self) -> float:
        return self.take_profits[0][0] if self.take_profits else 0.0

    @property
    def tp2(self) -> float:
        return self.take_profits[1][0] if len(self.take_profits) > 1 else 0.0

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.side} {self.qty:.4g} @ {self.entry:.6g} | "
            f"止损 {self.stop_loss:.6g} | 止盈 {self.tp1:.6g}/{self.tp2:.6g} | "
            f"杠杆 {self.leverage:.2f}x | 风险 ${self.risk_amount:.2f} | RR {self.rr:.2f}"
        )


def parse_step(filters: List[Dict[str, Any]], filter_type: str, key: str = "stepSize") -> float:
    for f in filters or []:
        if f.get("filterType") == filter_type:
            try:
                return float(f.get(key, 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def estimate_slippage(
    candidate: MemeCandidate,
    order_notional: float,
    broker_cfg: BrokerConfig,
    side: str = "SHORT",
) -> float:
    """按盘口深度估算本次下单的真实滑点。

    做空市价卖单砸向买盘（bid_depth_usd），做多市价买单吃卖盘（ask_depth_usd）。
    深度未知时按 2 倍基准惩罚处理。
    """
    base = broker_cfg.slippage_pct
    spread_cost = min(0.005, candidate.spread_bps / 1e4)   # 点差成本，封顶 0.5%
    if side.upper() == "SHORT":
        depth = candidate.bid_depth_usd or candidate.ask_depth_usd
    else:
        depth = candidate.ask_depth_usd
    if depth <= 0:
        return base * 2 + spread_cost
    if order_notional <= depth:
        return base + spread_cost
    ratio = order_notional / depth
    penalty = broker_cfg.slippage_depth_factor * min(1.0, ratio - 1.0) * 3 * base
    return base + spread_cost + penalty


def build_plan(
    candidate: MemeCandidate,
    equity: float,
    cfg: RiskConfig,
    ex: BinanceFutures,
    perp_meta: Optional[Dict[str, Any]] = None,
    existing_notional: float = 0.0,
    existing_positions: int = 0,
    broker_cfg: Optional[BrokerConfig] = None,
    side: str = "SHORT",
    risk_per_trade: Optional[float] = None,
    max_leverage: Optional[float] = None,
    structure_stop: Optional[float] = None,
) -> Optional[TradePlan]:
    """为一个候选标的生成交易计划；不合格返回 None。

    side = "SHORT" 做空；side = "LONG" 做多（反弹腿）。
    多空在四个地方不同：入场滑点方向、止损位方向、止盈方向、每单位风险。
    """
    if equity <= 0:
        log.warning("权益为 0，无法建仓")
        return None
    if existing_positions >= cfg.max_positions:
        log.info("已达最大持仓数 %d，跳过 %s", cfg.max_positions, candidate.symbol)
        return None

    is_long = side.upper() == "LONG"
    risk_pct = cfg.risk_per_trade if risk_per_trade is None else risk_per_trade
    lev_cap = cfg.max_leverage if max_leverage is None else max_leverage

    price = candidate.price
    if price <= 0:
        return None

    broker_cfg = broker_cfg or BrokerConfig()
    # 用单标的仓位上限作为参考下单额来估滑点（偏保守），避免和仓位计算互相依赖
    ref_notional = equity * cfg.max_position_pct
    slippage = estimate_slippage(candidate, ref_notional, broker_cfg, side=side)
    depth_used = (
        (candidate.bid_depth_usd or candidate.ask_depth_usd)
        if not is_long else candidate.ask_depth_usd
    )

    # 滑点按不利方向折价：
    #   做空 -> 在 bid 成交，入场价更低才是更差的结果（止损固定时亏更多）
    #   做多 -> 在 ask 成交，入场价更高才是更差的结果
    entry = price * (1.0 + slippage) if is_long else price * (1.0 - slippage)
    notes: List[str] = [
        f"入场价含滑点 {slippage * 100:.2f}%（{'做多' if is_long else '做空'}方向不利折价；"
        f"点差 {candidate.spread_bps:.1f}bp，0.5%盘口深度 ${depth_used / 1e6:.2f}M）"
    ]

    # ---- 止损位 -------------------------------------------------------
    # 做空：止损在近 20 日前高（阻力位）之上；做多：止损在近 20 日前低（支撑位）之下
    df = ex.klines(candidate.symbol, "1d", limit=40)
    stop_raw = None
    if df is not None and len(df) >= 20:
        a = atr(df, 14)
        atr_val = float(a.iloc[-1]) if (len(a) >= 2 and a.iloc[-1] > 0) else 0.0
        if is_long:
            prior_low = float(df["low"].iloc[-21:-1].min()) if len(df) >= 21 else float(df["low"].iloc[-20:-1].min())
            # 波动率下限：止损不得窄于 stop_atr_mult 倍 ATR。
            # 暴跌后做多时「近期低点」往往离现价只有 1% 不到，那是噪音宽度，
            # 照搬结构位必然被扫。ATR 是这类标的唯一可靠的止损尺度。
            vol_stop = entry - cfg.stop_atr_mult * atr_val if atr_val > 0 else entry * (1 - cfg.min_stop_pct * 2)
            if prior_low < entry and prior_low * 0.995 < vol_stop:
                # 前低远在下方，是更强的支撑，用它放宽止损
                stop_raw = prior_low * 0.995
                notes.append(f"止损基于 20 日前低 {prior_low:.6g}")
            else:
                stop_raw = vol_stop
                notes.append(
                    f"止损按 ATR 波动率下限 {cfg.stop_atr_mult}x（{vol_stop:.6g}）；"
                    f"20 日前低 {prior_low:.6g} 离现价太近或已被击穿，不可作支撑"
                )
        else:
            prior_high = float(df["high"].iloc[-21:-1].max()) if len(df) >= 21 else float(df["high"].iloc[-20:-1].max())
            stop_raw = prior_high * 1.005
            notes.append(f"止损基于 20 日前高 {prior_high:.6g}")
            if atr_val > 0:
                atr_stop = entry + cfg.stop_atr_mult * atr_val
                # ATR 止损更近时取更近的（风控优先），但不能贴到前高保护之内
                if atr_stop < stop_raw and atr_stop > entry * (1 + cfg.min_stop_pct):
                    stop_raw = atr_stop
                    notes.append(f"ATR 收紧止损至 {atr_stop:.6g}")
            if structure_stop is not None and structure_stop > entry:
                # 盘中反弹受阻结构位比日线止损更紧时，优先采用更紧结构止损以提升盈亏比
                if stop_raw is None or structure_stop < stop_raw:
                    stop_raw = structure_stop
                    notes.append(f"采用盘中反弹受阻结构止损 {structure_stop:.6g}")

    if stop_raw is None or (is_long and stop_raw >= entry) or (not is_long and stop_raw <= entry):
        stop_raw = entry * (1 - cfg.min_stop_pct * 2) if is_long else entry * (1 + cfg.min_stop_pct * 2)
        notes.append("无历史数据，使用默认止损")

    if is_long:
        min_stop = entry * (1 - cfg.min_stop_pct)
        max_stop = entry * (1 - cfg.max_stop_pct)
    else:
        min_stop = entry * (1 + cfg.min_stop_pct)
        max_stop = entry * (1 + cfg.max_stop_pct)
    if (is_long and stop_raw > min_stop) or (not is_long and stop_raw < min_stop):
        stop_raw = min_stop
        notes.append(f"止损放宽到最小间距 {cfg.min_stop_pct * 100:.1f}%")
    elif (is_long and stop_raw < max_stop) or (not is_long and stop_raw > max_stop):
        stop_raw = max_stop
        notes.append(f"止损截断至最大 {cfg.max_stop_pct * 100:.1f}%")

    risk_per_unit = (entry - stop_raw) if is_long else (stop_raw - entry)
    if risk_per_unit <= 0:
        return None

    # ---- 止盈 ---------------------------------------------------------
    sign = 1.0 if is_long else -1.0
    tp1 = entry * (1 + sign * cfg.tp1_pct)
    tp2 = entry * (1 + sign * cfg.tp2_pct)
    tps = [(tp1, cfg.tp1_fraction)]
    if cfg.tp2_pct > 0:
        tps.append((tp2, cfg.tp2_fraction))

    # ---- 仓位 ---------------------------------------------------------
    risk_budget = equity * risk_pct
    qty_by_risk = risk_budget / risk_per_unit

    pos_cap = equity * cfg.max_position_pct
    qty_by_position = pos_cap / entry

    total_cap = equity * cfg.max_total_notional_pct
    room = max(0.0, total_cap - existing_notional)
    qty_by_total = room / entry if entry > 0 else 0.0

    qty = min(qty_by_risk, qty_by_position, qty_by_total)
    if qty <= 0:
        log.info("%s 总敞口额度已用尽，跳过", candidate.symbol)
        return None

    # 按交易对数量精度向下取整
    step = parse_step((perp_meta or {}).get("filters", []), "LOT_SIZE", "stepSize")
    if step > 0:
        qty = round_to_step(qty, step, "down")
    if qty <= 0:
        return None

    notional = qty * entry
    leverage = notional / equity
    if leverage > lev_cap:
        # 杠杆是天花板，超了就按杠杆缩仓位（并按新仓位重算风险）
        scale = lev_cap / leverage
        qty *= scale
        step2 = parse_step((perp_meta or {}).get("filters", []), "LOT_SIZE", "stepSize")
        if step2 > 0:
            qty = round_to_step(qty, step2, "down")
        if qty <= 0:
            return None
        notional = qty * entry
        leverage = notional / equity

    # ---- 净盈亏比校验（扣除双边手续费）---------------------------------
    expected = 0.0
    remaining = 1.0
    for price_tp, frac in tps:
        expected += abs(price_tp - entry) * qty * frac
        remaining -= frac
    if remaining > 1e-6:
        expected += abs(tp2 - entry) * qty * remaining
    rt_fee_per_unit = entry * broker_cfg.taker_fee * 2.0
    net_expected = max(0.0, expected - rt_fee_per_unit * qty)
    net_risk = qty * (risk_per_unit + rt_fee_per_unit)
    rr = net_expected / max(1e-9, net_risk)
    if rr < cfg.min_rr:
        log.info("%s(%s) 净盈亏比 %.2f < %.2f，放弃", candidate.symbol, side, rr, cfg.min_rr)
        return None

    plan = TradePlan(
        symbol=candidate.symbol,
        side="LONG" if is_long else "SHORT",
        qty=qty,
        entry=entry,
        stop_loss=stop_raw,
        take_profits=tps,
        leverage=leverage,
        risk_amount=qty * risk_per_unit,
        notional=notional,
        expected_profit=expected,
        rr=rr,
        slippage=slippage,
        notes=notes,
    )
    stop_pct = risk_per_unit / entry
    notes.append(
        f"实际杠杆 {leverage:.2f}x ≈ 风险预算 {risk_pct:.1%} ÷ 止损 {stop_pct:.1%}"
        f"（杠杆上限 {lev_cap}x 为天花板，风控实际约束更强）"
    )
    return plan
