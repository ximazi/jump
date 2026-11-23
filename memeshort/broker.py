"""执行层：纸面撮合（Paper）与实盘（Live）共用同一套决策逻辑。

决策顺序（对空头而言按最坏情况优先）：
    止损 -> 轧空信号 -> 止盈 -> 保本移动止损

成交模型（Paper）：
    - 开仓：做空成交价 = 计划入场价（已含预估滑点）
    - 止损：买入平仓，成交价 = 止损价 * (1 + 止损滑点)   —— 变差
    - 止盈：卖出平仓，成交价 = 止盈价 * (1 - 滑点)       —— 变差
    - 手续费：taker 费率逐笔扣除
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import BrokerConfig, RiskConfig
from .exchange import BinanceFutures
from .risk import TradePlan
from .state import Position

log = logging.getLogger("meme.broker")


@dataclass
class BrokerAction:
    kind: str        # OPEN / CLOSE / STOP / TP / BREAKEVEN / SQUEEZE
    symbol: str
    reason: str = ""
    price: float = 0.0
    pnl: float = 0.0
    fraction: float = 1.0
    qty: float = 0.0          # 本次平掉的实际数量
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseBroker:
    def __init__(
        self,
        ex: BinanceFutures,
        broker_cfg: BrokerConfig,
        risk_cfg: RiskConfig,
    ):
        self.ex = ex
        self.cfg = broker_cfg
        self.risk_cfg = risk_cfg
        self.mode = "PAPER"

    # ---------------------------------------------------- 抽象原语

    def _do_open(self, plan: TradePlan) -> Optional[Position]:
        raise NotImplementedError

    def _do_close(self, pos: Position, reason: str, price: float, fraction: float) -> BrokerAction:
        raise NotImplementedError

    def _do_sync(self, positions: List[Position]) -> List[Position]:
        raise NotImplementedError

    def _do_cancel_all(self, symbol: str) -> None:
        return None

    def _do_update_stop(self, pos: Position, new_stop: float) -> bool:
        return True

    # ---------------------------------------------------- 公共决策

    def manage(
        self, pos: Position, price: float, funding: float = 0.0
    ) -> List[BrokerAction]:
        """对一个持仓做一次状态判定，返回触发的动作列表。"""
        actions: List[BrokerAction] = []
        if pos.status != "OPEN" or pos.remaining_qty <= 0:
            return actions

        pos.last_price = price
        is_long = pos.side == "LONG"
        profit_pct = ((price - pos.entry) if is_long else (pos.entry - price)) / pos.entry \
            if pos.entry else 0.0
        pos.peak_profit_pct = max(pos.peak_profit_pct, profit_pct)

        # 1) 止损优先 —— 最坏情况先处理
        #    多头：跌破止损；空头：涨破止损
        stop_hit = (price <= pos.stop_loss) if is_long else (price >= pos.stop_loss)
        if stop_hit:
            actions.append(self._do_close(pos, "STOP", pos.stop_loss, 1.0))
            return actions

        # 2) 轧空信号：资金费率深度负值 -> 空头拥挤（只对空头腿有意义）
        if not is_long and funding <= self.risk_cfg.squeeze_funding:
            act = self._do_close(pos, "SQUEEZE", price, 1.0)
            act.extra["funding"] = funding
            actions.append(act)
            return actions

        # 3) 追踪止损（保护大幅浮盈，应对急跌后的剧烈反抽）
        trail_activate = getattr(self.risk_cfg, "trailing_activate_pct", 0.08)
        trail_callback = getattr(self.risk_cfg, "trailing_callback_pct", 0.04)
        if trail_activate > 0 and trail_callback > 0 and pos.peak_profit_pct >= trail_activate:
            if is_long:
                peak_price = pos.entry * (1.0 + pos.peak_profit_pct)
                trail_stop = peak_price * (1.0 - trail_callback)
                if trail_stop > pos.stop_loss:
                    pos.stop_loss = trail_stop
                    self._do_update_stop(pos, trail_stop)
                    actions.append(
                        BrokerAction(
                            kind="BREAKEVEN",
                            symbol=pos.symbol,
                            reason=f"追踪止损上移至 {trail_stop:.8g}",
                            price=trail_stop,
                        )
                    )
            else:
                trough_price = pos.entry * (1.0 - pos.peak_profit_pct)
                trail_stop = trough_price * (1.0 + trail_callback)
                if trail_stop < pos.stop_loss:
                    pos.stop_loss = trail_stop
                    self._do_update_stop(pos, trail_stop)
                    actions.append(
                        BrokerAction(
                            kind="BREAKEVEN",
                            symbol=pos.symbol,
                            reason=f"追踪止损下移至 {trail_stop:.8g}",
                            price=trail_stop,
                        )
                    )

        # 4) 止盈
        tp = pos.next_tp()
        if tp is not None:
            tp_hit = (price >= float(tp[0])) if is_long else (price <= float(tp[0]))
        else:
            tp_hit = False
        if tp_hit:
            frac = float(tp[1])
            unfilled_before = sum(1 for t in pos.take_profits if not t[2])
            close_frac = 1.0 if unfilled_before <= 1 else frac
            actions.append(self._do_close(pos, "TP", float(tp[0]), close_frac))
            tp[2] = 1  # 标记已触发
            # 第一档止盈后止损移至「含双边手续费保本位」，且同步更新交易所保护单
            if (
                pos.status == "OPEN"
                and self.risk_cfg.move_stop_to_breakeven
                and not pos.stop_moved_to_breakeven
            ):
                fee_buf = self.cfg.taker_fee * 2.5
                be_price = pos.entry * (1.0 + fee_buf) if is_long else pos.entry * (1.0 - fee_buf)
                better = (be_price > pos.stop_loss) if is_long else (be_price < pos.stop_loss)
                if better:
                    pos.stop_loss = be_price
                    pos.stop_moved_to_breakeven = True
                    self._do_update_stop(pos, be_price)
                    actions.append(
                        BrokerAction(
                            kind="BREAKEVEN",
                            symbol=pos.symbol,
                            reason=f"止损移至含费保本位 {be_price:.8g}",
                            price=be_price,
                        )
                    )
        return actions

    # ---------------------------------------------------- 对外接口

    def open_position(self, plan: TradePlan) -> Optional[Position]:
        return self._do_open(plan)

    # 兼容旧调用名
    open_short = open_position

    def close(self, pos: Position, reason: str, price: float, fraction: float = 1.0) -> BrokerAction:
        return self._do_close(pos, reason, price, fraction)

    def sync(self, positions: List[Position]) -> List[Position]:
        return self._do_sync(positions)


# ==================================================================== Paper


class PaperBroker(BaseBroker):
    def __init__(self, ex: BinanceFutures, broker_cfg: BrokerConfig, risk_cfg: RiskConfig):
        super().__init__(ex, broker_cfg, risk_cfg)
        self.mode = "PAPER"

    def _fee(self, notional: float) -> float:
        return notional * self.cfg.taker_fee

    def _do_open(self, plan: TradePlan) -> Optional[Position]:
        fill = plan.entry
        fee = self._fee(fill * plan.qty)
        pos = Position(
            symbol=plan.symbol,
            side=plan.side,
            qty=plan.qty,
            remaining_qty=plan.qty,
            entry=fill,
            stop_loss=plan.stop_loss,
            take_profits=[[float(p), float(f), 0] for p, f in plan.take_profits],
            opened_at=time.time(),
            risk_amount=plan.risk_amount,
            leverage=plan.leverage,
            slippage=plan.slippage,
            fees_paid=fee,
            last_price=fill,
            mode="PAPER",
            notes=list(plan.notes),
        )
        log.info("[PAPER] 开仓 %s qty=%.6g @ %.8g 止损 %.8g", plan.symbol, plan.qty, fill, plan.stop_loss)
        return pos

    def _do_close(self, pos: Position, reason: str, price: float, fraction: float) -> BrokerAction:
        """平仓成交。

        平空 = 买入，平多 = 卖出。「变差」的方向因此相反：
          止损（亏损）时成交价往对我不利的方向走；
          止盈（限价单挂在目标价）成交价略好于目标价。
        """
        slip = pos.slippage if getattr(pos, "slippage", 0) else self.cfg.slippage_pct
        is_long = pos.side == "LONG"
        if reason == "TP":
            # 限价止盈按目标价成交，不赠送有利方向滑点
            fill = price
        else:
            # STOP / SQUEEZE / PHASE / TIME 均为市价平仓，成交价向不利方向劣化
            fill = price * (1 - slip) if is_long else price * (1 + slip)

        if fraction >= 1.0:
            qty = pos.remaining_qty
        else:
            # 按原始开仓数量计算分档平仓量，避免 0.5 + 0.5 残留 25%
            base_qty = pos.qty if pos.qty > 0 else pos.remaining_qty
            qty = min(pos.remaining_qty, base_qty * fraction)
        if qty <= 0:
            return BrokerAction(kind="NOOP", symbol=pos.symbol, reason="无仓可平")

        pnl = ((fill - pos.entry) if is_long else (pos.entry - fill)) * qty
        fee = self._fee(fill * qty)
        pnl -= fee
        pos.realized_pnl += pnl
        pos.fees_paid += fee
        pos.remaining_qty -= qty
        pos.last_price = fill

        if pos.remaining_qty <= 1e-12:
            pos.remaining_qty = 0.0
            pos.status = "CLOSED"
            pos.close_reason = reason

        log.info(
            "[PAPER] 平仓 %s %s qty=%.6g @ %.8g pnl=%+.2f 剩余 %.6g",
            pos.symbol, reason, qty, fill, pnl, pos.remaining_qty,
        )
        return BrokerAction(
            kind="CLOSE" if reason == "STOP" else ("TP" if reason == "TP" else reason),
            symbol=pos.symbol,
            reason=reason,
            price=fill,
            pnl=pnl,
            fraction=fraction,
            qty=qty,
        )

    def _do_sync(self, positions: List[Position]) -> List[Position]:
        return positions

    def accrue_funding(self, pos: Position, funding_rate: float, hours: float = 8.0) -> float:
        """资金费：费率为正时多头付给空头，空头收取；多头则付出。

        返回本次结算给账户的金额（正数 = 收到）。
        """
        amount = funding_rate * pos.entry * pos.remaining_qty * (hours / 8.0)
        if pos.side == "LONG":
            amount = -amount
        pos.funding_paid += amount       # 正数 = 收到
        pos.realized_pnl += amount
        return amount


# ==================================================================== Live


class LiveBroker(BaseBroker):
    def __init__(self, ex: BinanceFutures, broker_cfg: BrokerConfig, risk_cfg: RiskConfig):
        super().__init__(ex, broker_cfg, risk_cfg)
        self.mode = "LIVE"

    def _do_open(self, plan: TradePlan) -> Optional[Position]:
        sym = plan.symbol
        is_long = plan.side == "LONG"
        lev = max(1, int(round(plan.leverage)))
        try:
            self.ex.set_margin_type(sym, "ISOLATED")
        except Exception as e:
            log.warning("设置逐仓失败 %s: %s", sym, e)
        try:
            self.ex.set_leverage(sym, lev)
        except Exception as e:
            log.error("设置杠杆失败 %s: %s", sym, e)
            return None

        entry_side = "BUY" if is_long else "SELL"
        exit_side = "SELL" if is_long else "BUY"
        try:
            resp = self.ex.place_order(sym, entry_side, plan.qty, "MARKET")
        except Exception as e:
            log.error("开仓下单失败 %s: %s", sym, e)
            return None

        fill = float(resp.get("avgPrice") or 0.0)
        if fill <= 0:
            fill = plan.entry
            log.warning("%s 成交均价为 0，回退用计划价 %s", sym, fill)
        filled_qty = float(resp.get("executedQty") or plan.qty)

        # 实际成交价偏离过大 -> 止损方向不对，紧急平仓
        risk_dist = (fill - plan.stop_loss) if is_long else (plan.stop_loss - fill)
        if risk_dist <= 0:
            log.error("%s 成交价 %.8g 已越过止损，紧急平仓", sym, fill)
            try:
                self.ex.place_order(sym, exit_side, filled_qty, "MARKET", reduce_only=True)
            except Exception as e:
                log.error("紧急平仓失败 %s: %s", sym, e)
            return None

        order_ids: Dict[str, Any] = {"entry": resp.get("orderId")}

        # 止损单（多头是卖出止损，空头是买入止损）
        try:
            sl = self.ex.place_order(
                sym, exit_side, filled_qty, "STOP_MARKET",
                stop_price=plan.stop_loss, reduce_only=True, working_type="MARK_PRICE",
            )
            order_ids["stop"] = sl.get("orderId")
        except Exception as e:
            log.error("挂止损单失败 %s: %s", sym, e)

        # 分档止盈单
        tps: List[List[float]] = []
        for price, frac in plan.take_profits:
            qty = filled_qty * frac
            if qty <= 0:
                continue
            tps.append([float(price), float(frac), 0])
            try:
                tp = self.ex.place_order(
                    sym, exit_side, qty, "TAKE_PROFIT_MARKET",
                    stop_price=price, reduce_only=True, working_type="MARK_PRICE",
                )
                order_ids[f"tp_{price}"] = tp.get("orderId")
            except Exception as e:
                log.error("挂止盈单失败 %s @ %s: %s", sym, price, e)

        pos = Position(
            symbol=sym,
            side=plan.side,
            qty=filled_qty,
            remaining_qty=filled_qty,
            entry=fill,
            stop_loss=plan.stop_loss,
            take_profits=tps,
            opened_at=time.time(),
            risk_amount=filled_qty * risk_dist,
            leverage=plan.leverage,
            slippage=plan.slippage,
            last_price=fill,
            order_ids=order_ids,
            mode="LIVE",
            notes=list(plan.notes),
        )
        log.info("[LIVE] 开仓 %s %s qty=%.6g @ %.8g 止损 %.8g",
                 sym, plan.side, filled_qty, fill, plan.stop_loss)
        return pos

    def _do_cancel_all(self, symbol: str) -> None:
        try:
            for o in self.ex.open_orders(symbol):
                try:
                    self.ex.cancel_order(symbol, int(o["orderId"]))
                except Exception as e:
                    log.warning("撤单失败 %s %s: %s", symbol, o.get("orderId"), e)
        except Exception as e:
            log.warning("获取挂单失败 %s: %s", symbol, e)

    def _do_update_stop(self, pos: Position, new_stop: float) -> bool:
        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        old_stop_id = pos.order_ids.get("stop")
        if old_stop_id:
            try:
                self.ex.cancel_order(pos.symbol, int(old_stop_id))
            except Exception as e:
                log.warning("撤销旧止损单失败 %s %s: %s", pos.symbol, old_stop_id, e)
        if pos.remaining_qty <= 0:
            return False
        try:
            sl = self.ex.place_order(
                pos.symbol, exit_side, pos.remaining_qty, "STOP_MARKET",
                stop_price=new_stop, reduce_only=True, working_type="MARK_PRICE",
            )
            pos.order_ids["stop"] = sl.get("orderId")
            log.info("[LIVE] 更新止损单 %s -> %.8g (qty=%.6g)", pos.symbol, new_stop, pos.remaining_qty)
            return True
        except Exception as e:
            log.error("重挂新止损单失败 %s @ %s: %s", pos.symbol, new_stop, e)
            return False

    def _do_close(self, pos: Position, reason: str, price: float, fraction: float) -> BrokerAction:
        qty = min(pos.remaining_qty, pos.remaining_qty * fraction)
        if qty <= 0:
            return BrokerAction(kind="NOOP", symbol=pos.symbol, reason="无仓可平")
        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        # 先撤所有保护单，避免平仓后被 TP/SL 反手
        self._do_cancel_all(pos.symbol)
        try:
            resp = self.ex.place_order(pos.symbol, exit_side, qty, "MARKET", reduce_only=True)
            fill = float(resp.get("avgPrice") or 0.0) or price
        except Exception as e:
            log.error("平仓失败 %s: %s", pos.symbol, e)
            return BrokerAction(kind="ERROR", symbol=pos.symbol, reason=str(e))

        pnl = ((fill - pos.entry) if pos.side == "LONG" else (pos.entry - fill)) * qty
        pos.realized_pnl += pnl
        pos.remaining_qty -= qty
        pos.last_price = fill
        if pos.remaining_qty <= 1e-12:
            pos.remaining_qty = 0.0
            pos.status = "CLOSED"
            pos.close_reason = reason
        log.info("[LIVE] 平仓 %s %s qty=%.6g @ %.8g pnl=%+.2f", pos.symbol, reason, qty, fill, pnl)
        return BrokerAction(
            kind="CLOSE" if reason == "STOP" else ("TP" if reason == "TP" else reason),
            symbol=pos.symbol,
            reason=reason,
            price=fill,
            pnl=pnl,
            fraction=fraction,
            qty=qty,
        )

    def _do_sync(self, positions: List[Position]) -> List[Position]:
        """以交易所为唯一事实来源，同步实际持仓。"""
        try:
            raw = self.ex.positions()
        except Exception as e:
            log.error("同步持仓失败: %s", e)
            return positions
        by_symbol = {p["symbol"]: p for p in raw}
        for pos in positions:
            if pos.status != "OPEN":
                continue
            r = by_symbol.get(pos.symbol)
            if not r:
                continue
            amt = float(r.get("positionAmt", 0) or 0)
            if abs(amt) < 1e-12:
                pos.remaining_qty = 0.0
                pos.status = "CLOSED"
                pos.close_reason = "交易所已平"
                continue
            pos.remaining_qty = abs(amt)
            entry = float(r.get("entryPrice", 0) or 0)
            if entry > 0:
                pos.entry = entry
            up = float(r.get("unRealizedProfit", 0) or 0)
            q = max(pos.remaining_qty, 1e-12)
            if pos.side == "SHORT":
                pos.last_price = pos.entry - up / q
            else:
                pos.last_price = pos.entry + up / q
        return positions


def make_broker(
    ex: BinanceFutures, broker_cfg: BrokerConfig, risk_cfg: RiskConfig, live: bool
) -> BaseBroker:
    return LiveBroker(ex, broker_cfg, risk_cfg) if live else PaperBroker(ex, broker_cfg, risk_cfg)
