"""主引擎：把暴跌检测、选币、风控、执行、监控串成闭环。

每个轮询周期做四件事：
  1. 同步并管理已有持仓（止损 / 止盈 / 轧空 / 保本移动止损）
  2. 结算资金费（纸面模式）
  3. 检测大盘暴跌信号（降频，避免频繁拉数据）
  4. 若信号成立且未触发冷却/熔断，则选币、建仓
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .broker import BaseBroker, BrokerAction, PaperBroker, make_broker
from .config import Config
from .crash import CrashDetector, CrashEval, CrashSignal
from .exchange import BinanceFutures
from .notify import Notifier
from .risk import TradePlan, build_plan
from .screener import MemeCandidate, MemeScreener
from .state import Position, StateStore
from .utils import fmt_qty, fmt_usd

log = logging.getLogger("meme.engine")

# 暴跌事件冷却：同一轮暴跌不重复开仓（小时）
EVENT_COOLDOWN_HOURS = 12.0
# 暴跌检测频率（每 N 个轮询周期做一次）
DETECT_EVERY_N = 3


@dataclass
class ScanReport:
    triggered: bool
    btc: CrashEval
    eth: CrashEval
    candidates: List[MemeCandidate]
    plans: List[TradePlan]
    signal: Optional[CrashSignal] = None


class Engine:
    def __init__(self, cfg: Config, live: bool = False):
        self.cfg = cfg
        self.live = live
        self.ex = BinanceFutures(
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            proxy=cfg.proxy,
            testnet=cfg.testnet,
        )
        self.store = StateStore("state")
        self.acct: Dict[str, Any] = self.store.load_account()
        self.acct = self.store.rollover_day(self.acct)
        self.positions: List[Position] = self.store.load_positions()

        self.broker: BaseBroker = make_broker(
            self.ex, cfg.broker, cfg.risk, live
        )
        self.notifier = Notifier(cfg.notify, logger=log)
        self.detector = CrashDetector(self.ex, cfg.crash, cfg.bounce)
        self.screener = MemeScreener(self.ex, cfg.screener)

        if not self.acct.get("initial_equity"):
            self.acct["initial_equity"] = cfg.risk.account_equity
            self.acct["equity"] = cfg.risk.account_equity

        self._tick = 0
        self._next_funding_ts = self._next_funding_time()

    # ---------------------------------------------------- 资金费结算

    @staticmethod
    def _next_funding_time(now: Optional[float] = None) -> float:
        """币安资金费在 UTC 00:00 / 08:00 / 16:00 结算。"""
        now = now or time.time()
        base = (now // 28800) * 28800
        return base + 28800

    def _settle_funding(self) -> None:
        now = time.time()
        if now < self._next_funding_ts:
            return
        # 补齐可能跨过的多个结算点
        while now >= self._next_funding_ts:
            total = 0.0
            for pos in self.positions:
                if pos.status != "OPEN" or pos.remaining_qty <= 0:
                    continue
                try:
                    rate = float(self.ex.premium_index(pos.symbol)["lastFundingRate"])
                except Exception:
                    rate = 0.0
                amount = rate * pos.entry * pos.remaining_qty
                if isinstance(self.broker, PaperBroker):
                    self.broker.accrue_funding(pos, rate, hours=8.0)
                else:
                    pos.funding_paid += amount
                    pos.realized_pnl += amount
                total += amount
            if abs(total) > 0.01:
                log.info("资金费结算：合计 %+.2f USDT", total)
                self.acct["realized_pnl_today"] = (
                    self.acct.get("realized_pnl_today", 0.0) + total
                )
            self._next_funding_ts += 28800

    # ---------------------------------------------------- 风控

    @property
    def equity(self) -> float:
        """账户权益。实盘读交易所钱包余额；纸面按 初始权益 + 已实现 + 浮动 计算。"""
        if self.live:
            try:
                acc = self.ex.account()
                return float(acc.get("totalWalletBalance", 0.0))
            except Exception:
                pass
        base = float(self.acct.get("initial_equity") or self.cfg.risk.account_equity)
        realized = float(self.acct.get("total_realized_pnl", 0.0))
        unreal = sum(p.unrealized_pnl for p in self.positions if p.status == "OPEN")
        return base + realized + unreal

    def _daily_halt(self) -> Optional[str]:
        eq = self.equity
        loss = float(self.acct.get("realized_pnl_today", 0.0))
        limit = eq * self.cfg.risk.daily_loss_limit_pct
        if limit > 0 and loss <= -limit:
            return f"当日已实现亏损 ${loss:.2f}，超过熔断线 ${limit:.2f}（{self.cfg.risk.daily_loss_limit_pct:.1%}）"
        return None

    def _event_cooldown_active(self) -> bool:
        last = float(self.acct.get("last_event_ts", 0) or 0)
        if last <= 0:
            return False
        return (time.time() - last) < EVENT_COOLDOWN_HOURS * 3600

    # ---------------------------------------------------- 持仓管理

    def _manage_positions(self) -> None:
        open_positions = [p for p in self.positions if p.status == "OPEN"]
        if not open_positions:
            return
        self.positions = self.broker.sync(self.positions)
        for pos in self.positions:
            if pos.status != "OPEN":
                continue
            try:
                prem = self.ex.premium_index(pos.symbol)
                price = float(prem["markPrice"])
                funding = float(prem.get("lastFundingRate", 0.0))
            except Exception as e:
                log.warning("取价失败 %s: %s", pos.symbol, e)
                continue
            actions = self.broker.manage(pos, price, funding)
            for a in actions:
                self._handle_action(pos, a)

    def _handle_action(self, pos: Position, a: BrokerAction) -> None:
        if a.kind == "BREAKEVEN":
            self.notifier.send(
                f"移动止损 {a.symbol}",
                a.reason,
                level="INFO",
            )
            return
        if a.kind in ("CLOSE", "STOP", "TP", "SQUEEZE", "PHASE", "TIME", "ERROR", "NOOP"):
            if a.kind == "NOOP":
                return
            if a.kind == "ERROR":
                self.notifier.send(f"⚠ 平仓异常 {a.symbol}", a.reason, level="ERROR")
                return
            self.acct["realized_pnl_today"] = (
                self.acct.get("realized_pnl_today", 0.0) + a.pnl
            )
            self.acct["total_realized_pnl"] = (
                self.acct.get("total_realized_pnl", 0.0) + a.pnl
            )
            self.acct["trade_count"] = self.acct.get("trade_count", 0) + 1
            if a.pnl > 0:
                self.acct["win_count"] = self.acct.get("win_count", 0) + 1
            base_qty = a.qty or (pos.remaining_qty or pos.qty)
            pct = a.pnl / max(1e-9, pos.entry * base_qty)
            label = {"PHASE": "阶段切换", "TIME": "事件窗口结束"}.get(a.reason, a.reason)
            # 阶段切换/时间平仓是计划内的操作，小幅滑点亏损不该报 ERROR
            lvl = "INFO" if a.reason in ("PHASE", "TIME") else None
            self.notifier.position_closed(pos.symbol, label, a.pnl, pct, level=lvl)
            if a.reason == "SQUEEZE":
                self.notifier.squeeze_warning(pos.symbol, float(a.extra.get("funding", 0.0)))

    # ---------------------------------------------------- 选币与建仓

    def scan(self, with_plans: bool = True) -> ScanReport:
        """只扫描不交易，输出当前状态。"""
        signal = self.detector.detect()
        btc = signal.btc or CrashEval(symbol="BTC")
        eth = signal.eth or CrashEval(symbol="ETH")
        side = "LONG" if signal.phase == "bounce" else "SHORT"
        candidates = self.screener.screen(side=side) if signal.triggered else []
        plans: List[TradePlan] = []
        if with_plans and candidates:
            plans = self._build_plans(candidates, side=side)
        return ScanReport(signal.triggered, btc, eth, candidates, plans, signal=signal)

    def _build_plans(
        self, candidates: List[MemeCandidate], side: str = "SHORT"
    ) -> List[TradePlan]:
        eq = self.equity
        open_pos = [p for p in self.positions if p.status == "OPEN"]
        existing_notional = sum(p.notional for p in open_pos)
        perps = self.ex.perpetual_symbols(self.cfg.screener.quote_asset)
        plans: List[TradePlan] = []
        is_long = side.upper() == "LONG"
        risk_pct = self.cfg.bounce.long_risk_per_trade if is_long else None
        lev_cap = self.cfg.bounce.long_max_leverage if is_long else None
        for c in candidates:
            if len(plans) + len(open_pos) >= self.cfg.risk.max_positions:
                break
            plan = build_plan(
                c, eq, self.cfg.risk, self.ex,
                perp_meta=perps.get(c.symbol),
                existing_notional=existing_notional + sum(p.notional for p in plans),
                existing_positions=len(open_pos) + len(plans),
                broker_cfg=self.cfg.broker,
                side=side,
                risk_per_trade=risk_pct,
                max_leverage=lev_cap,
            )
            if plan:
                plans.append(plan)
        return plans

    def _open_positions(
        self, candidates: List[MemeCandidate], side: str = "SHORT",
        phase: str = "", event_day: str = "",
    ) -> List[Position]:
        opened: List[Position] = []
        plans = self._build_plans(candidates, side=side)
        if not plans:
            log.info("没有通过风控的交易计划")
            return opened
        for plan in plans:
            pos = self.broker.open_position(plan)
            if pos:
                pos.phase = phase
                pos.event_day = event_day
                opened.append(pos)
                self.notifier.order_opened(plan, plan.symbol)
        if opened:
            self.acct["last_event_ts"] = time.time()
            self.positions.extend(opened)
        return opened

    # ---------------------------------------------------- 事件与阶段

    def _event_state(self) -> Dict[str, Any]:
        return self.acct.setdefault("event_state", {})

    def _close_by_side(self, side: str, reason: str) -> None:
        """平掉某一方向的持仓（阶段切换时用）。"""
        for pos in [p for p in self.positions if p.status == "OPEN" and p.side == side]:
            try:
                price = float(self.ex.premium_index(pos.symbol)["markPrice"])
            except Exception:
                price = pos.last_price or pos.entry
            act = self.broker.close(pos, "PHASE", price, 1.0)
            if act.kind != "ERROR":
                self._handle_action(pos, act)

    def _event_drift(self, signal: CrashSignal) -> Optional[float]:
        """当前价格相对事件收盘价（T0 收盘）的涨幅。

        用来判断「反弹是不是已经走完了」—— 回测显示 T0->T0+1 涨幅超过 2% 时，
        再进场做反弹腿的胜率是 0%。
        """
        if signal.climax is None:
            return None
        try:
            df = self.ex.klines(self.cfg.crash.btc_symbol, self.cfg.crash.tf_daily, limit=10)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        try:
            day = datetime.strptime(str(signal.climax.day), "%Y-%m-%d").date()
        except ValueError:
            return None
        days = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
        mask = days == day
        if not mask.any():
            return None
        t0_close = float(df["close"][mask].iloc[-1])
        try:
            now = float(self.ex.premium_index(self.cfg.crash.btc_symbol)["markPrice"])
        except Exception:
            return None
        if t0_close <= 0:
            return None
        return now / t0_close - 1.0

    def _handle_event(self, signal: CrashSignal) -> None:
        """按当前阶段执行：反弹腿做多，做空腿做空。"""
        day = signal.event_day or ""
        phase = signal.phase
        st = self._event_state()
        key = f"{day}:{phase}"
        if st.get("key") == key:
            return  # 同一个 (事件, 阶段) 已处理过

        side = "LONG" if phase == "bounce" else "SHORT"

        # 「反弹已经走完就别追」：进场价相对事件收盘价的涨幅超过阈值就跳过反弹腿
        if phase == "bounce" and self.cfg.bounce.enabled:
            drift = self._event_drift(signal)
            if drift is not None and drift > self.cfg.bounce.max_entry_drift:
                log.info(
                    "价格已较事件收盘价上涨 %.2f%%，超过 %.2f%% 阈值 -> 跳过反弹腿，等待做空腿",
                    drift * 100, self.cfg.bounce.max_entry_drift * 100,
                )
                self.notifier.send(
                    f"跳过反弹腿 {day}",
                    f"价格已较 T0 收盘价上涨 {drift * 100:.2f}%，超过 "
                    f"{self.cfg.bounce.max_entry_drift * 100:.2f}% 阈值；"
                    f"反弹收益已被走完，直接等 T0+{self.cfg.bounce.short_start_offset} 做空腿",
                    level="INFO",
                )
                st["key"] = key          # 用同一个 key 保持幂等，避免每轮重复通知
                st["day"] = day
                st["phase"] = "short"
                st["offset"] = signal.offset
                st["skipped_bounce"] = True
                return

        log.info("事件 %s 进入阶段 %s (T0%+.0f 天)，方向 %s", day, phase, signal.offset, side)
        # 通知只在阶段真正切换时发一次（幂等键保证不重复）
        self.notifier.crash_detected(
            signal.btc.score if signal.btc else 0.0,
            signal.eth.score if signal.eth else 0.0,
            signal.reasons + [
                f"阶段：{phase}（T0{signal.offset:+d} 天），方向 {side}"
            ],
        )

        # 阶段切换：先平掉相反方向的持仓
        if phase == "bounce":
            self._close_by_side("SHORT", "阶段切换：转入反弹腿")
        else:
            self._close_by_side("LONG", "阶段切换：转入做空腿")

        if self._daily_halt():
            log.warning("触发日内熔断，本阶段不开新仓")
        else:
            cands = self.screener.screen(side=side)
            for c in cands:
                log.info("候选[%s] %s", side, c.summary())
            self._open_positions(cands, side=side, phase=phase, event_day=day)

        st["key"] = key
        st["day"] = day
        st["phase"] = phase
        st["offset"] = signal.offset

    def _expire_positions(self, signal: CrashSignal) -> None:
        """事件窗口结束后，按时间平掉剩余仓位。

        用当前日期与事件日直接算偏移量，不依赖「当期还能不能检测到 climax」——
        事件过了生命周期窗口后 climax 就检不出来了，但仓位仍需要按时平掉。
        """
        st = self._event_state()
        if not st.get("day"):
            return
        try:
            today = datetime.now(timezone.utc).date()
            offset = (today - date.fromisoformat(str(st["day"]))).days
        except (ValueError, TypeError):
            return
        if offset <= self.cfg.bounce.short_end_offset:
            return
        open_pos = [p for p in self.positions if p.status == "OPEN"]
        if not open_pos:
            st["key"] = None
            return
        log.info("事件窗口结束（已过 T0+%d 天），按时间平仓", offset)
        for pos in open_pos:
            try:
                price = float(self.ex.premium_index(pos.symbol)["markPrice"])
            except Exception:
                price = pos.last_price or pos.entry
            act = self.broker.close(pos, "TIME", price, 1.0)
            if act.kind != "ERROR":
                self._handle_action(pos, act)
        st["key"] = None

    # ---------------------------------------------------- 主循环

    def run_once(self) -> None:
        self._tick += 1
        self._settle_funding()
        self._manage_positions()

        # 熔断检查
        halt = self._daily_halt()
        if halt and not self.acct.get("halted"):
            self.acct["halted"] = True
            self.notifier.risk_halt(halt)
        if self.acct.get("halted"):
            self._persist()
            return

        # 暴跌检测（降频）
        if self._tick % DETECT_EVERY_N == 1 or self._tick == 1:
            signal = self.detector.detect()
            btc = signal.btc or CrashEval(symbol="BTC")
            eth = signal.eth or CrashEval(symbol="ETH")
            log.info(
                "信号检测：%s | BTC %.0f 分 / ETH %.0f 分 | 阶段 %s (T0%+.0f 天)",
                signal.summary(), btc.score, eth.score, signal.phase, signal.offset,
            )
            if signal.triggered:
                if signal.phase != "wait":
                    self._handle_event(signal)
                else:
                    log.info("事件已超出操作窗口（阶段=wait），不操作")
            else:
                log.info("未触发信号")
            # 无论信号是否仍触发，都要检查是否有到期的仓位需要按时间平掉
            self._expire_positions(signal)

        self._persist()

    def _persist(self) -> None:
        self.acct["equity"] = self.equity
        self.store.save_positions(self.positions)
        self.store.save_account(self.acct)

    def run_forever(self) -> None:
        log.info(
            "启动 | 模式 %s | 权益 $%.2f | 轮询 %ds",
            "实盘" if self.live else "纸面", self.equity, self.cfg.broker.poll_seconds,
        )
        if self.live:
            self.notifier.send(
                "🚀 实盘模式启动",
                f"权益 ${self.equity:.2f}，最大持仓 {self.cfg.risk.max_positions}，"
                f"杠杆上限 {self.cfg.risk.max_leverage}x",
                level="WARNING",
            )
        try:
            while True:
                try:
                    self.run_once()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log.exception("轮询异常: %s", e)
                    self.notifier.send("⚠ 主循环异常", str(e), level="ERROR")
                time.sleep(self.cfg.broker.poll_seconds)
        except KeyboardInterrupt:
            log.info("收到中断，保存状态退出")
            self.notifier.send("🛑 程序停止", "已保存持仓状态", level="WARN")
            self._persist()

    # ---------------------------------------------------- 报表

    def status(self) -> str:
        eq = self.equity
        open_pos = [p for p in self.positions if p.status == "OPEN"]
        unreal = sum(p.unrealized_pnl for p in open_pos)
        lines = [
            f"模式      : {'实盘' if self.live else '纸面'}",
            f"权益      : ${eq:.2f}",
            f"今日已实现: ${self.acct.get('realized_pnl_today', 0.0):+.2f}",
            f"累计已实现: ${self.acct.get('total_realized_pnl', 0.0):+.2f}",
            f"浮动盈亏  : ${unreal:+.2f}",
            f"持仓数    : {len(open_pos)}/{self.cfg.risk.max_positions}",
            f"总名义敞口: {fmt_usd(sum(p.notional for p in open_pos))}",
            "",
        ]
        if not open_pos:
            lines.append("（无持仓）")
        for p in open_pos:
            tps = " / ".join(f"{t[0]:.8g}{'✓' if t[2] else ''}" for t in p.take_profits)
            phase_cn = {"bounce": "反弹腿", "short": "做空腿"}.get(p.phase, "")
            lines += [
                f"── {p.symbol} {p.side} {fmt_qty(p.remaining_qty)} @ {p.entry:.8g}"
                + (f"  [{phase_cn} {p.event_day}]" if p.phase else ""),
                f"   现价 {p.last_price:.8g} | 浮动 {p.unrealized_pnl:+.2f} ({p.unrealized_pnl_pct:+.2%})",
                f"   止损 {p.stop_loss:.8g}{' (保本)' if p.stop_moved_to_breakeven else ''} | 止盈 {tps}",
                f"   杠杆 {p.leverage:.2f}x | 模式 {p.mode}",
            ]
        st = self.acct.get("event_state") or {}
        if st.get("day"):
            lines += ["", f"当前事件：{st.get('day')} | 阶段 {st.get('phase')} | T0{st.get('offset', 0):+d} 天"]
        return "\n".join(lines)
