#!/usr/bin/env python
"""自检脚本：在纸面模式下强制触发暴跌信号，跑通
「事件检测 -> 选币 -> 建仓 -> 持仓管理 -> 阶段切换 -> 平仓」全链路。

不碰真实下单接口，只用公开行情。用于验证程序逻辑是否连通。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from memeshort.config import Config
from memeshort.crash import VolumeClimax
from memeshort.engine import Engine
from memeshort.utils import setup_logging


def main() -> int:
    setup_logging("logs", "INFO", True)
    cfg = Config.load("config.yaml")
    cfg.risk.account_equity = 10_000.0

    eng = Engine(cfg, live=False)

    # ---- 1. 选币（做空方向）----
    print("\n=== 1. 选币（SHORT 方向）===\n")
    cands = eng.screener.screen(side="SHORT")
    for c in cands:
        print(f"  {c.summary()}")
        for r in c.reasons:
            print(f"      ↳ {r}")
        for w in c.warnings:
            print(f"      ⚠ {w}")

    # ---- 2. 交易计划（多空各一份）----
    print("\n=== 2. 交易计划 ===\n")
    for side in ("SHORT", "LONG"):
        plans = eng._build_plans(cands, side=side)
        print(f"  --- {side} ---")
        if not plans:
            print("    无计划通过风控")
        for p in plans:
            print(f"    {p.summary()}")
            for n in p.notes:
                print(f"        · {n}")

    # ---- 3. 阶段切换：建空头腿 ----
    print("\n=== 3. 阶段：做空腿建仓 ===\n")
    shorts = eng._open_positions(cands, side="SHORT", phase="short", event_day="2026-11-20")
    print(f"  建仓 {len(shorts)} 个空头")
    for pos in shorts:
        print(f"    {pos.symbol} {pos.side} qty={pos.remaining_qty:.6g} entry={pos.entry:.8g} "
              f"stop={pos.stop_loss:.8g} 风险=${pos.risk_amount:.2f}")

    # ---- 4. 空头持仓管理（每个场景用独立持仓，避免前一个场景已平仓影响后一个）----
    print("\n=== 4. 空头持仓管理 ===\n")
    eng._close_by_side("SHORT", "自检清理")
    for label, price_fn, keep_open in [
        ("越过止损", lambda p: p.stop_loss * 1.01, False),
        ("-8%  (未触发)", lambda p: p.entry * 0.92, False),
        ("-16% (应触发止盈1)", lambda p: p.entry * 0.84, True),
        ("+3%  (保本止损应被扫出)", lambda p: p.entry * 1.03, False),
    ]:
        # 前三个场景各自新开仓；最后一个复用止盈场景留下的持仓（此时止损已移到成本价）
        still = [p for p in eng.positions if p.status == "OPEN" and p.side == "SHORT"]
        if label.startswith("+3%"):
            if not still:
                print(f"  {label:<34} （无持仓可复用）")
                continue
            pos = still[0]
        else:
            fresh = eng._open_positions(cands, side="SHORT", phase="short", event_day="2026-11-20")
            if not fresh:
                print("  （无法建仓，跳过）")
                break
            pos = fresh[0]
        price = price_fn(pos)
        actions = eng.broker.manage(pos, price, funding=0.0)
        for a in actions:
            eng._handle_action(pos, a)
        extra = f"  [止损现为 {pos.stop_loss:.8g}]" if label.startswith("+3%") else ""
        print(f"  {label:<34} 价格 {price:.8g} -> 动作 {[a.kind for a in actions] or '无'}{extra}")
        if not keep_open:
            eng._close_by_side("SHORT", "自检清理")
    eng._close_by_side("SHORT", "自检清理")

    # ---- 5. 阶段切换：平空头、开多头 ----
    print("\n=== 5. 阶段切换：转入反弹腿 ===\n")
    eng._close_by_side("SHORT", "阶段切换：转入反弹腿")
    longs = eng._open_positions(cands, side="LONG", phase="bounce", event_day="2026-11-20")
    print(f"  建仓 {len(longs)} 个多头")
    for pos in longs:
        print(f"    {pos.symbol} {pos.side} qty={pos.remaining_qty:.6g} entry={pos.entry:.8g} "
              f"stop={pos.stop_loss:.8g} 风险=${pos.risk_amount:.2f}")

    # ---- 6. 多头腿方向检查 ----
    print("\n=== 6. 多头腿方向检查 ===\n")
    long_pos = [p for p in eng.positions if p.side == "LONG" and p.status == "OPEN"]
    if long_pos:
        pos = long_pos[0]
        actions = eng.broker.manage(pos, pos.entry * 1.02, funding=0.0)
        print(f"  多头涨 2%        -> 动作 {[a.kind for a in actions] or '无'}（止盈更上方，未到）")
        actions = eng.broker.manage(pos, pos.take_profits[0][0] * 1.001, funding=0.0)
        print(f"  多头涨到第一档止盈 -> 动作 {[a.kind for a in actions] or '无'}（应触发 TP）")
        pos.status = "OPEN"   # 复位以便单独测止损
        pos.take_profits[0][2] = 1
        actions = eng.broker.manage(pos, pos.stop_loss * 0.99, funding=0.0)
        print(f"  多头跌破止损      -> 动作 {[a.kind for a in actions] or '无'}（应触发 STOP）")
    else:
        print("  （无多头持仓可测）")

    # ---- 7. 轧空路径（只对空头腿生效）----
    print("\n=== 7. 轧空路径（资金费率深度负值）===\n")
    eng._close_by_side("LONG", "自检清理")
    eng._close_by_side("SHORT", "自检清理")
    shorts2 = eng._open_positions(cands, side="SHORT", phase="short", event_day="2026-11-20")
    if shorts2:
        pos = shorts2[0]
        actions = eng.broker.manage(pos, pos.entry * 0.99, funding=-0.002)
        for a in actions:
            eng._handle_action(pos, a)
        print(f"  空头腿 资金费率 -0.2% -> 动作 {[a.kind for a in actions]}（应触发 SQUEEZE 全平）")
        eng._close_by_side("SHORT", "自检清理")

    longs2 = eng._open_positions(cands, side="LONG", phase="bounce", event_day="2026-11-20")
    if longs2:
        pos = longs2[0]
        actions = eng.broker.manage(pos, pos.entry * 0.99, funding=-0.002)
        print(f"  多头腿 资金费率 -0.2% -> 动作 {[a.kind for a in actions] or '无'}（应为空：轧空信号只平空头）")
        eng._close_by_side("LONG", "自检清理")

    # ---- 8. 事件阶段推进 ----
    print("\n=== 8. 事件阶段推进 ===\n")
    base = date(2026, 11, 20)
    for offset, expect in [(1, "bounce"), (2, "bounce"), (3, "short"), (7, "short"), (8, "wait")]:
        climax = VolumeClimax(
            day=str(base - timedelta(days=offset)),
            days_ago=offset, volume_ratio=2.18, drop_pct=-0.049, close=56000,
        )
        phase, off = eng.detector._phase_for(climax)
        mark = "✓" if phase == expect else "✗"
        print(f"  T0+{offset} -> 阶段 {phase:<7} {mark} (期望 {expect})")

    # ---- 9. 引擎级事件闭环：信号 -> 阶段 -> 建仓 -> 切换 -> 幂等 ----
    print("\n=== 9. 引擎级事件闭环 ===\n")
    from memeshort.crash import CrashEval, CrashSignal

    def make_signal(offset, phase):
        return CrashSignal(
            triggered=True, mode="event",
            btc=CrashEval(symbol="BTC", score=64.0, price=56000.0),
            eth=CrashEval(symbol="ETH", score=40.0, price=1560.0),
            climax=VolumeClimax(day="2026-11-20", days_ago=offset,
                                volume_ratio=2.18, drop_pct=-0.049, close=56000),
            phase=phase, offset=offset,
        )

    # 9a. 漂移保护：阈值调高 -> 允许做反弹腿
    eng.positions = []
    eng.acct["event_state"] = {}
    saved_drift = eng.cfg.bounce.max_entry_drift
    eng.cfg.bounce.max_entry_drift = 0.99      # 放开漂移限制
    eng._handle_event(make_signal(1, "bounce"))
    open_pos = [p for p in eng.positions if p.status == "OPEN"]
    sides = {p.side for p in open_pos}
    print(f"  放开漂移限制 -> T0+1 建仓 {len(open_pos)} 个 {sides or '空'} "
          f"{'✓' if sides == {'LONG'} else '✗'} (期望 LONG)")
    st = eng._event_state()
    print(f"      事件状态 key={st.get('key')} phase={st.get('phase')}")
    before = len([p for p in eng.positions if p.status == "OPEN"])
    eng._handle_event(make_signal(1, "bounce"))
    after = len([p for p in eng.positions if p.status == "OPEN"])
    print(f"      幂等：重复处理后持仓 {before} -> {after} {'✓' if before == after else '✗'}")

    # 9b. 阶段推进到做空腿：平掉多头、开空头
    eng._handle_event(make_signal(3, "short"))
    open_pos = [p for p in eng.positions if p.status == "OPEN"]
    sides = {p.side for p in open_pos}
    print(f"  T0+3 阶段切换 -> 持仓 {len(open_pos)} 个 {sides or '空'} "
          f"{'✓' if sides == {'SHORT'} else '✗'} (期望 SHORT)")

    # 9c. 漂移保护生效：阈值调低 -> 跳过反弹腿
    eng.positions = []
    eng.acct["event_state"] = {}
    eng.cfg.bounce.max_entry_drift = 0.0       # 任何涨幅都算走完
    eng._handle_event(make_signal(1, "bounce"))
    open_pos = [p for p in eng.positions if p.status == "OPEN"]
    st = eng._event_state()
    print(f"  触发漂移保护 -> 建仓 {len(open_pos)} 个 "
          f"{'✓' if not open_pos else '✗'} (期望跳过反弹腿)")
    print(f"      事件状态 skipped_bounce={st.get('skipped_bounce')} phase={st.get('phase')}")
    eng.cfg.bounce.max_entry_drift = saved_drift

    eng._persist()
    print("\n=== 10. 最终状态 ===\n")
    print(eng.status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
