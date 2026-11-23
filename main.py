#!/usr/bin/env python
"""Meme 币做空策略 —— 命令行入口。

用法：
    python main.py scan                 只扫描：当前暴跌状态 + 候选币 + 交易计划预览
    python main.py run                  纸面交易主循环（默认）
    python main.py run --live           实盘（需 API Key 与二次确认）
    python main.py status               查看持仓与盈亏
    python main.py backtest --days 730  历史回测，验证策略有效性
    python main.py reset                清空纸面持仓与账户状态
"""

from __future__ import annotations

import argparse
import os
import sys

from memeshort.config import Config
from memeshort.engine import Engine
from memeshort.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meme-short",
        description="暴跌发现 -> 做空高 Beta Meme 币策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("command", choices=["scan", "run", "status", "backtest", "reset"])
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    p.add_argument("--live", action="store_true", help="实盘模式（默认纸面）")
    p.add_argument("--yes", "-y", action="store_true", help="跳过实盘二次确认")
    p.add_argument("--days", type=int, default=730, help="回测天数")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--explain", action="store_true",
                   help="scan 时同时列出被过滤掉的币及其原因")
    p.add_argument("--sweep", action="store_true",
                   help="backtest 时额外做阈值敏感性扫描")
    return p


def cmd_scan(cfg: Config, args) -> int:
    eng = Engine(cfg, live=False)
    print("\n" + "=" * 72)
    print("  暴跌信号扫描")
    print("=" * 72)
    rep = eng.scan(with_plans=True)
    sig = rep.signal
    print(f"\n  状态: {'🔴 已触发暴跌信号' if rep.triggered else '⚪ 未触发暴跌信号'}")
    if sig:
        phase_cn = {"wait": "等待（不在操作窗口）", "bounce": "反弹腿（做多）",
                    "short": "做空腿（做空）"}[sig.phase]
        print(f"  模式: {sig.mode}    阶段: {phase_cn}"
              + (f"    T0 {sig.event_day} ({sig.offset:+d} 天)" if sig.event_day else ""))
    if sig and sig.climax:
        c = sig.climax
        print(f"  放量杀跌事件: {c.day}  量比 {c.volume_ratio:.2f}x  跌 {c.drop_pct:.1%}"
              f"  ({c.days_ago} 天前)")
    print(f"  BTC  {rep.btc.score:5.1f} 分   价格 ${rep.btc.price:,.2f}")
    for r in rep.btc.reasons:
        print(f"      · {r}")
    print(f"  ETH  {rep.eth.score:5.1f} 分   价格 ${rep.eth.price:,.2f}")
    for r in rep.eth.reasons:
        print(f"      · {r}")
    if sig:
        for r in sig.reasons:
            print(f"      · {r}")

    if not rep.triggered:
        print("\n  未触发，不进入选币环节。（可放宽 config.yaml 的 crash 参数）")
        if args.explain:
            print_filter_log(eng)
        return 0
    if sig and sig.phase == "wait":
        print("\n  事件已超出操作窗口（wait），本轮不交易。")
        if args.explain:
            print_filter_log(eng)
        return 0

    side = "LONG" if sig and sig.phase == "bounce" else "SHORT"
    print(f"\n  候选 Meme 币 {len(rep.candidates)} 只（方向 {side}）：")
    for c in rep.candidates:
        print(f"    {c.summary()}")
        for r in c.reasons:
            print(f"        ↳ {r}")
        for w in c.warnings:
            print(f"        ⚠ {w}")

    if rep.plans:
        print(f"\n  交易计划（权益 ${eng.equity:,.2f}，杠杆上限 {cfg.risk.max_leverage}x）：")
        for p in rep.plans:
            print(f"    {p.summary()}")
            for n in p.notes:
                print(f"        · {n}")
    else:
        print("\n  没有标的通过风控检查。")
    if args.explain:
        print_filter_log(eng)
    print()
    return 0


def _with_threshold(crash_cfg, threshold: float):
    """复制一份配置并覆盖 BTC 触发分（其余参数保持不变）。"""
    import copy

    c = copy.deepcopy(crash_cfg)
    c.btc_min_score = float(threshold)
    return c


def print_filter_log(eng: Engine) -> None:
    """列出被硬过滤掉的币与原因 —— 调参时最有用的一张表。"""
    log_map = getattr(eng.screener, "filter_log", {}) or {}
    if not log_map:
        return
    print(f"\n  被过滤掉的币（{len(log_map)} 个）：")
    for sym, reason in sorted(log_map.items(), key=lambda kv: kv[1]):
        print(f"    {sym:<18} {reason}")


def cmd_run(cfg: Config, args) -> int:
    if args.live:
        if not cfg.api_key or not cfg.api_secret:
            print("\n✖ 实盘模式需要设置环境变量 MEME_API_KEY / MEME_API_SECRET\n", file=sys.stderr)
            return 2
        print("\n" + "!" * 72)
        print("  即将进入【实盘】模式，会真实下单并产生资金风险。")
        print("  请确认该 API Key 仅具备交易权限、无提现权限。")
        print("!" * 72)
        if not args.yes:
            ans = input("  输入 YES 确认启动实盘: ").strip()
            if ans != "YES":
                print("  已取消。")
                return 1
    eng = Engine(cfg, live=args.live)
    eng.run_forever()
    return 0


def cmd_status(cfg: Config, args) -> int:
    eng = Engine(cfg, live=False)
    print("\n" + eng.status() + "\n")
    return 0


def cmd_backtest(cfg: Config, args) -> int:
    from memeshort.backtest import (
        analyze,
        backtest_phase_strategy,
        detect_climax_events,
        load_backtest_data,
        sweep_thresholds,
    )
    from memeshort.exchange import BinanceFutures

    ex = BinanceFutures(proxy=cfg.proxy, testnet=cfg.testnet)
    data = load_backtest_data(ex, args.days)
    res = analyze(data, cfg.crash)
    print("\n" + res.report() + "\n")

    # ---- 事件型（放量杀跌）信号 ----
    climax = detect_climax_events(data["btc_daily"], cfg.crash)
    print("=" * 74)
    print(f"  事件型信号（放量杀跌：量比>={cfg.crash.climax_volume_ratio}x 且 "
          f"跌幅<={cfg.crash.climax_drop_pct:.0%}）")
    print("=" * 74)
    print(f"  事件数 {len(climax)}：{', '.join(climax[-14:])}")
    ph = backtest_phase_strategy(data, cfg.crash, cfg.bounce, climax)
    print()
    print(f"  {'持有期':<8}{'只做空收益':>13}{'胜率':>8}{'基准':>11}{'样本':>7}")
    print("  " + "-" * 48)
    for h in ph["horizons"]:
        so = ph.get(f"short_only_{h}") or {}
        b = ph["baseline"].get(h, {})
        if so.get("n"):
            print(f"  {h}日{'':<4}{so['avg']:>12.2%}{so['win_rate']:>8.0%}"
                  f"{b.get('avg', 0.0):>11.2%}{int(so['n']):>7}")
    p = ph.get("phase")
    so = ph.get("short_only")
    if p and p.get("n"):
        print("  " + "-" * 48)
        print(f"  两段式（先多{cfg.bounce.long_days}天再空{cfg.bounce.short_end_offset - cfg.bounce.long_days}天）"
              f"：平均 {p['avg']:+.2%}  中位 {p['median']:+.2%}  胜率 {p['win_rate']:.0%}  n={p['n']}")
        if so and so.get("n"):
            print(f"  只做空 7 天               ：平均 {so['avg']:+.2%}  中位 {so['median']:+.2%}  "
                  f"胜率 {so['win_rate']:.0%}  n={so['n']}")
        f1, f2 = ph.get("phase_first"), ph.get("phase_second")
        if f1 and f1.get("n"):
            print(f"  前后半段：前半 {f1['avg']:+.2%}（胜率 {f1['win_rate']:.0%}, n={f1['n']}）"
                  f" / 后半 {f2['avg']:+.2%}（胜率 {f2['win_rate']:.0%}, n={f2['n']}）")
    print()

    if args.sweep:
        rows = sweep_thresholds(data, cfg.crash)
        print("=" * 74)
        print(f"  阈值敏感性扫描（{args.days} 天，7 日持有期）")
        print("=" * 74)
        base = rows[0]["baseline"] if rows else 0.0
        print(f"  {'触发分':>8}{'事件数':>8}{'做空收益':>12}{'中位数':>10}{'胜率':>8}{'超额':>10}{'截尾超额':>11}")
        print("  " + "-" * 72)
        for r in rows:
            print(f"  {r['threshold']:>8.0f}{r['events']:>8}{r['short_avg']:>11.2%}"
                  f"{r['median']:>10.2%}{r['win_rate']:>8.0%}{r['excess']:>10.2%}"
                  f"{r['trimmed_excess']:>11.2%}")
        print("  " + "-" * 72)
        print(f"  基准（随机一天做空）：{base:+.2%}")
        signs = {1 if r["excess"] > 0 else -1 for r in rows if r["n"] >= 5}
        trimmed_signs = {1 if r["trimmed_excess"] > 0 else -1 for r in rows if r["n"] >= 8}
        if len(signs) > 1:
            print("  ⚠ 均值超额随阈值正负翻转 —— 信号不具备稳健性")
        if len(trimmed_signs) > 1:
            print("  ⚠ 截尾超额同样翻转 —— 结论依赖少数极端事件，样本量不足")
        if len(signs) == 1 and len(trimmed_signs) == 1:
            print("  ✓ 各阈值下超额方向一致（均值与截尾均值同向）")
        print()

        # 分段检验：前半段与后半段的方向是否一致
        dates = sorted(data["btc_daily"]["dt"])
        mid = str(dates[len(dates) // 2])
        print(f"  分段检验（前半段 ~{mid} / 后半段 {mid}~）：")
        print(f"  {'触发分':>8}{'前半超额':>12}{'后半超额':>12}{'方向':>8}")
        print("  " + "-" * 44)
        for th in (40, 50, 60, 70):
            h1 = analyze(data, _with_threshold(cfg.crash, th), end_date=mid)
            h2 = analyze(data, _with_threshold(cfg.crash, th), start_date=mid)
            s1 = h1.horizon_stats.get(7, {})
            s2 = h2.horizon_stats.get(7, {})
            e1, e2 = s1.get("excess", 0.0), s2.get("excess", 0.0)
            same = "一致" if (e1 > 0) == (e2 > 0) else "相反"
            print(f"  {th:>8.0f}{e1:>11.2%}{e2:>12.2%}{same:>8}"
                  f"   (前半 {s1.get('n', 0)} 次 / 后半 {s2.get('n', 0)} 次)")
        print()
    return 0


def cmd_reset(cfg: Config, args) -> int:
    from memeshort.state import StateStore

    store = StateStore("state")
    for f in (store.positions_file, store.account_file):
        if os.path.exists(f):
            os.remove(f)
            print(f"已删除 {f}")
    print("纸面状态已重置。")
    return 0


def main() -> int:
    # Windows 终端默认 GBK，中文与 emoji 会崩 —— 强制 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args()
    cfg = Config.load(args.config)
    logger = setup_logging(cfg.notify.log_dir, args.log_level, cfg.notify.log_file)
    logger.info("配置已加载：%s", args.config)

    handlers = {
        "scan": cmd_scan,
        "run": cmd_run,
        "status": cmd_status,
        "backtest": cmd_backtest,
        "reset": cmd_reset,
    }
    try:
        return handlers[args.command](cfg, args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
