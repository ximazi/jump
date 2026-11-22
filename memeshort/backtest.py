"""历史回测：验证「BTC 暴跌 -> 做空 Meme 币」这套逻辑历史上是否有效。

做法：
  1. 用与实盘完全相同的 evaluate_crash 打分函数，在 BTC 的 4H 数据上滚动找暴跌事件
  2. 对每个事件，测量 BTC 与 Meme 币组合在 1/3/7/14 个自然日的远期收益
  3. 做空 Meme 的收益 = -远期收益，统计均值与胜率

注意：这是事件驱动的统计回测，不含手续费与资金费，不能直接当收益预期。
"""

from __future__ import annotations

import bisect
import logging

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import CrashConfig
from .crash import drop_incomplete_daily, evaluate_crash
from .exchange import BinanceFutures

log = logging.getLogger("meme.backtest")

DEFAULT_BASKET = [
    "DOGE", "SHIB", "PEPE", "WIF", "BONK",
    "FLOKI", "TRUMP", "TURBO", "MOG", "PNUT",
    "MEME", "FARTCOIN", "NEIRO", "POPCAT", "BOME",
    "SPX6900", "PENGU", "ACT", "GOAT", "SATS",
]

HORIZONS = [1, 3, 7, 14]


def fetch_klines_paged(
    ex: BinanceFutures, symbol: str, interval: str, days: int
) -> pd.DataFrame:
    """分页拉取历史 K 线（单次上限 1500 根）。"""
    ms_per = {"1h": 3600_000, "4h": 14400_000, "1d": 86400_000}.get(interval, 14400_000)
    need = int(days * 86400_000 / ms_per) + 20
    frames: List[pd.DataFrame] = []
    start = int(pd.Timestamp.utcnow().timestamp() * 1000) - need * ms_per
    guard = 0
    while guard < 12:
        guard += 1
        df = ex.klines(symbol, interval, limit=1500, start_time=start)
        if df is None or df.empty:
            break
        frames.append(df)
        last_t = int(df["open_time"].iloc[-1])
        if len(df) < 1500:
            break
        start = last_t + ms_per
    if not frames:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    out["dt"] = pd.to_datetime(out["open_time"], unit="ms", utc=True).dt.date
    return out


@dataclass
class CrashEvent:
    ts: int
    date: str
    btc_score: float
    btc_price: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    events: List[CrashEvent]
    horizon_stats: Dict[int, Dict[str, float]]
    basket_size: int
    days: int
    btc_daily: Optional[pd.DataFrame] = None
    meme_daily: Dict[str, pd.DataFrame] = field(default_factory=dict)
    per_coin: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    baseline: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            "=" * 74,
            f"  回测区间 {self.days} 天 | 暴跌事件 {len(self.events)} 次 | Meme 篮子 {self.basket_size} 只",
            "=" * 74,
            "",
            f"{'持有期':<8}{'BTC均收益':>12}{'Meme均收益':>14}{'做空Meme收益':>16}{'胜率':>10}{'样本':>8}{'超额':>10}",
            "-" * 74,
        ]
        for h in HORIZONS:
            s = self.horizon_stats.get(h)
            if not s or s.get("n", 0) == 0:
                lines.append(f"{h}日{'':<4}  样本不足")
                continue
            lines.append(
                f"{h}日{'':<4}{s['btc_avg']:>11.2%}{s['meme_avg']:>14.2%}"
                f"{s['short_avg']:>16.2%}{s['win_rate']:>10.0%}{int(s['n']):>8}"
                f"{s.get('excess', 0.0):>10.2%}"
            )
        lines += ["", "稳健性（均值 vs 中位数 vs 截尾均值 —— 去掉最好最差各 20%）：", "-" * 74]
        for h in HORIZONS:
            s = self.horizon_stats.get(h)
            if not s or s.get("n", 0) == 0:
                continue
            lines.append(
                f"  {h}日  均值 {s['short_avg']:+.2%}   中位数 {s.get('median', 0.0):+.2%}   "
                f"截尾均值 {s.get('trimmed_avg', 0.0):+.2%}   "
                f"截尾超额 {s.get('trimmed_excess', 0.0):+.2%}"
            )
        if self.baseline:
            lines += ["", "基准对照（整个区间内随机选一天做空 Meme 篮子）：", "-" * 74]
            for h in HORIZONS:
                b = self.baseline.get(h)
                if not b:
                    continue
                lines.append(
                    f"  {h}日  基准做空收益 {b['avg']:+.2%}   基准胜率 {b['win_rate']:.0%}   "
                    f"样本 {b['n']} 天"
                )
            lines.append("  超额 = 信号做空收益 - 基准做空收益（这才是信号本身的贡献）")

        if self.per_coin:
            lines += ["", "各币种做空收益（7 日 / 14 日）：", "-" * 74]
            rows = []
            for sym, per_h in self.per_coin.items():
                s7 = per_h.get(7, {})
                s14 = per_h.get(14, {})
                rows.append((
                    sym,
                    s7.get("short_avg", 0.0), s7.get("win_rate", 0.0), s7.get("n", 0),
                    s14.get("short_avg", 0.0), s14.get("win_rate", 0.0), s14.get("n", 0),
                ))
            rows.sort(key=lambda r: -r[1])
            lines.append(f"  {'币种':<16}{'7日收益':>11}{'胜率':>9}{'样本':>7}"
                         f"{'14日收益':>12}{'胜率':>9}{'样本':>7}")
            for sym, r7, w7, n7, r14, w14, n14 in rows:
                lines.append(
                    f"  {sym:<16}{r7:>10.2%}{w7:>9.0%}{int(n7):>7}"
                    f"{r14:>12.2%}{w14:>9.0%}{int(n14):>7}"
                )

        lines += ["", "暴跌事件明细：", "-" * 74]
        for e in self.events:
            lines.append(
                f"  {e.date}  评分 {e.btc_score:5.1f}  BTC ${e.btc_price:,.0f}  "
                + "; ".join(e.reasons[:2])
            )
        lines += [
            "-" * 74,
            "说明：做空 Meme 收益 = -(Meme 远期收益)；未计手续费与资金费；",
            "      存在幸存者偏差（篮子只含当前仍活跃的币）。",
        ]
        return "\n".join(lines)


def _forward_return(daily: pd.DataFrame, date, h: int) -> Optional[float]:
    """date 起 h 个自然日后的收益。"""
    if daily.empty or date not in daily.index:
        return None
    idx = daily.index.get_loc(date)
    # 找到 +h 自然日的位置
    target = pd.Timestamp(date) + pd.Timedelta(days=h)
    positions = [i for i, d in enumerate(daily.index) if d >= target.date()]
    if not positions:
        return None
    j = positions[0]
    if j >= len(daily) or j <= idx:
        return None
    p0 = float(daily["close"].iloc[idx])
    p1 = float(daily["close"].iloc[j])
    return (p1 - p0) / p0 if p0 else None


def detect_events(
    btc_4h: pd.DataFrame,
    btc_daily: pd.DataFrame,
    cfg: CrashConfig,
    lookback: int = 300,
    cooldown_days: int = 14,
    eth_4h: Optional[pd.DataFrame] = None,
    eth_daily: Optional[pd.DataFrame] = None,
) -> List[CrashEvent]:
    """在 4H 数据上滚动打分，挑出跌破阈值的暴跌事件。

    日线部分必须传真实的日线切片 —— 直接拿 4H 当日线用会让「日线破位」
    因子退化成 4 小时低点，与实盘逻辑不一致。

    ETH 确认与实盘保持一致：BTC 达标之外，ETH 也要达到 eth_min_score，
    或者 BTC 分数高到可以单边触发（btc_solo_score）。
    """
    events: List[CrashEvent] = []
    if btc_4h is None or len(btc_4h) < cfg.slow_ema + 20:
        return events
    min_idx = cfg.slow_ema + 20
    last_event_day = None
    daily_dates = btc_daily["dt"] if btc_daily is not None and not btc_daily.empty else None

    # ETH 的 4H 序列按时间戳建有序索引，便于和 BTC 对齐（bisect，避免逐 bar 排序）
    eth_index = None
    if eth_4h is not None and not eth_4h.empty:
        pairs = sorted((int(t), i) for i, t in enumerate(eth_4h["open_time"]))
        eth_index = ([p[0] for p in pairs], [p[1] for p in pairs])

    def _daily_slice(daily: Optional[pd.DataFrame], d) -> Optional[pd.DataFrame]:
        if daily is None or daily.empty:
            return None
        sub = daily[daily["dt"] <= d]
        if len(sub) < 22:
            return None
        return sub.iloc[-22:]

    for i in range(min_idx, len(btc_4h)):
        w = btc_4h.iloc[max(0, i - lookback + 1): i + 1]
        if len(w) < cfg.slow_ema + 10:
            continue
        d = btc_4h["dt"].iloc[i]
        w_daily = _daily_slice(btc_daily, d)
        if w_daily is None:
            w_daily = w   # 日线数据不足时退回，宁可少计分也不误判
        ev = evaluate_crash(w, w_daily, 0.0, cfg, symbol="BTC")
        if ev.score < cfg.btc_min_score:
            continue

        # ETH 确认（与实盘一致）
        if eth_index is not None:
            ts = int(btc_4h["open_time"].iloc[i])
            eth_ts, eth_ids = eth_index
            k = bisect.bisect_right(eth_ts, ts) - 1
            j = eth_ids[k] if k >= 0 else None
            if j is not None and j >= cfg.slow_ema + 20:
                ew = eth_4h.iloc[max(0, j - lookback + 1): j + 1]
                ed = _daily_slice(eth_daily, d)
                if ed is None:
                    ed = ew
                eth_ev = evaluate_crash(ew, ed, 0.0, cfg, symbol="ETH")
                both = ev.score >= cfg.btc_min_score and eth_ev.score >= cfg.eth_min_score
                solo = ev.score >= cfg.btc_solo_score
                if not (both or solo):
                    continue
                ev.reasons = ev.reasons + [f"ETH 确认 {eth_ev.score:.0f} 分"]
            elif ev.score < cfg.btc_solo_score:
                continue

        if last_event_day is not None and (d - last_event_day).days < cooldown_days:
            continue
        last_event_day = d
        events.append(
            CrashEvent(
                ts=int(btc_4h["open_time"].iloc[i]),
                date=str(d),
                btc_score=ev.score,
                btc_price=ev.price,
                reasons=list(ev.reasons),
            )
        )
    return events


def detect_climax_events(df_daily: pd.DataFrame, cfg: CrashConfig) -> List[str]:
    """扫描整段历史里所有放量杀跌日（事件型信号的触发点）。"""
    if df_daily is None or len(df_daily) < 25:
        return []
    df = drop_incomplete_daily(df_daily)
    vma = df["volume"].rolling(20).mean()
    ratio = df["volume"] / vma
    chg = (df["close"] - df["open"]) / df["open"]
    days: List[str] = []
    for i in range(len(df)):
        if ratio.iloc[i] >= cfg.climax_volume_ratio and chg.iloc[i] <= cfg.climax_drop_pct:
            days.append(str(pd.to_datetime(df["open_time"].iloc[i], unit="ms", utc=True).date()))
    return days


def backtest_phase_strategy(
    data: Dict[str, Any],
    cfg: CrashConfig,
    bounce_cfg,
    climax_days: Sequence[str],
    horizons: Sequence[int] = (3, 5, 7, 10, 14),
) -> Dict[str, Any]:
    """对每个放量杀跌日 T0，比较三种走法做空 Meme 篮子的收益：

      bounce_then_short : 先多 long_days 天，再空到 short_end（两段式）
      short_only        : T0 收盘直接空 7 天
      baseline          : 区间内随机一天做空

    外加前后半段一致性检验。
    """
    meme = data["meme_daily"]
    daily = data["btc_daily"]
    dates = list(daily["dt"])
    mid = dates[len(dates) // 2] if dates else ""

    def meme_fwd(d0, n):
        rs = [_forward_return(df, d0, n) for df in meme.values()]
        rs = [r for r in rs if r is not None]
        return sum(rs) / len(rs) if rs else None

    rows: List[Dict[str, Any]] = []
    for day in climax_days:
        d0 = pd.Timestamp(day).date()
        rec: Dict[str, Any] = {"day": day, "half": "first" if str(d0) < str(mid) else "second"}
        for h in horizons:
            r = meme_fwd(d0, h)
            rec[f"fwd{h}"] = r
            rec[f"short{h}"] = -r if r is not None else None
        # 两段式：先多 long_days，再空到 short_end
        ld = bounce_cfg.long_days
        se = bounce_cfg.short_end_offset
        lm = meme_fwd(d0, ld)
        sm = meme_fwd(d0, se)
        if lm is not None and sm is not None:
            long_leg = lm
            short_leg = -(sm - lm) / (1 + lm)
            rec["phase"] = long_leg + short_leg
        else:
            rec["phase"] = None
        # 只做空：从 T0 收盘空到 T0+7
        r7 = meme_fwd(d0, 7)
        rec["short_only"] = -r7 if r7 is not None else None
        rows.append(rec)

    # 基准：区间内随机一天做空的平均
    base: Dict[int, Dict[str, float]] = {}
    all_days = list(meme.values())[0].index if meme else []
    for h in horizons:
        vals = []
        for d0 in all_days:
            v = meme_fwd(d0, h)
            if v is not None:
                vals.append(-v)
        if vals:
            base[h] = {
                "avg": sum(vals) / len(vals),
                "win_rate": sum(1 for v in vals if v > 0) / len(vals),
                "n": len(vals),
            }

    def agg(sel, key):
        vals = [r[key] for r in sel if r.get(key) is not None]
        if not vals:
            return None
        return {
            "avg": sum(vals) / len(vals),
            "median": float(np.median(vals)),
            "win_rate": sum(1 for v in vals if v > 0) / len(vals),
            "n": len(vals),
        }

    out: Dict[str, Any] = {"events": rows, "baseline": base, "horizons": list(horizons)}
    for h in horizons:
        out[f"short_only_{h}"] = agg(rows, f"short{h}")
        out[f"event_{h}"] = agg(rows, f"short{h}")
    out["phase"] = agg(rows, "phase")
    out["phase_first"] = agg([r for r in rows if r["half"] == "first"], "phase")
    out["phase_second"] = agg([r for r in rows if r["half"] == "second"], "phase")
    out["short_only"] = agg(rows, "short_only")
    return out


def load_backtest_data(
    ex: BinanceFutures,
    days: int = 730,
    basket: Sequence[str] = tuple(DEFAULT_BASKET),
) -> Dict[str, Any]:
    """拉取回测所需的全部数据（与分析分离，便于做敏感性扫描）。"""
    log.info("拉取 BTC 4H 历史数据（%d 天）...", days)
    btc_4h = fetch_klines_paged(ex, "BTCUSDT", "4h", days)
    btc_daily = fetch_klines_paged(ex, "BTCUSDT", "1d", days)
    if btc_daily.empty:
        raise RuntimeError("BTC 日线数据获取失败")
    log.info("回测窗口 %s ~ %s", btc_daily["dt"].iloc[0], btc_daily["dt"].iloc[-1])

    eth_4h = fetch_klines_paged(ex, "ETHUSDT", "4h", days)
    eth_daily = fetch_klines_paged(ex, "ETHUSDT", "1d", days)

    meme_daily: Dict[str, pd.DataFrame] = {}
    alias = ex.perpetual_alias_map("USDT")
    for base in basket:
        sym = alias.get(base.upper())
        if not sym:
            log.warning("%s 无对应永续合约，跳过", base)
            continue
        try:
            df = fetch_klines_paged(ex, sym, "1d", days)
            if not df.empty:
                meme_daily[sym] = df.set_index("dt")[["close"]]
        except Exception as e:
            log.warning("拉取 %s(%s) 失败: %s", base, sym, e)
    log.info("Meme 篮子可用 %d 只：%s", len(meme_daily), ", ".join(meme_daily))

    return {
        "btc_4h": btc_4h, "btc_daily": btc_daily,
        "eth_4h": eth_4h, "eth_daily": eth_daily,
        "meme_daily": meme_daily, "days": days,
    }


def basket_forward_returns(
    meme_daily: Dict[str, pd.DataFrame], horizon: int
) -> List[Tuple[Any, float]]:
    """所有交易日的 Meme 篮子远期收益 —— 用作「随便哪天做空」的基准。"""
    if not meme_daily:
        return []
    all_dates = set()
    for df in meme_daily.values():
        all_dates.update(df.index)
    out: List[Tuple[Any, float]] = []
    for d in sorted(all_dates):
        rets = []
        for df in meme_daily.values():
            r = _forward_return(df, d, horizon)
            if r is not None:
                rets.append(r)
        if rets:
            out.append((d, sum(rets) / len(rets)))
    return out


def analyze(
    data: Dict[str, Any],
    cfg: CrashConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """在已加载的数据上跑一遍事件检测与收益统计。"""
    btc_4h = data["btc_4h"]
    btc_daily = data["btc_daily"]
    meme_daily = data["meme_daily"]

    events = detect_events(
        btc_4h, btc_daily, cfg,
        eth_4h=data.get("eth_4h"),
        eth_daily=data.get("eth_daily"),
    )
    if start_date:
        events = [e for e in events if e.date >= start_date]
    if end_date:
        events = [e for e in events if e.date <= end_date]
    log.info("识别到 %d 次暴跌事件", len(events))

    btc_idx = btc_daily.set_index("dt")[["close"]]
    stats: Dict[int, Dict[str, float]] = {}
    per_coin: Dict[str, Dict[int, Dict[str, float]]] = {}
    baseline: Dict[int, Dict[str, float]] = {}

    for h in HORIZONS:
        # 基准：整个区间内「随便哪天做空 Meme 篮子」的平均结果
        base_series = basket_forward_returns(meme_daily, h)
        base_vals = [v for _, v in base_series]
        if base_vals:
            baseline[h] = {
                "avg": sum(base_vals) / len(base_vals),
                "win_rate": sum(1 for v in base_vals if -v > 0) / len(base_vals),
                "n": len(base_vals),
            }
        btc_rets: List[float] = []
        meme_rets: List[float] = []
        coin_rets: Dict[str, List[float]] = {sym: [] for sym in meme_daily}
        for ev in events:
            d = pd.Timestamp(ev.date).date()
            b = _forward_return(btc_idx, d, h)
            if b is not None:
                btc_rets.append(b)
            per_event: List[float] = []
            for sym, df in meme_daily.items():
                r = _forward_return(df, d, h)
                if r is not None:
                    per_event.append(r)
                    coin_rets[sym].append(r)
            if per_event:
                meme_rets.append(sum(per_event) / len(per_event))

        if not meme_rets:
            continue
        shorts = [-r for r in meme_rets]
        base_avg = baseline.get(h, {}).get("avg", 0.0)
        srt = sorted(shorts)
        n = len(srt)
        # 截尾均值：去掉最好和最差的各 20%，看结论是否靠少数极端事件撑起来
        k = int(n * 0.2)
        trimmed = srt[k:n - k] if n - 2 * k >= 2 else srt
        median = srt[n // 2] if n else 0.0
        stats[h] = {
            "btc_avg": sum(btc_rets) / len(btc_rets) if btc_rets else 0.0,
            "meme_avg": sum(meme_rets) / len(meme_rets),
            "short_avg": sum(shorts) / len(shorts),
            "median": median,
            "trimmed_avg": sum(trimmed) / len(trimmed) if trimmed else 0.0,
            "win_rate": sum(1 for s in shorts if s > 0) / len(shorts),
            "n": len(shorts),
            "meme_worst": min(meme_rets),
            "meme_best": max(meme_rets),
            # 超额：相对「随便哪天做空」的基准，信号到底加了多少价值
            "excess": sum(shorts) / len(shorts) - (-base_avg),
            "median_excess": median - (-base_avg),
            "trimmed_excess": (sum(trimmed) / len(trimmed) if trimmed else 0.0) - (-base_avg),
            "returns": shorts,
        }
        for sym, rets in coin_rets.items():
            if not rets:
                continue
            s = [-r for r in rets]
            per_coin.setdefault(sym, {})[h] = {
                "short_avg": sum(s) / len(s),
                "win_rate": sum(1 for x in s if x > 0) / len(s),
                "n": len(s),
            }

    return BacktestResult(
        events=events,
        horizon_stats=stats,
        basket_size=len(meme_daily),
        days=int(data.get("days", 0)),
        btc_daily=btc_idx,
        meme_daily=meme_daily,
        per_coin=per_coin,
        baseline=baseline,
    )


def sweep_thresholds(
    data: Dict[str, Any],
    cfg: CrashConfig,
    thresholds: Sequence[float] = (40, 50, 60, 70, 78, 85),
    horizon: int = 7,
) -> List[Dict[str, Any]]:
    """阈值敏感性扫描：若结论随阈值反复翻转，说明是噪声而非策略。"""
    import copy

    rows: List[Dict[str, Any]] = []
    for th in thresholds:
        c = copy.deepcopy(cfg)
        c.btc_min_score = float(th)
        # 注意：btc_solo_score 与 eth_min_score 保持配置原值，
        # 否则 ETH 确认会被旁路，扫描结果就不等于实盘逻辑了。
        res = analyze(data, c)
        s = res.horizon_stats.get(horizon, {})
        b = res.baseline.get(horizon, {})
        rows.append({
            "threshold": th,
            "events": len(res.events),
            "btc_avg": s.get("btc_avg", 0.0),
            "meme_avg": s.get("meme_avg", 0.0),
            "short_avg": s.get("short_avg", 0.0),
            "median": s.get("median", 0.0),
            "trimmed_avg": s.get("trimmed_avg", 0.0),
            "trimmed_excess": s.get("trimmed_excess", 0.0),
            "win_rate": s.get("win_rate", 0.0),
            "excess": s.get("excess", 0.0),
            "baseline": b.get("avg", 0.0),
            "baseline_win": b.get("win_rate", 0.0),
            "n": s.get("n", 0),
        })
    return rows


def run_backtest(
    ex: BinanceFutures,
    cfg: CrashConfig,
    days: int = 730,
    basket: Sequence[str] = tuple(DEFAULT_BASKET),
) -> BacktestResult:
    data = load_backtest_data(ex, days, basket)
    return analyze(data, cfg)
