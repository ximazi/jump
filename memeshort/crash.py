"""暴跌信号检测。

对 BTC / ETH 分别做 7 因子加权打分，0~100 分：
  1. 4H 均线空头排列/死叉        2. 跌破 N 根区间下沿
  3. 高位回撤                    4. 放量阴线
  5. 缩量反弹（反弹无力）        6. 日线破位
  7. 资金费率多头拥挤

触发条件：BTC >= btc_min_score 且 ETH >= eth_min_score，或 BTC >= btc_solo_score。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import CrashConfig
from .indicators import ema
from .exchange import BinanceFutures

log = logging.getLogger("meme.crash")


@dataclass
class VolumeClimax:
    """放量杀跌事件。"""

    day: str                       # 事件日期 YYYY-MM-DD
    days_ago: int = 0              # 距今天数
    volume_ratio: float = 0.0      # 量比（相对 20 日均量）
    drop_pct: float = 0.0          # 当日跌幅（负数）
    close: float = 0.0

    @property
    def strength(self) -> float:
        """事件强度：量比越大、跌得越狠，强度越高。"""
        return min(3.0, self.volume_ratio) * min(3.0, abs(self.drop_pct) / 0.04)

    def summary(self) -> str:
        return (
            f"{self.day} 放量杀跌（量比 {self.volume_ratio:.2f}x，"
            f"跌 {self.drop_pct:.1%}，{self.days_ago} 天前）"
        )


def drop_incomplete_daily(df: pd.DataFrame) -> pd.DataFrame:
    """丢掉还没收盘的日线。

    当日 K 线是动态变化的：昨天收盘时它是完整的一天，现在只剩半天。
    用它做「放量杀跌」判定会误报，所以必须只用已收盘的日线。
    """
    if df is None or df.empty:
        return df
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if "close_time" in df.columns:
        finished = df[df["close_time"] <= now_ms]
        if not finished.empty:
            return finished
    return df.iloc[:-1] if len(df) > 1 else df


def detect_volume_climax(
    df_daily: pd.DataFrame,
    cfg: CrashConfig,
    today: Optional[date] = None,
    max_age_days: Optional[int] = None,
) -> Optional[VolumeClimax]:
    """找最近一次放量杀跌事件。

    条件：日线成交量 >= 20 日均量 * climax_volume_ratio，且当日跌幅 <= climax_drop_pct。

    max_age_days：回看多少天。必须覆盖整个事件生命周期（到 T0+short_end_offset），
    否则事件过了新鲜期就再也检不出来，阶段推进（反弹腿 -> 做空腿）会断掉。
    """
    if df_daily is None or len(df_daily) < 25:
        return None
    df = drop_incomplete_daily(df_daily)
    if len(df) < 25:
        return None

    vma = df["volume"].rolling(20).mean()
    chg = (df["close"] - df["open"]) / df["open"]
    ratio = df["volume"] / vma

    lookback = max_age_days if max_age_days is not None else cfg.climax_max_age_days
    window = min(len(df), lookback + 1)
    tail = df.iloc[-window:].copy()
    tail_ratio = ratio.iloc[-window:]
    tail_chg = chg.iloc[-window:]

    today = today or datetime.now(timezone.utc).date()
    best: Optional[VolumeClimax] = None
    max_vr = getattr(cfg, "climax_max_volume_ratio", 0) or 0
    for i in range(len(tail)):
        r = float(tail_ratio.iloc[i])
        c = float(tail_chg.iloc[i])
        if r < cfg.climax_volume_ratio or r > max_vr:
            continue          # 量比太低不触发；太高 = 恐慌见底，不做空
        if c <= cfg.climax_drop_pct:
            d = tail.iloc[i]
            day = pd.to_datetime(d["open_time"], unit="ms", utc=True).date()
            vc = VolumeClimax(
                day=str(day),
                days_ago=(today - day).days,
                volume_ratio=r,
                drop_pct=c,
                close=float(d["close"]),
            )
            # 保留最新的那个
            best = vc
    return best


@dataclass
class CrashEval:
    symbol: str
    score: float = 0.0
    price: float = 0.0
    triggered: bool = False
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    climax: Optional[VolumeClimax] = None

    def summary(self) -> str:
        flag = "🔴 触发" if self.triggered else "⚪ 未触发"
        return f"{self.symbol} {self.score:.0f}分 {flag}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def evaluate_crash(
    df_intraday: pd.DataFrame,
    df_daily: pd.DataFrame,
    funding_rate: float,
    cfg: CrashConfig,
    symbol: str = "",
) -> CrashEval:
    """核心打分函数，纯函数便于回测复用。"""
    ev = CrashEval(symbol=symbol)
    if df_intraday is None or len(df_intraday) < cfg.slow_ema + 10:
        ev.reasons.append("K线数据不足")
        return ev

    close = df_intraday["close"]
    high = df_intraday["high"]
    low = df_intraday["low"]
    open_ = df_intraday["open"]
    vol = df_intraday["volume"]

    last = float(close.iloc[-1])
    ev.price = last
    score = 0.0
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    ema_f = ema(close, cfg.fast_ema)
    ema_s = ema(close, cfg.slow_ema)

    # --- 1. 均线：空头排列 / 近 4 根内死叉 -------------------------------
    if float(ema_f.iloc[-1]) < float(ema_s.iloc[-1]):
        recent = (ema_f.iloc[-4:-1] >= ema_s.iloc[-4:-1]).any()
        w = cfg.w_ema_cross if recent else cfg.w_ema_cross * 0.6
        score += w
        tag = "死叉" if recent else "空头排列"
        reasons.append(f"{cfg.tf_intraday} EMA{cfg.fast_ema}/EMA{cfg.slow_ema} {tag}")
    metrics["ema_fast"] = float(ema_f.iloc[-1])
    metrics["ema_slow"] = float(ema_s.iloc[-1])

    # --- 2. 跌破区间下沿（不含当前 K 线）--------------------------------
    prior = low.iloc[-(cfg.swing_lookback + 1):-1]
    if len(prior):
        swing_low = float(prior.min())
        metrics["swing_low"] = swing_low
        if last < swing_low:
            score += cfg.w_swing_break
            reasons.append(
                f"跌破{cfg.swing_lookback}根区间下沿 {swing_low:.4f} (现价 {last:.4f})"
            )

    # --- 3. 高位回撤 ----------------------------------------------------
    hh = float(high.iloc[-cfg.swing_lookback:].max())
    if hh > 0:
        dd = (hh - last) / hh
        metrics["drawdown"] = dd
        metrics["swing_high"] = hh
        if dd >= cfg.drawdown_threshold:
            score += cfg.w_drawdown * min(1.0, dd / (cfg.drawdown_threshold * 2))
            reasons.append(f"自{cfg.swing_lookback}根高点回撤 {_fmt_pct(dd)}")

    # --- 4. 放量阴线 ----------------------------------------------------
    if len(vol) >= 20:
        vma = float(vol.rolling(20).mean().iloc[-1])
        metrics["vol_ratio"] = float(vol.iloc[-1] / vma) if vma else 0.0
        if float(close.iloc[-1]) < float(open_.iloc[-1]) and metrics["vol_ratio"] >= cfg.volume_spike:
            score += cfg.w_volume
            reasons.append(f"放量阴线 (量能 {metrics['vol_ratio']:.2f}x MA20)")

    # --- 5. 缩量反弹（反弹无力）----------------------------------------
    if len(df_intraday) >= 12:
        leg = df_intraday.iloc[-10:]
        recent = df_intraday.iloc[-5:]
        made_higher_high = float(recent["high"].max()) > float(leg["high"].iloc[:5].max())
        vol_shrank = float(recent["volume"].mean()) < float(leg["volume"].iloc[:5].mean()) * 0.8
        below_ema = last < float(ema_f.iloc[-1])
        if made_higher_high and vol_shrank and below_ema:
            score += cfg.w_weak_bounce
            reasons.append("反弹缩量且仍在EMA20下方（反弹无力）")

    # --- 6. 日线破位 ----------------------------------------------------
    if df_daily is not None and len(df_daily) >= 22:
        d_prior = df_daily["low"].iloc[-21:-1]
        if len(d_prior):
            d_low = float(d_prior.min())
            metrics["daily_swing_low"] = d_low
            if float(df_daily["close"].iloc[-1]) < d_low:
                score += cfg.w_daily_break
                reasons.append(f"日线跌破20日低点 {d_low:.4f}")

    # --- 7. 资金费率多头拥挤 --------------------------------------------
    metrics["funding_rate"] = funding_rate
    if funding_rate >= cfg.funding_crowd_level:
        score += cfg.w_funding_crowd * min(1.0, funding_rate / (cfg.funding_crowd_level * 2))
        reasons.append(f"资金费率 {_fmt_pct(funding_rate)}/8h，多头拥挤")

    total_w = (
        cfg.w_ema_cross + cfg.w_swing_break + cfg.w_drawdown + cfg.w_volume
        + cfg.w_weak_bounce + cfg.w_daily_break + cfg.w_funding_crowd
    )
    ev.score = round(min(100.0, score / total_w * 100.0), 1)
    ev.reasons = reasons
    ev.metrics = metrics
    return ev


def score_at_event(
    df_intraday: pd.DataFrame,
    df_daily: pd.DataFrame,
    event_day: str,
    cfg: CrashConfig,
) -> Tuple[float, List[str]]:
    """计算**事件发生时刻**的状态分。

    为什么要用事件时刻而不是当前状态：
    状态分衡量的是「价格是否处在局部低位 + 指标是否空头」。价格一反弹，
    「跌破区间下沿」「日线破位」就不再计分，分数反而从 64 掉到 21。
    拿它当事件确认线，就会在最该做反弹的时候把信号关掉 —— 这是设计缺陷。
    事件时刻的分数才是事件的固有属性，且任何时候重算结果都一样。
    """
    if df_intraday is None or df_daily is None or df_daily.empty:
        return 0.0, []
    try:
        day = datetime.strptime(str(event_day), "%Y-%m-%d").date()
    except ValueError:
        return 0.0, []

    # T0 当天已收盘的日线（含 T0）
    d = df_daily[pd.to_datetime(df_daily["open_time"], unit="ms", utc=True).dt.date <= day]
    if len(d) < 22:
        return 0.0, []
    # T0 收盘时最后一个已完成的 4H bar：开在 T0 20:00，收在 T0 24:00
    cutoff = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000) + 86400_000 - 4 * 3600_000
    i = df_intraday[pd.to_datetime(df_intraday["open_time"], unit="ms", utc=True).dt.date < day]
    if len(i) < cfg.slow_ema + 20:
        return 0.0, []
    w = i.iloc[max(0, len(i) - 300):]
    ev = evaluate_crash(w, d.iloc[-22:], 0.0, cfg, symbol="EVENT")
    return ev.score, list(ev.reasons)


@dataclass
class CrashSignal:
    """完整的暴跌信号。

    mode = event 时，触发依据是「刚发生放量杀跌」这个事件；
    mode = state 时，触发依据是「价格在低位 + 指标空头」这个状态。
    回测显示事件型显著优于状态型（见 README 第三节）。
    """

    triggered: bool = False
    mode: str = "event"
    btc: Optional[CrashEval] = None
    eth: Optional[CrashEval] = None
    climax: Optional[VolumeClimax] = None
    reasons: List[str] = field(default_factory=list)
    phase: str = "wait"       # wait | bounce | short
    offset: int = 0           # 距事件日 T0 的天数
    event_score: float = 0.0  # 事件发生时刻的状态分（用于确认）

    @property
    def event_day(self) -> Optional[str]:
        return self.climax.day if self.climax else None

    def summary(self) -> str:
        flag = "🔴 触发" if self.triggered else "⚪ 未触发"
        phase_cn = {"wait": "等待", "bounce": "反弹腿(做多)", "short": "做空腿"}[self.phase]
        return (
            f"{flag} | 模式 {self.mode} | 阶段 {phase_cn} (T0{self.offset:+d} 天) | "
            f"BTC {self.btc.score if self.btc else 0:.0f} 分"
        )


class CrashDetector:
    """拉数据 + 评估 BTC/ETH + 事件/状态判定 + 阶段计算。"""

    def __init__(self, ex: BinanceFutures, cfg: CrashConfig, bounce_cfg=None):
        self.ex = ex
        self.cfg = cfg
        self.bounce_cfg = bounce_cfg

    def _eval_symbol(self, symbol: str) -> CrashEval:
        df_i = self.ex.klines(symbol, self.cfg.tf_intraday, limit=300)
        df_d = self.ex.klines(symbol, self.cfg.tf_daily, limit=60)
        try:
            funding = float(self.ex.premium_index(symbol)["lastFundingRate"])
        except Exception:
            funding = 0.0
        return evaluate_crash(df_i, df_d, funding, self.cfg, symbol=symbol)

    def _lifecycle_end(self) -> int:
        """事件生命周期结束于 T0+第几天。"""
        bc = self.bounce_cfg
        if bc is not None and bc.enabled:
            return bc.short_end_offset
        return 7

    def _phase_for(self, climax: Optional[VolumeClimax]) -> Tuple[str, int]:
        """根据事件日算出当前应处于哪个阶段。

        T0 = 放量杀跌当日
          offset 1 ~ long_days            -> bounce（做反弹腿）
          short_start_offset ~ short_end  -> short（做空腿）
        """
        if climax is None:
            return "wait", 0
        offset = climax.days_ago
        bc = self.bounce_cfg
        if bc is not None and bc.enabled:
            if 1 <= offset <= bc.long_days:
                return "bounce", offset
            if bc.short_start_offset <= offset <= bc.short_end_offset:
                return "short", offset
            return "wait", offset
        # 未开启两段式：事件当天即可做空
        if 1 <= offset <= 7:
            return "short", offset
        return "wait", offset

    def detect(self) -> CrashSignal:
        """评估并返回完整信号。"""
        btc = self._eval_symbol(self.cfg.btc_symbol)
        eth = self._eval_symbol(self.cfg.eth_symbol)

        # 检测窗口必须覆盖整个生命周期，否则阶段推进会断
        lookback = max(self.cfg.climax_max_age_days, self._lifecycle_end())
        try:
            df_daily = self.ex.klines(self.cfg.btc_symbol, self.cfg.tf_daily, limit=60)
            climax = detect_volume_climax(df_daily, self.cfg, max_age_days=lookback)
        except Exception as e:
            log.warning("放量杀跌检测失败: %s", e)
            climax = None
        btc.climax = climax

        state_ok = btc.score >= self.cfg.btc_min_score and eth.score >= self.cfg.eth_min_score
        solo_ok = btc.score >= self.cfg.btc_solo_score

        # 确认用「事件发生时刻」的状态分，而不是当前状态分。
        # 当前状态分会随着反弹失效（跌破区间下沿、日线破位都不再计分），
        # 用它当确认线会在最该做反弹的时候把信号关掉。
        event_score = 0.0
        event_reasons: List[str] = []
        if climax is not None:
            try:
                df_i = self.ex.klines(self.cfg.btc_symbol, self.cfg.tf_intraday, limit=300)
                event_score, event_reasons = score_at_event(
                    df_i, df_daily, climax.day, self.cfg
                )
            except Exception as e:
                log.warning("事件时刻评分失败: %s", e)

        state_confirmed = event_score >= self.cfg.climax_confirm_score
        if self.cfg.climax_require_eth:
            state_confirmed = state_confirmed and eth.score >= self.cfg.eth_min_score
        active = climax is not None and climax.days_ago <= self._lifecycle_end()
        event_ok = active and state_confirmed
        fresh = climax is not None and climax.days_ago <= self.cfg.climax_max_age_days

        mode = (self.cfg.signal_mode or "event").lower()
        if mode == "event":
            triggered = event_ok
        elif mode == "state":
            triggered = state_ok or solo_ok
        else:  # any
            triggered = (state_ok or solo_ok) or event_ok

        reasons: List[str] = []
        if climax is not None:
            reasons.append(climax.summary())
            reasons.append(f"事件时刻状态分 {event_score:.0f}（当前状态分 {btc.score:.0f}）")
            if not fresh:
                reasons.append(f"事件已过新鲜期（{climax.days_ago} 天前），仍在生命周期内")
        if state_ok or solo_ok:
            reasons.append(f"状态评分 BTC {btc.score:.0f} / ETH {eth.score:.0f}")
        if mode == "event" and not triggered:
            if climax is None:
                reasons.append("近期无放量杀跌事件")
            elif not active:
                reasons.append(f"放量杀跌已过 {climax.days_ago} 天，超出生命周期")
            if event_score < self.cfg.climax_confirm_score:
                reasons.append(f"事件时刻状态分 {event_score:.0f} < 确认线 {self.cfg.climax_confirm_score:.0f}")

        phase, offset = self._phase_for(climax if active else None)
        btc.triggered = triggered
        eth.triggered = triggered
        return CrashSignal(
            triggered=triggered,
            mode=mode,
            btc=btc,
            eth=eth,
            climax=climax,
            reasons=reasons,
            phase=phase,
            offset=offset,
            event_score=event_score,
        )
