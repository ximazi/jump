"""盘中做空入场时机过滤器：破位 -> 弱反弹受阻 -> 二次转弱。

解决直接市价追空容易空在急跌底部、止损过宽导致净盈亏比不达标的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .indicators import atr, ema


@dataclass
class EntrySignal:
    ready: bool
    reason: str
    entry_price: float = 0.0
    structure_stop: float = 0.0
    atr_value: float = 0.0


def evaluate_rebound_reject_short(
    df_intraday: pd.DataFrame,
    ema_len: int = 20,
    lookback_bars: int = 16,
    atr_buffer_mult: float = 0.35,
    drop_last_incomplete: bool = True,
) -> EntrySignal:
    """判断短周期（默认 15m）是否出现「破位后弱反弹受阻」做空时机。

    仅使用已收盘 K 线，避免未来函数：
      1. 趋势偏空：最新收盘价位于 EMA 下方。
      2. 曾有破位下探：近 8 根低点低于此前 lookback_bars 区间低点。
      3. 弱反弹受阻：近 4 根内出现局部反弹高点，但最新一根收盘转阴并跌破前一根低点，
         或出现明显上影受阻且反弹阶段缩量。
      4. 止损锚点：局部反弹高点 + atr_buffer_mult * ATR14。
    """
    min_rows = max(ema_len + 10, lookback_bars + 10) + (1 if drop_last_incomplete else 0)
    if df_intraday is None or len(df_intraday) < min_rows:
        return EntrySignal(False, "盘中 K 线数据不足")

    df = df_intraday.iloc[:-1].copy() if drop_last_incomplete else df_intraday.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)

    ema_f = ema(close, ema_len)
    atr_s = atr(df, 14)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_ema = float(ema_f.iloc[-1])
    last_atr = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else 0.0

    if last_close >= last_ema:
        return EntrySignal(
            False,
            f"现价 {last_close:.6g} 未处于 EMA{ema_len}({last_ema:.6g}) 下方",
        )

    prior_low = float(low.iloc[-(lookback_bars + 8):-8].min())
    recent_low = float(low.iloc[-8:].min())
    if recent_low >= prior_low:
        return EntrySignal(False, "近期未出现有效结构破位")

    rebound_window = df.iloc[-5:-1]
    rebound_high = float(rebound_window["high"].max())
    prev_leg_vol = float(vol.iloc[-10:-5].mean())
    rebound_vol = float(rebound_window["volume"].mean())
    vol_weak = prev_leg_vol > 0 and rebound_vol <= prev_leg_vol * 0.90

    turned_bearish = (last_close < last_open) and (last_close < float(low.iloc[-2]))
    upper_wick = float(high.iloc[-1]) - max(last_open, last_close)
    body = abs(last_close - last_open)
    rejected_wick = upper_wick > max(body * 0.6, 1e-12) and last_close < last_open

    if not (turned_bearish or (rejected_wick and vol_weak)):
        return EntrySignal(False, "等待反弹受阻确认（尚未出现转弱阴线或缩量上影）")

    stop_price = max(rebound_high, float(high.iloc[-1])) + atr_buffer_mult * last_atr
    return EntrySignal(
        ready=True,
        reason=(
            f"破位后弱反弹受阻（收 {last_close:.6g} < EMA{ema_len} {last_ema:.6g}，"
            f"反弹高点 {rebound_high:.6g}，止损含 {atr_buffer_mult}x ATR 缓冲）"
        ),
        entry_price=last_close,
        structure_stop=stop_price,
        atr_value=last_atr,
    )
