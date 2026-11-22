"""技术指标（纯 pandas/numpy，无 TA-Lib 依赖）。"""

from __future__ import annotations


import numpy as np
import pandas as pd


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(loss.notna(), 50.0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """真实波幅均值。df 需含 high/low/close。"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(length).mean()


def log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1)).dropna()


def beta_vs_bench(asset: pd.Series, bench: pd.Series) -> float:
    """asset 相对 bench 的 beta（对数收益率的协方差/方差）。"""
    a = log_returns(pd.DataFrame({"close": asset}))
    b = log_returns(pd.DataFrame({"close": bench}))
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 15:
        return 0.0
    ja, jb = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = jb.var()
    if var_b == 0 or np.isnan(var_b):
        return 0.0
    return float(ja.cov(jb) / var_b)


def downside_beta(asset: pd.Series, bench: pd.Series, quantile: float = 0.15) -> float:
    """下跌 beta：只在 benchmark 最差的那部分日子里，资产跌幅是 benchmark 的几倍。

    这才是「高 Beta 放大」策略真正关心的指标 —— 常规回归 beta 会被上涨日
    的噪音拉低，掩盖掉暴跌时的踩踏放大。quantile 越小，取的越是极端的下跌日。
    """
    a = log_returns(pd.DataFrame({"close": asset}))
    b = log_returns(pd.DataFrame({"close": bench}))
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return 0.0
    joined.columns = ["a", "b"]
    mask = joined["b"] <= joined["b"].quantile(quantile)
    if int(mask.sum()) < 5:
        return 0.0
    mb = float(joined.loc[mask, "b"].mean())
    ma = float(joined.loc[mask, "a"].mean())
    if mb == 0:
        return 0.0
    return ma / mb


def swing_high(series: pd.Series, lookback: int = 20, exclude_last: int = 1) -> float:
    """最近 lookback 根内的最高价，忽略最后 exclude_last 根（避免用到当前bar）。"""
    return float(series.iloc[-(lookback + exclude_last):-exclude_last].max())


def swing_low(series: pd.Series, lookback: int = 20, exclude_last: int = 1) -> float:
    return float(series.iloc[-(lookback + exclude_last):-exclude_last].min())


def rolling_zscore(s: pd.Series, length: int = 50) -> float:
    m = s.rolling(length).mean().iloc[-1]
    sd = s.rolling(length).std().iloc[-1]
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((s.iloc[-1] - m) / sd)


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1.0).min())


def realized_vol(s: pd.Series, periods_per_year: int = 365) -> float:
    r = log_returns(pd.DataFrame({"close": s}))
    if len(r) < 5:
        return 0.0
    return float(r.std() * np.sqrt(periods_per_year))
