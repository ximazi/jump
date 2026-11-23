"""Meme 币筛选。

在确认大盘暴跌后，从海量 Meme 币里挑出「最适合做空」的一小撮：

硬性过滤（不做就等着被拉盘爆仓）
  - 市值 >= min_market_cap（默认 5 亿），低于此庄家可随手拉盘 50%
  - 24h 成交额 >= min_quote_volume_24h，防滑点
  - Binance USDT 永续合约已上线 >= min_onboard_days
  - 点差 <= max_spread_bps，流动性太差直接跳过
  - 资金费率异常高（> max_funding）视为新币/异常，排除

软性打分（决定优先级）
  - 资金费率拥挤度：多头越拥挤，做空越有踩踏空间
  - 相对 BTC 的 Beta：越高越符合「高 Beta 放大」逻辑
  - 流动性：越大滑点越小
  - 前期涨幅：爆炒过头的回落风险更大
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import ScreenerConfig
from .exchange import BinanceFutures
from .indicators import beta_vs_bench, downside_beta
from .utils import fmt_usd

log = logging.getLogger("meme.screener")

# 币安对单位价格 < 1 的币用 1000x / 1000000x 前缀开合约：
#   PEPE -> 1000PEPEUSDT, SHIB -> 1000SHIBUSDT, MOG -> 1000000MOGUSDT
# 做符号映射时必须剥掉这层前缀，否则头部 Meme 币会被整体漏掉。
# 映射逻辑见 exchange.perpetual_alias_map()


def _date_indexed(df: pd.DataFrame) -> pd.Series:
    """把日线收盘价转成以日期为索引的 Series（供 Beta 按日期对齐）。"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    return pd.Series(df["close"].to_numpy(dtype=float), index=pd.Index(idx))


# CoinGecko 不可用时的兜底白名单（头部 Meme，永续流动性较好）
FALLBACK_MEMES = [
    "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "TRUMP", "MELANIA",
    "POPCAT", "TURBO", "BOME", "NEIRO", "SATS", "MOG", "RATS", "MYRO",
    "SPX6900", "PENGU", "PNUT", "ACT", "GOAT", "MOODENG", "CHILLGUY",
    "GIGACHAD", "AI16Z", "BRETT", "MEME", "FARTCOIN", "ANDY", "LUNA",
    "TIGER", "BABYDOGE", "MEW", "DODO", "POL", "SXT",
]


@dataclass
class MemeCandidate:
    symbol: str
    base: str
    market_cap: float = 0.0
    quote_volume_24h: float = 0.0
    funding_rate: float = 0.0
    beta: float = 0.0
    downside_beta: float = 0.0   # 暴跌日的跌幅放大倍数（策略核心指标）
    spread_bps: float = 0.0
    ask_depth_usd: float = 0.0
    bid_depth_usd: float = 0.0
    rel_weakness: float = 0.0    # 近 3 日相对 BTC 弱势度（正值表示比 BTC 更弱）
    onboard_days: int = 999
    price: float = 0.0
    change_24h: float = 0.0
    change_7d: float = 0.0
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.symbol:<12} 分{self.score:5.1f} | 市值{fmt_usd(self.market_cap):>9} "
            f"| 量{fmt_usd(self.quote_volume_24h):>9} | 费率{self.funding_rate * 100:+.4f}% "
            f"| β{self.beta:.2f} | 跌β{self.downside_beta:.2f} | 点差{self.spread_bps:.1f}bp"
        )


class MemeScreener:
    def __init__(self, ex: BinanceFutures, cfg: ScreenerConfig):
        self.ex = ex
        self.cfg = cfg
        self._tickers: Optional[Dict[str, dict]] = None
        self._premium: Optional[Dict[str, dict]] = None
        self._btc_daily: Optional[pd.Series] = None
        self._mcap_cache: Optional[Dict[str, float]] = None
        self._alias: Optional[Dict[str, str]] = None
        self.filter_log: Dict[str, str] = {}

    # ------------------------------------------------------------ 缓存

    def _load_tickers(self) -> Dict[str, dict]:
        if self._tickers is None:
            rows = self.ex.all_tickers()
            self._tickers = {r["symbol"]: r for r in rows}
        return self._tickers

    def _load_premium(self) -> Dict[str, dict]:
        if self._premium is None:
            rows = self.ex.premium_index()
            self._premium = {r["symbol"]: r for r in rows}
        return self._premium

    def _btc_series(self) -> pd.Series:
        """按日期索引的 BTC 收盘价 —— Beta 对齐必须按日期，不能按数组位置。"""
        if self._btc_daily is None:
            df = self.ex.klines("BTCUSDT", "1d", limit=self.cfg.beta_lookback_days + 10)
            self._btc_daily = _date_indexed(df)
        return self._btc_daily

    # ------------------------------------------------------------ Meme 全集

    def fetch_market_caps(self) -> Dict[str, float]:
        """返回 {大写symbol: 市值}。CoinGecko 失败时返回空 dict。"""
        if self._mcap_cache is not None:
            return self._mcap_cache
        caps: Dict[str, float] = {}
        if self.cfg.use_coingecko:
            try:
                r = self.ex._session.get(
                    self.cfg.coingecko_url,
                    params={
                        "vs_currency": "usd",
                        "category": self.cfg.coingecko_category,
                        "order": "market_cap_desc",
                        "per_page": 250,
                        "page": 1,
                    },
                    timeout=20,
                )
                if r.status_code == 200:
                    for coin in r.json():
                        sym = str(coin.get("symbol", "")).upper()
                        mc = float(coin.get("market_cap") or 0)
                        if not sym or mc <= 0:
                            continue
                        # 同名币保留市值更高的那个
                        if sym not in caps or mc > caps[sym]:
                            caps[sym] = mc
                    log.info("CoinGecko 拉到 %d 个 Meme 币市值", len(caps))
                else:
                    log.warning("CoinGecko 返回 %s，改用内置白名单", r.status_code)
            except Exception as e:
                log.warning("CoinGecko 请求失败（%s），改用内置白名单", e)
        self._mcap_cache = caps
        return caps

    def alias_map(self) -> Dict[str, str]:
        """{归一化 base: 实际合约 symbol}，委托给交易所客户端统一处理。"""
        if self._alias is None:
            self._alias = self.ex.perpetual_alias_map(self.cfg.quote_asset)
        return self._alias

    def meme_universe(self) -> List[Tuple[str, str]]:
        """返回 [(合约 symbol, CoinGecko base)] 列表。"""
        alias = self.alias_map()
        caps = self.fetch_market_caps()

        if caps:
            uni = [(alias[s], s) for s in caps if s in alias]
            log.info("CoinGecko universe 与币安永续交集 %d 个", len(uni))
            if uni:
                return uni
        uni = []
        for b in FALLBACK_MEMES:
            if b in alias:
                uni.append((alias[b], b))
        log.info("使用内置白名单，命中 %d 个合约", len(uni))
        return uni

    # ------------------------------------------------------------ 主流程

    def screen(self, limit: Optional[int] = None, side: str = "SHORT") -> List[MemeCandidate]:
        """筛选候选标的。

        side 决定资金费率的偏好方向：
          SHORT -> 偏好费率为正（多头拥挤，空头收租）
          LONG  -> 偏好费率为负（空头拥挤，做多方收租）
        """
        cfg = self.cfg
        is_long = side.upper() == "LONG"
        perps = self.ex.perpetual_symbols(cfg.quote_asset)
        tickers = self._load_tickers()
        premium = self._load_premium()
        caps = self.fetch_market_caps()
        btc = self._btc_series()

        universe = self.meme_universe()
        cands: List[MemeCandidate] = []
        skipped: Dict[str, int] = {}
        self.filter_log: Dict[str, str] = {}   # 合约 symbol -> 淘汰原因（用于排查）

        def skip(code: str, symbol: str = "", detail: str = ""):
            skipped[code] = skipped.get(code, 0) + 1
            if symbol:
                self.filter_log[symbol] = f"{code}{('：' + detail) if detail else ''}"

        for symbol, cg_base in universe:
            meta = perps.get(symbol)
            if not meta:
                skip("无永续合约", symbol)
                continue

            onboard_days = (int(time.time() * 1000) - meta["onboard_date"]) / 86_400_000
            if onboard_days < cfg.min_onboard_days:
                skip("合约上线过新", symbol, f"{onboard_days:.0f}天")
                continue

            tk = tickers.get(symbol)
            if not tk:
                skip("无行情", symbol)
                continue
            qv = float(tk.get("quoteVolume", 0))
            if qv < cfg.min_quote_volume_24h:
                skip("成交额不足", symbol, f"${qv / 1e6:.1f}M<${cfg.min_quote_volume_24h / 1e6:.0f}M")
                continue

            mcap = caps.get(cg_base, 0.0)
            if caps and (mcap < cfg.min_market_cap or mcap > cfg.max_market_cap):
                skip("市值不在区间", symbol, f"${mcap / 1e6:.0f}M")
                continue

            prem = premium.get(symbol)
            if not prem:
                skip("无资金费率", symbol)
                continue
            funding = float(prem.get("lastFundingRate", 0.0))
            if abs(funding) > cfg.max_funding:
                skip("资金费率异常", symbol, f"{funding * 100:+.3f}%")
                continue

            c = MemeCandidate(
                symbol=symbol,
                base=cg_base,
                market_cap=mcap,
                quote_volume_24h=qv,
                funding_rate=funding,
                onboard_days=int(onboard_days),
                price=float(tk.get("lastPrice", 0.0)),
                change_24h=float(tk.get("priceChangePercent", 0.0)) / 100.0,
            )
            cands.append(c)

        # --- Beta + 点差/深度（逐个请求，成本可控）------------------------
        for c in cands:
            try:
                df = self.ex.klines(c.symbol, "1d", limit=cfg.beta_lookback_days + 10)
                series = _date_indexed(df)
                c.beta = beta_vs_bench(series, btc)
                c.downside_beta = downside_beta(series, btc, cfg.crash_quantile)
                c.change_7d = (
                    float(df["close"].iloc[-1] / df["close"].iloc[-8] - 1) if len(df) >= 8 else 0.0
                )
                if len(series) >= 4 and len(btc) >= 4:
                    asset_3d = float(series.iloc[-1] / series.iloc[-4] - 1.0)
                    btc_3d = float(btc.iloc[-1] / btc.iloc[-4] - 1.0)
                    c.rel_weakness = max(0.0, btc_3d - asset_3d)
            except Exception:
                c.beta = 0.0
                c.downside_beta = 0.0
            try:
                di = self.ex.depth_info(c.symbol)
                c.spread_bps = di.spread_bps
                c.ask_depth_usd = di.ask_depth_usd_05pct
                c.bid_depth_usd = di.bid_depth_usd_05pct
            except Exception:
                c.spread_bps = 999.0

            if c.beta < cfg.min_beta:
                c.warnings.append(f"常规 β={c.beta:.2f} < {cfg.min_beta}")
            if c.downside_beta < cfg.min_downside_beta:
                c.warnings.append(f"下跌 β={c.downside_beta:.2f} < {cfg.min_downside_beta}（暴跌放大不足）")
            if c.funding_rate <= -0.0005 and not is_long:
                c.warnings.append(
                    f"资金费率已为负({c.funding_rate * 100:+.4f}%)，空头拥挤，轧空风险高"
                )
            if c.funding_rate >= 0.0005 and is_long:
                c.warnings.append(
                    f"资金费率为正({c.funding_rate * 100:+.4f}%)，多头拥挤，做多要付费持仓"
                )
            if c.spread_bps > 50:
                c.warnings.append(f"点差 {c.spread_bps:.0f}bp 偏大")

        # --- 打分 --------------------------------------------------------
        # 门槛以「下跌 β」为准 —— 策略要的是暴跌时的放大倍数，不是全周期相关性
        viable = [
            c for c in cands
            if c.downside_beta >= cfg.min_downside_beta
            and c.beta >= cfg.min_beta
            and c.spread_bps <= 50
        ]
        if not viable:
            log.warning(
                "没有同时满足 下跌β>=%.1f、β>=%.1f 且点差<=50bp 的标的；按下跌β放宽重排",
                cfg.min_downside_beta, cfg.min_beta,
            )
            viable = sorted(cands, key=lambda x: x.downside_beta, reverse=True)[: limit or cfg.top_n]

        max_vol = max((c.quote_volume_24h for c in viable), default=1.0) or 1.0
        for c in viable:
            crowd = min(1.0, max(0.0, c.funding_rate / max(cfg.crowded_funding, 1e-9)))
            if is_long:
                # 做多：偏好费率为负（空头拥挤，做多方反而收租）
                crowd = min(1.0, max(0.0, -c.funding_rate / max(cfg.crowded_funding, 1e-9)))
            beta_s = min(1.0, max(0.0, (c.downside_beta - 0.8) / max(cfg.min_downside_beta - 0.8, 0.1)))
            liq_s = (c.quote_volume_24h / max_vol) ** 0.35
            mom_s = min(1.0, max(0.0, c.change_7d / 0.5))  # 前期涨得越多越拥挤
            weak_s = min(1.0, max(0.0, c.rel_weakness / 0.10))
            c.score = round(
                25 * crowd + 30 * beta_s + 20 * liq_s + 10 * mom_s + 15 * weak_s, 1
            )
            # 做空时：费率已转负 -> 空头拥挤，赔率变差，扣分
            # 做多时：费率为正 -> 多头拥挤，赔率变差，扣分（方向相反）
            adverse = c.funding_rate if is_long else -c.funding_rate
            if adverse > 0:
                penalty = min(0.45, adverse / 0.001 * 0.45)
                c.score = round(c.score * (1.0 - penalty), 1)
            if c.funding_rate >= cfg.crowded_funding:
                c.reasons.append(f"资金费率 {c.funding_rate * 100:+.4f}% 多头拥挤")
            if c.downside_beta >= cfg.min_downside_beta:
                c.reasons.append(f"下跌 β={c.downside_beta:.2f}（BTC 跌 1% 它跌 {c.downside_beta:.2f}%）")
            if c.quote_volume_24h >= cfg.min_quote_volume_24h * 5:
                c.reasons.append("流动性充足")
            if c.change_7d > 0.2:
                c.reasons.append(f"7日已涨 {c.change_7d * 100:.0f}%，回撤风险大")

        viable.sort(key=lambda x: x.score, reverse=True)
        out = viable[: (limit or cfg.top_n)]
        log.info("筛选完成：候选 %d / 过滤掉 %s", len(out),
                 {k: v for k, v in sorted(skipped.items(), key=lambda kv: -kv[1])})
        return out
