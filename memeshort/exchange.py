"""Binance USDT-M 合约客户端。

- 公开行情无需 Key；私有下单用 HMAC-SHA256 签名
- 支持代理与多镜像自动切换（国内直连 fapi.binance.com 常被拒）
- 内置简单的权重限速，避免触发 429
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from .utils import to_float

log = logging.getLogger("meme.exchange")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


class BinanceError(RuntimeError):
    def __init__(self, code: int, msg: str, url: str = ""):
        super().__init__(f"[{code}] {msg} ({url})")
        self.code = code
        self.msg = msg


@dataclass
class DepthInfo:
    bid: float
    ask: float
    spread_bps: float
    ask_depth_usd_05pct: float   # 卖盘 0.5% 深度（美元）
    bid_depth_usd_05pct: float   # 买盘 0.5% 深度（美元）


class BinanceFutures:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        proxy: str = "",
        testnet: bool = False,
        timeout: float = 12.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.base_urls = (
            ["https://testnet.binancefuture.com"] if testnet else
            ["https://fapi.binance.com",
             "https://fapi1.binance.com",
             "https://fapi2.binance.com",
             "https://fapi3.binance.com",
             "https://fapi4.binance.com"]
        )
        self._active = self.base_urls[0]
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "meme-short/0.1"})
        if proxy:
            self._session.proxies = {"http": proxy, "https": proxy}
        self._weight = 0.0
        self._weight_window = 0.0
        self._exchange_info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ 底层

    def _rate_limit(self, weight: float) -> None:
        """Binance 每分钟 2400 权重；这里做粗粒度节流。"""
        now = time.time()
        if now - self._weight_window > 60:
            self._weight_window = now
            self._weight = 0.0
        self._weight += weight
        if self._weight > 2000:
            sleep_for = 60 - (now - self._weight_window) + 0.5
            if sleep_for > 0:
                log.warning("触发限速阈值，等待 %.1fs", sleep_for)
                time.sleep(sleep_for)
            self._weight_window = time.time()
            self._weight = 0.0

    def _sign(self, params: Dict[str, Any]) -> str:
        qs = urlencode(params)
        return hmac.new(
            self.api_secret.encode(), qs.encode(), hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: float = 1.0,
        retries: int = 2,
    ) -> Any:
        params = dict(params or {})
        headers: Dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceError(-1, "缺少 API Key/Secret，无法进行私有请求")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            qs = urlencode(params)
            params["signature"] = hmac.new(
                self.api_secret.encode(), qs.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key

        last_err: Optional[Exception] = None
        urls = [self._active] + [u for u in self.base_urls if u != self._active]
        for attempt in range(retries + 1):
            for url in urls:
                try:
                    self._rate_limit(weight)
                    r = self._session.request(
                        method, url + path, params=params, headers=headers, timeout=self.timeout
                    )
                    if r.status_code in (418, 429):
                        # 服务端限速：优先遵守 Retry-After，否则按退避等待
                        retry_after = r.headers.get("Retry-After")
                        delay = float(retry_after) if (retry_after or "").isdigit() else 0.0
                        raise BinanceError(
                            r.status_code,
                            f"被限速 (HTTP {r.status_code})，需等待 {delay or 30:.0f}s "
                            f"{r.text[:100]}",
                            url,
                        )
                    if r.status_code == 451:
                        last_err = BinanceError(451, "地区限制，换镜像重试", url)
                        continue
                    if r.status_code >= 400:
                        body = r.text[:200]
                        raise BinanceError(
                            r.status_code,
                            f"HTTP {r.status_code}: {body}",
                            url,
                        )
                    self._active = url
                    try:
                        return r.json()
                    except ValueError:
                        # 非 JSON 响应：通常是 WAF / 验证页 / 地区限制页
                        raise BinanceError(
                            r.status_code,
                            f"响应不是 JSON（可能被 WAF 或地区限制拦截）："
                            f"{r.text[:160]!r}",
                            url,
                        )
                except BinanceError as e:
                    last_err = e
                    if e.code in (418, 429):
                        # 限速：换镜像没用，等待更久再试
                        m = re.search(r"需等待 (\d+(?:\.\d+)?)s", e.msg)
                        wait = float(m.group(1)) if m else 30.0
                        log.warning("触发服务端限速，等待 %.0fs 后重试", wait)
                        time.sleep(min(wait, 120.0))
                        break
                    continue
                except requests.RequestException as e:
                    last_err = e
                    continue
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
        if isinstance(last_err, BinanceError):
            raise last_err
        raise BinanceError(-2, f"请求失败: {last_err}")

    # ------------------------------------------------------------ 公开行情

    def ping(self) -> bool:
        try:
            self._request("GET", "/fapi/v1/ping", weight=1)
            return True
        except Exception as e:
            log.warning("ping 失败: %s", e)
            return False

    def server_time(self) -> int:
        return int(self._request("GET", "/fapi/v1/time", weight=1)["serverTime"])

    def klines(
        self, symbol: str, interval: str = "4h", limit: int = 500, start_time: Optional[int] = None
    ) -> pd.DataFrame:
        """返回 DataFrame: open_time(open) high low close volume quote_volume ..."""
        p: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time:
            p["startTime"] = start_time
        raw = self._request("GET", "/fapi/v1/klines", p, weight=2 if limit > 100 else 1)
        df = pd.DataFrame(raw, columns=KLINE_COLS)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = df[c].astype(float)
        # 注意：必须显式 int64。部分环境下 .astype(int) 会落到 int32，
        # 毫秒时间戳会被截断成负数（表现为 1969 年日期）。
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        return df

    def exchange_info(self, refresh: bool = False) -> Dict[str, Any]:
        if self._exchange_info is None or refresh:
            self._exchange_info = self._request("GET", "/fapi/v1/exchangeInfo", weight=1)
        return self._exchange_info

    def perpetual_symbols(self, quote: str = "USDT") -> Dict[str, Dict[str, Any]]:
        """返回 {symbol: {base, quote, onboardDate, status, ...}}，仅永续且可交易。"""
        info = self.exchange_info()
        out: Dict[str, Dict[str, Any]] = {}
        for s in info.get("symbols", []):
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != quote:
                continue
            out[s["symbol"]] = {
                "symbol": s["symbol"],
                "base": s.get("baseAsset", s["symbol"].replace(quote, "")),
                "quote": s.get("quoteAsset", quote),
                "onboard_date": int(s.get("onboardDate", 0)),
                "price_precision": int(s.get("pricePrecision", 8)),
                "quantity_precision": int(s.get("quantityPrecision", 4)),
                "filters": s.get("filters", []),
            }
        return out

    def perpetual_alias_map(self, quote: str = "USDT") -> Dict[str, str]:
        """{归一化 base: 合约 symbol}。

        币安对单位价格 < 1 的币用 1000x / 1000000x 前缀开合约：
            PEPE -> 1000PEPEUSDT,  SHIB -> 1000SHIBUSDT,  MOG -> 1000000MOGUSDT
        不做这层归一化会整体漏掉市值最高的一批 Meme 币。
        同名合约同时存在时（如 SHIBUSDT 与 1000SHIBUSDT），优先取不带前缀的。
        """
        import re

        perps = self.perpetual_symbols(quote)
        alias: Dict[str, str] = {}
        for sym, meta in perps.items():
            alias.setdefault(str(meta["base"]).upper(), sym)
        for sym, meta in perps.items():
            base = str(meta["base"]).upper()
            norm = re.sub(r"^\d+", "", base)
            if norm and norm != base:
                alias.setdefault(norm, sym)
        return alias

    def ticker_24h(self, symbol: Optional[str] = None) -> Any:
        p = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/ticker/24hr", p, weight=40 if symbol is None else 1)

    def all_tickers(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v1/ticker/24hr", weight=40)

    def premium_index(self, symbol: Optional[str] = None) -> Any:
        """资金费率。symbol=None 返回全市场列表。"""
        p = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/premiumIndex", p, weight=10 if symbol else 1)

    def funding_rate_history(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self._request(
            "GET", "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}, weight=1
        )

    def depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit}, weight=2)

    def mark_price(self, symbol: str) -> float:
        return to_float(self.premium_index(symbol)["markPrice"])

    def depth_info(self, symbol: str) -> DepthInfo:
        d = self.depth(symbol, 20)
        bids = [(to_float(p), to_float(q)) for p, q in d.get("bids", [])]
        asks = [(to_float(p), to_float(q)) for p, q in d.get("asks", [])]
        if not bids or not asks:
            return DepthInfo(0, 0, 1e9, 0, 0)
        bid, ask = bids[0][0], asks[0][0]
        mid = (bid + ask) / 2 or bid
        spread_bps = (ask - bid) / mid * 1e4
        ask_depth = sum(p * q for p, q in asks if p <= mid * 1.005)
        bid_depth = sum(p * q for p, q in bids if p >= mid * 0.995)
        return DepthInfo(bid, ask, spread_bps, ask_depth, bid_depth)

    def recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v1/trades", {"symbol": symbol, "limit": limit}, weight=5)

    # ------------------------------------------------------------ 私有

    def account(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True, weight=5)

    def positions(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v2/positionRisk", signed=True, weight=5)

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return self._request(
            "POST", "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": max(1, int(leverage))},
            signed=True, weight=1,
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        try:
            return self._request(
                "POST", "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": margin_type},
                signed=True, weight=1,
            )
        except BinanceError as e:
            # -4046 = No need to change margin type
            if "-4046" in e.msg or "No need to change" in e.msg:
                return {"ok": True}
            raise

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        time_in_force: str = "GTC",
        working_type: str = "MARK_PRICE",
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": f"{qty:.8f}".rstrip("0").rstrip("."),
        }
        if order_type in ("LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"):
            if price is not None:
                p["price"] = f"{price:.10f}".rstrip("0").rstrip(".")
        if order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            p["stopPrice"] = f"{stop_price:.10f}".rstrip("0").rstrip(".")
            p["workingType"] = working_type
        if reduce_only:
            p["reduceOnly"] = "true"
        if close_position:
            p["closePosition"] = "true"
            p.pop("quantity", None)
        if order_type == "LIMIT":
            p["timeInForce"] = time_in_force
        return self._request("POST", "/fapi/v1/order", p, signed=True, weight=1)

    def modify_order(
        self, symbol: str, order_id: int, quantity: float, price: float,
        stop_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "orderId": order_id,
            "symbol": symbol,
            "side": "SELL" if quantity > 0 else "BUY",
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
            "price": f"{price:.10f}".rstrip("0").rstrip("."),
        }
        if stop_price is not None:
            p["stopPrice"] = f"{stop_price:.10f}".rstrip("0").rstrip(".")
        return self._request("PUT", "/fapi/v1/order", p, signed=True, weight=1)

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        return self._request(
            "DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id},
            signed=True, weight=1,
        )

    def open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        p = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", p, signed=True, weight=1)

    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        return self._request(
            "GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id},
            signed=True, weight=1,
        )


def klines_to_df(raw: List[List[Any]]) -> pd.DataFrame:
    """给回测/离线用的裸 klines 转换。"""
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype(float)
    return df


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
