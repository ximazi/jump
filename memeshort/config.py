"""配置：全部可用 config.yaml 覆盖，环境变量可注入密钥。

密钥永远从环境变量读（MEME_API_KEY / MEME_API_SECRET），不写进 yaml。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List


def _coerce(current: Any, value: Any) -> Any:
    """把 YAML 里被解析成字符串的数值转回来。

    PyYAML 遵循 YAML 1.1：'5.0e8' 这种没有指数符号的写法会被当成字符串，
    而 '5.0e+8' 才是浮点。这里按默认值的类型做一次转换。
    """
    if isinstance(value, str) and isinstance(current, (int, float)) and not isinstance(current, bool):
        try:
            return type(current)(value)
        except (TypeError, ValueError):
            return value
    return value


def _apply(dc, data: Dict[str, Any]):
    """把 dict 递归灌进 dataclass，未知键忽略。"""
    if not isinstance(data, dict):
        return dc
    valid = {f.name: f for f in fields(dc)}
    for k, v in data.items():
        if k not in valid:
            continue
        cur = getattr(dc, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _apply(cur, v)
        elif isinstance(cur, list) and isinstance(v, list):
            setattr(dc, k, v)
        else:
            setattr(dc, k, _coerce(cur, v))
    return dc


@dataclass
class CrashConfig:
    """暴跌信号检测参数。"""

    btc_symbol: str = "BTCUSDT"
    eth_symbol: str = "ETHUSDT"
    tf_intraday: str = "4h"
    tf_daily: str = "1d"

    fast_ema: int = 20
    slow_ema: int = 50
    swing_lookback: int = 30          # N 根内的区间下沿
    drawdown_threshold: float = 0.05  # 回撤 5% 起计入评分
    volume_spike: float = 1.5         # 放量倍数（相对 20 根均量）

    # 各因子权重（满权重合计 12.5 -> 归一化到 100 分）
    w_ema_cross: float = 2.5
    w_swing_break: float = 3.0
    w_drawdown: float = 1.5
    w_volume: float = 1.5
    w_weak_bounce: float = 1.0
    w_daily_break: float = 2.0
    w_funding_crowd: float = 1.0

    # 触发门槛
    # 默认值来自 730 天回测的阈值敏感性扫描：50 分是唯一在「全样本 / 截尾均值 /
    # 前半段 / 后半段」四项检验中方向一致为正的档位。60 分（等确认）各档位
    # 一致为负 —— 确认过的破位往往是下跌衰竭点，反而是反弹起点。
    # 样本量只有 20~30 次，务必用 backtest --sweep 在自己的区间上复验后再调参。
    btc_min_score: float = 50.0       # BTC 最低分
    eth_min_score: float = 40.0       # ETH 确认分
    btc_solo_score: float = 78.0      # BTC 单边够强的独立触发分
    funding_crowd_level: float = 0.0003   # 0.03% 视为多头拥挤

    # ---- 放量杀跌（事件型信号）------------------------------------
    # 回测发现：事件型信号（刚发生放量杀跌）远好于状态型信号（价格在低位、
    # 指标空头）。近 730 天回测中，「2 天内出现过放量杀跌」条件下做空 Meme
    # 篮子 7 日 +6.37%（胜率 70%）、14 日 +9.13%（胜率 88%），前后半段一致；
    # 无条件基准只有 +0.92% / +2.46%。
    signal_mode: str = "event"        # event | state | both
    climax_volume_ratio: float = 1.8  # 量比阈值（相对 20 日均量）
    climax_drop_pct: float = -0.04    # 当日跌幅阈值
    # 上限：量比太极端反而是恐慌见底，不该做空。
    # 回测（730 天，15 个事件）按量比分档的 7 日做空收益：
    #   1.8-2.2x  n=10  +10.17%  胜率 90%
    #   2.2-3.0x  n=4   +3.68%  胜率 50%
    #   >3.0x     n=1   -4.63%  胜率  0%
    # 样本极少（顶端只有 1 个），但方向明确且不对称：跳过一个负期望的交易
    # 没有代价，所以宁可漏也不错。
    climax_max_volume_ratio: float = 3.0
    climax_max_age_days: int = 2      # 事件有效期：几天内仍算「新鲜」
    climax_confirm_score: float = 25.0  # 事件触发时，状态分至少要达到多少
    climax_require_eth: bool = False  # 事件信号是否也要 ETH 确认
    climax_min_hold_days: int = 3     # 事件触发后的最短持有天数，避免当天进当天出


@dataclass
class BounceConfig:
    """先做反弹、再做空的两段式执行。

    放量杀跌后的形态（回测 n=15）：
        T0+2   反弹（中位 +1.75%，上涨概率 67%）
        T0+5   第二段下跌开始（中位 -2.20%）
        T0+7   主跌段（中位 -9.46%，上涨概率仅 27%）
    先多 2 天再空 5 天：+9.83%、胜率 87%，优于直接空 7 天的 +7.74% / 73%。
    """

    enabled: bool = True
    long_days: int = 2                # T0+1 ~ T0+2 做多
    short_start_offset: int = 3       # T0+3 开始做空
    short_end_offset: int = 7         # T0+7 之后平空（或触发止盈止损）
    long_risk_per_trade: float = 0.005   # 反弹腿单笔风险（更小，因为要接刀）
    long_max_leverage: float = 1.5    # 反弹腿杠杆上限（不加杠杆）
    # 「反弹已经走完就别追」保护：进场价相对事件收盘价（T0 收盘）的涨幅超过
    # 这个比例就跳过反弹腿，直接等做空腿。
    # 依据：回测中从 T0+1 收盘进场的反弹腿中位数是 -0.73%、胜率 47%，
    # 而 T0->T0+1 涨幅超过 2% 的样本反弹腿胜率为 0%（n=1，样本极小）。
    # 反正做空段才是主要贡献（+7.8% vs 反弹腿 +0.5%），跳过几乎不损失期望值，
    # 却能避开在反弹顶部接刀。
    max_entry_drift: float = 0.02


@dataclass
class ScreenerConfig:
    """Meme 币筛选参数。"""

    min_market_cap: float = 5e7         # 5 亿美元以下不做空（庄家易拉盘）
    max_market_cap: float = 5e10        # 500 亿美元
    min_quote_volume_24h: float = 5e6   # 24h 成交额下限，防滑点（按账户规模调整）（按账户规模调整）
    min_beta: float = 1.0               # 相对 BTC 的常规回归 beta 下限
    min_downside_beta: float = 1.5      # 下跌 beta：暴跌日的跌幅放大倍数（策略核心门槛）
    crash_quantile: float = 0.15        # 取 BTC 最差 15% 的日子算下跌 beta
    beta_lookback_days: int = 60
    crowded_funding: float = 0.0005     # 0.05% 视为多头杠杆拥挤（优先空）
    max_funding: float = 0.003          # 超过 0.3% 认为是异常/新币，排除
    min_onboard_days: int = 7           # 合约上线不足 7 天的新币排除
    top_n: int = 8                      # 最终候选数量
    use_coingecko: bool = True
    coingecko_url: str = "https://api.coingecko.com/api/v3/coins/markets"
    coingecko_category: str = "meme-token"
    quote_asset: str = "USDT"


@dataclass
class RiskConfig:
    """仓位与风控参数。"""

    account_equity: float = 10_000.0     # 纸面账户初始权益
    risk_per_trade: float = 0.01         # 单笔风险占权益比例
    max_leverage: float = 2.0            # 杠杆上限（用户建议 2~3，绝不超过 5）
    hard_leverage_cap: float = 5.0
    max_position_pct: float = 0.20       # 单个标的最大名义敞口占权益
    max_total_notional_pct: float = 1.0  # 总名义敞口占权益（杠杆意义上）
    max_positions: int = 4               # 最大同时持仓数
    daily_loss_limit_pct: float = 0.05   # 单日亏损达 5% 熔断，停止开新仓

    stop_atr_mult: float = 1.5           # ATR 止损倍数
    min_stop_pct: float = 0.015          # 最小止损 1.5%
    max_stop_pct: float = 0.12           # 最大止损 12%
    tp1_pct: float = 0.15                # 第一档止盈 +15%
    tp1_fraction: float = 0.5            # 平掉一半
    tp2_pct: float = 0.30                # 第二档止盈 +30%
    tp2_fraction: float = 0.5            # 平掉剩余
    move_stop_to_breakeven: bool = True  # 第一档止盈后止损移到成本价

    trailing_activate_pct: float = 0.08  # 浮盈达到该比例后启动追踪止损
    trailing_callback_pct: float = 0.04  # 自最佳浮盈回撤该比例触发追踪止损

    squeeze_funding: float = -0.001      # 资金费率深度负值 -> 轧空预警，全平
    min_rr: float = 1.5                  # 最小净盈亏比（含往返手续费），低于则放弃


@dataclass
class EntryConfig:
    """盘中做空入场择时参数。"""

    mode: str = "rebound_reject"         # rebound_reject（破位后弱反弹受阻再空）| immediate
    timeframe: str = "15m"               # 盘中择时周期
    ema_len: int = 20                    # 趋势均线周期
    lookback_bars: int = 16              # 破位结构回看根数
    atr_buffer_mult: float = 0.35        # 局部反弹高点上方 ATR 缓冲倍数


@dataclass
class BrokerConfig:
    """成交与执行参数。"""

    live: bool = False                   # 默认纸面
    taker_fee: float = 0.0005            # 0.05% taker
    maker_fee: float = 0.0002            # 0.02% maker
    slippage_pct: float = 0.003          # 基准滑点 0.3%
    slippage_depth_factor: float = 0.5   # 盘口深度不足时的额外惩罚系数
    poll_seconds: int = 20               # 持仓监控轮询间隔
    max_spread_bps: float = 40.0         # 点差超过 40bp 视为流动性差，跳过


@dataclass
class NotifyConfig:
    """通知配置。"""

    console: bool = True
    log_file: bool = True
    log_dir: str = "logs"
    telegram: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""
    webhook: bool = False
    webhook_url: str = ""
    webhook_type: str = "wechat"         # wechat | dingtalk
    level: str = "INFO"


@dataclass
class Config:
    exchange: str = "binance"
    base_url: str = "https://fapi.binance.com"
    fallback_urls: List[str] = field(
        default_factory=lambda: [
            "https://fapi1.binance.com",
            "https://fapi2.binance.com",
            "https://fapi3.binance.com",
            "https://fapi4.binance.com",
        ]
    )
    proxy: str = ""                      # 例如 http://127.0.0.1:7890
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    min_free_margin_pct: float = 0.30    # 剩余保证金低于 30% 强制减仓

    crash: CrashConfig = field(default_factory=CrashConfig)
    bounce: BounceConfig = field(default_factory=BounceConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)

    # ---------------------------------------------------------- 加载

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        cfg = cls()
        if os.path.exists(path):
            import yaml

            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            _apply(cfg, raw)
        # 环境变量覆盖（密钥绝不入 yaml）
        cfg.api_key = os.environ.get("MEME_API_KEY", cfg.api_key)
        cfg.api_secret = os.environ.get("MEME_API_SECRET", cfg.api_secret)
        if not cfg.proxy:
            cfg.proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        cfg.notify.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", cfg.notify.telegram_token)
        cfg.notify.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", cfg.notify.telegram_chat_id)
        cfg.notify.webhook_url = os.environ.get("WEBHOOK_URL", cfg.notify.webhook_url)

        # 安全护栏：杠杆硬上限
        cfg.risk.max_leverage = min(cfg.risk.max_leverage, cfg.risk.hard_leverage_cap)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["api_key"] = "***" if d.get("api_key") else ""
        d["api_secret"] = "***" if d.get("api_secret") else ""
        return d
