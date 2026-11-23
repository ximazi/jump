"""通知：终端 + 日志 + Telegram + 企业微信/钉钉 Webhook。

任何通道失败都只记录日志，绝不让通知异常打断交易主循环。
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from .config import NotifyConfig

log = logging.getLogger("meme.notify")

LEVEL_ICON = {
    "INFO": "ℹ",
    "WARN": "⚠",
    "WARNING": "⚠",
    "ERROR": "✖",
    "SUCCESS": "✔",
    "CRASH": "🔴",
    "SQUEEZE": "🧨",
}


class Notifier:
    def __init__(self, cfg: NotifyConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or log
        self._session = requests.Session()

    # -------------------------------------------------- 对外接口

    def send(self, title: str, message: str = "", level: str = "INFO") -> None:
        icon = LEVEL_ICON.get(level, "ℹ")
        text = f"{icon} {title}" + (f"\n{message}" if message else "")

        # 终端
        if self.cfg.console:
            if level in ("ERROR", "CRITICAL"):
                self.log.error("%s", text.replace("\n", "\n    "))
            elif level in ("WARNING", "WARN"):
                self.log.warning("%s", text.replace("\n", "\n    "))
            else:
                self.log.info("%s", text.replace("\n", "\n    "))

        # 文件（走 logging 即可）
        if self.cfg.log_file:
            self.log.info("[事件] %s | %s", title, message.replace("\n", " / "))

        # Telegram
        if self.cfg.telegram and self.cfg.telegram_token and self.cfg.telegram_chat_id:
            self._send_telegram(f"[{level}] {title}\n{message}")

        # Webhook
        if self.cfg.webhook and self.cfg.webhook_url:
            self._send_webhook(f"[{level}] {title}\n{message}")

    # -------------------------------------------------- 具体通道

    def _send_telegram(self, text: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage"
            r = self._session.post(
                url,
                json={
                    "chat_id": self.cfg.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if r.status_code != 200:
                self.log.error("Telegram 推送失败 %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            self.log.error("Telegram 推送异常: %s", e)

    def _send_webhook(self, text: str) -> None:
        try:
            if self.cfg.webhook_type == "dingtalk":
                payload = {"msgtype": "markdown", "markdown": {"text": text}}
            else:  # 企业微信
                payload = {"msgtype": "text", "text": {"content": text}}
            r = self._session.post(self.cfg.webhook_url, json=payload, timeout=10)
            if r.status_code != 200:
                self.log.error("Webhook 推送失败 %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            self.log.error("Webhook 推送异常: %s", e)

    # -------------------------------------------------- 便捷事件

    def crash_detected(self, btc_score: float, eth_score: float, reasons: list) -> None:
        self.send(
            "🚨 检测到大盘暴跌信号",
            f"BTC 评分 {btc_score:.0f} / ETH 评分 {eth_score:.0f}\n"
            + "\n".join(f"  · {r}" for r in reasons[:8]),
            level="CRASH",
        )

    def order_opened(self, plan, symbol: str) -> None:
        self.send(
            f"做空开仓 {symbol}",
            f"数量 {plan.qty:.6g} @ {plan.entry:.8g}\n"
            f"止损 {plan.stop_loss:.8g} | 止盈 {plan.tp1:.8g} / {plan.tp2:.8g}\n"
            f"杠杆 {plan.leverage:.2f}x | 风险 ${plan.risk_amount:.2f} | RR {plan.rr:.2f}",
            level="SUCCESS",
        )

    def position_closed(
        self, symbol: str, reason: str, pnl: float, pnl_pct: float,
        level: Optional[str] = None,
    ) -> None:
        if level is None:
            level = "SUCCESS" if pnl >= 0 else "ERROR"
        self.send(
            f"平仓 {symbol} ({reason})",
            f"实现盈亏 ${pnl:+.2f} ({pnl_pct:+.2%})",
            level=level,
        )

    def squeeze_warning(self, symbol: str, funding: float) -> None:
        self.send(
            f"🧨 轧空预警 {symbol}",
            f"资金费率 {funding * 100:+.4f}% 已深度转负，空头拥挤，建议减仓/全平",
            level="SQUEEZE",
        )

    def risk_halt(self, reason: str) -> None:
        self.send("🛑 风控熔断", reason, level="ERROR")
