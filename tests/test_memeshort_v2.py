"""离线单元测试：验证 memeshort v2 的 7 处关键修复与新增择时/追踪止损逻辑。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from memeshort.broker import LiveBroker, PaperBroker
from memeshort.config import BrokerConfig, Config, CrashConfig, RiskConfig, ScreenerConfig
from memeshort.crash import score_at_event
from memeshort.entry import evaluate_rebound_reject_short
from memeshort.risk import TradePlan, build_plan, estimate_slippage
from memeshort.screener import MemeCandidate, MemeScreener
from memeshort.state import Position, StateStore


class TestMemeShortV2(unittest.TestCase):
    def test_01_screener_negative_funding_penalizes_short(self):
        """做空方向：负资金费率（高轧空风险）必须扣分而非加分。"""
        ex = MagicMock()
        ex.perpetual_symbols.return_value = {
            "POSUSDT": {"onboard_date": 0},
            "NEGUSDT": {"onboard_date": 0},
        }
        ex.perpetual_alias_map.return_value = {"POS": "POSUSDT", "NEG": "NEGUSDT"}
        ex.all_tickers.return_value = [
            {"symbol": "POSUSDT", "quoteVolume": "50000000", "lastPrice": "1.0", "priceChangePercent": "10"},
            {"symbol": "NEGUSDT", "quoteVolume": "50000000", "lastPrice": "1.0", "priceChangePercent": "10"},
        ]
        ex.premium_index.return_value = [
            {"symbol": "POSUSDT", "lastFundingRate": "0.0001"},
            {"symbol": "NEGUSDT", "lastFundingRate": "-0.0008"},
        ]
        # 构造相同高下跌 beta 的日线序列
        days = pd.date_range("2026-09-01", periods=70, freq="D", tz="UTC")
        btc_ret = np.where(np.arange(70) % 5 == 0, -0.05, 0.01)
        btc_close = 60000.0 * np.cumprod(1.0 + btc_ret)
        alt_close = 1.0 * np.cumprod(1.0 + btc_ret * 2.2)

        def fake_klines(sym, tf, limit=70):
            c = btc_close if sym == "BTCUSDT" else alt_close
            return pd.DataFrame({
                "open_time": (days.astype("int64") // 10**6).to_numpy(),
                "open": c, "high": c * 1.02, "low": c * 0.98, "close": c, "volume": 1000.0,
            })

        ex.klines.side_effect = fake_klines
        depth = MagicMock(spread_bps=5.0, ask_depth_usd_05pct=500000.0, bid_depth_usd_05pct=500000.0)
        ex.depth_info.return_value = depth

        cfg = ScreenerConfig(use_coingecko=False, min_quote_volume_24h=1e6, min_beta=0.5, min_downside_beta=1.2)
        scr = MemeScreener(ex, cfg)
        scr.meme_universe = lambda: [("POSUSDT", "POS"), ("NEGUSDT", "NEG")]
        res = scr.screen(side="SHORT")
        by_sym = {c.symbol: c for c in res}
        self.assertGreater(by_sym["POSUSDT"].score, by_sym["NEGUSDT"].score)
        self.assertEqual(by_sym["POSUSDT"].bid_depth_usd, 500000.0)

    def test_02_paper_broker_slippage_and_partial_tp_zero_residual(self):
        """TP 按目标价成交且两档 0.5+0.5 完全清仓；TIME/STOP 按不利方向滑点成交。"""
        ex = MagicMock()
        bcfg = BrokerConfig(taker_fee=0.0005, slippage_pct=0.003)
        rcfg = RiskConfig(move_stop_to_breakeven=True, trailing_activate_pct=0.50)
        broker = PaperBroker(ex, bcfg, rcfg)

        plan = TradePlan(
            symbol="WIFUSDT", side="SHORT", qty=100.0, entry=1.0,
            stop_loss=1.08, take_profits=[(0.85, 0.5), (0.70, 0.5)],
            leverage=1.0, risk_amount=8.0, notional=100.0, slippage=0.003,
        )
        pos = broker.open_position(plan)
        assert pos is not None

        # 触发 TP1：应平掉 50.0，成交价=0.85（不送有利滑点），止损移至含费保本位 < 1.0
        acts1 = broker.manage(pos, price=0.84, funding=0.0)
        self.assertEqual(acts1[0].kind, "TP")
        self.assertAlmostEqual(acts1[0].price, 0.85, places=8)
        self.assertAlmostEqual(acts1[0].qty, 50.0, places=8)
        self.assertAlmostEqual(pos.remaining_qty, 50.0, places=8)
        self.assertTrue(pos.stop_moved_to_breakeven)
        self.assertLess(pos.stop_loss, pos.entry)

        # 触发 TP2：应平掉剩余全部 50.0，零残留
        acts2 = broker.manage(pos, price=0.69, funding=0.0)
        self.assertEqual(acts2[0].kind, "TP")
        self.assertAlmostEqual(acts2[0].qty, 50.0, places=8)
        self.assertEqual(pos.remaining_qty, 0.0)
        self.assertEqual(pos.status, "CLOSED")

        # 测试 TIME 市价平空：滑点应向不利方向（更高价买回）
        pos2 = broker.open_position(plan)
        assert pos2 is not None
        act_time = broker.close(pos2, "TIME", price=0.95, fraction=1.0)
        self.assertGreater(act_time.price, 0.95)

    def test_03_trailing_stop_and_live_update_stop(self):
        """浮盈达到阈值后启动追踪止损，且 LiveBroker 自动撤旧单重挂新止损单。"""
        ex = MagicMock()
        ex.place_order.return_value = {"orderId": 9999}
        bcfg = BrokerConfig(taker_fee=0.0005, slippage_pct=0.003)
        rcfg = RiskConfig(trailing_activate_pct=0.08, trailing_callback_pct=0.04)
        broker = LiveBroker(ex, bcfg, rcfg)

        pos = Position(
            symbol="1000PEPEUSDT", side="SHORT", qty=1000.0, remaining_qty=1000.0,
            entry=1.0, stop_loss=1.06, take_profits=[[0.80, 0.5, 0], [0.70, 0.5, 0]],
            order_ids={"stop": 1111}, mode="LIVE",
        )
        # 价格跌至 0.90（浮盈 10% >= 8%），追踪止损应下移至 0.90 * 1.04 = 0.936
        acts = broker.manage(pos, price=0.90, funding=0.0)
        self.assertTrue(any(a.kind == "BREAKEVEN" for a in acts))
        self.assertAlmostEqual(pos.stop_loss, 0.936, places=6)
        ex.cancel_order.assert_called_once_with("1000PEPEUSDT", 1111)
        ex.place_order.assert_called_once()
        self.assertEqual(pos.order_ids["stop"], 9999)

    def test_04_score_at_event_includes_t0_intraday_bars(self):
        """score_at_event 必须包含 T0 当天已完成的 4H K 线（<= T0 20:00 UTC）。"""
        t_4h = pd.date_range("2026-10-01 00:00", periods=300, freq="4h", tz="UTC")
        t_1d = pd.date_range("2026-10-01", periods=50, freq="D", tz="UTC")
        df_i = pd.DataFrame({
            "open_time": (t_4h.astype("int64") // 10**6).to_numpy(),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
        })
        # 在 T0 (2026-11-10) 4H 暴跌破位
        t0_mask = t_4h.date == pd.Timestamp("2026-11-10").date()
        df_i.loc[t0_mask, ["open", "high", "low", "close", "volume"]] = [96.0, 96.5, 85.0, 86.0, 5000.0]

        df_d = pd.DataFrame({
            "open_time": (t_1d.astype("int64") // 10**6).to_numpy(),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10000.0,
        })
        df_d.loc[t_1d.date == pd.Timestamp("2026-11-10").date(), ["open", "high", "low", "close", "volume"]] = [
            100.0, 100.0, 85.0, 86.0, 30000.0
        ]

        score, reasons = score_at_event(df_i, df_d, "2026-11-10", CrashConfig())
        self.assertGreater(score, 40.0)
        self.assertTrue(any("跌破" in r for r in reasons))

    def test_05_rollover_day_resets_halted(self):
        """跨日重置必须解除日内熔断标志 halted。"""
        store = StateStore("/tmp/test_memeshort_state")
        acct = {"day": "2020-01-01", "realized_pnl_today": -600.0, "halted": True}
        out = store.rollover_day(acct)
        self.assertFalse(out["halted"])
        self.assertEqual(out["realized_pnl_today"], 0.0)

    def test_06_slippage_uses_bid_depth_for_short_and_net_rr(self):
        """做空滑点应基于买盘深度 bid_depth_usd，且 build_plan 校验扣费后的净 RR。"""
        cand = MemeCandidate(
            symbol="DOGSUSDT", base="DOGS", price=1.0, spread_bps=10.0,
            ask_depth_usd=1_000_000.0, bid_depth_usd=500.0,
        )
        bcfg = BrokerConfig(slippage_pct=0.003, slippage_depth_factor=0.5, taker_fee=0.0005)
        slip_short = estimate_slippage(cand, 2000.0, bcfg, side="SHORT")
        slip_long = estimate_slippage(cand, 2000.0, bcfg, side="LONG")
        self.assertGreater(slip_short, slip_long)

    def test_07_entry_rebound_reject_short(self):
        """破位 -> 缩量反弹 -> 转弱阴线跌破前低时，触发 rebound_reject 信号。"""
        n = 45
        closes = np.full(n, 1.00)
        highs = np.full(n, 1.01)
        lows = np.full(n, 0.99)
        opens = np.full(n, 1.00)
        vols = np.full(n, 1000.0)

        # 构造破位下跌
        closes[35:39] = [0.96, 0.93, 0.91, 0.90]
        opens[35:39] = [0.99, 0.96, 0.93, 0.91]
        highs[35:39] = [0.995, 0.965, 0.935, 0.915]
        lows[35:39] = [0.955, 0.925, 0.905, 0.895]
        vols[35:39] = 2500.0

        # 构造 3 根缩量弱反弹（仍在 EMA20 下方）
        closes[39:42] = [0.91, 0.92, 0.922]
        opens[39:42] = [0.90, 0.91, 0.92]
        highs[39:42] = [0.915, 0.925, 0.928]
        lows[39:42] = [0.898, 0.908, 0.915]
        vols[39:42] = 900.0

        # 第 43 根（已收盘，索引 42）转阴并跌破前一根低点 0.915
        opens[42] = 0.922
        highs[42] = 0.924
        lows[42] = 0.905
        closes[42] = 0.908
        vols[42] = 1600.0

        # 索引 43..44 作为未收盘尾部（drop_last_incomplete=True 会丢弃最后一个）
        df = pd.DataFrame({
            "open": opens[:44], "high": highs[:44], "low": lows[:44],
            "close": closes[:44], "volume": vols[:44],
        })
        sig = evaluate_rebound_reject_short(df, ema_len=20, lookback_bars=16, atr_buffer_mult=0.35)
        self.assertTrue(sig.ready, msg=sig.reason)
        self.assertGreater(sig.structure_stop, 0.928)


if __name__ == "__main__":
    unittest.main()
