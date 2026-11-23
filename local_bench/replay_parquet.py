#!/usr/bin/env python
"""本地 5m Parquet 离线回放与漏斗诊断脚本。

直接读取本地 5m K 线 Parquet 文件（字段：ts, open, high, low, close, volume, quote_volume），
将 5m 聚合为 15m 信号周期，对比不同入场时机、止损、追踪止盈与最大持仓时长，
输出：
  - route_attempts.csv（所有候选与拒绝原因明细）
  - trades.csv（已结算交易明细）
  - summary.json（汇总指标与拒绝漏斗统计）

用法示例：
    python local_bench/replay_parquet.py --data-dir . --scenario local_bench/scenarios/rebound_reject.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def _load_scenario_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        out: Dict[str, Any] = {}
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                try:
                    out[k] = int(v) if "." not in v else float(v)
                except ValueError:
                    out[k] = v
        return out

# 允许直接从任意目录运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memeshort.entry import evaluate_rebound_reject_short
from memeshort.indicators import atr, ema


def resample_5m_to_15m(df5: pd.DataFrame) -> pd.DataFrame:
    df = df5.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    agg = df.resample("15min", label="left", closed="left").agg(
        ts=("ts", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
    ).dropna()
    return agg.reset_index(drop=True)


def run_symbol_replay(symbol: str, df5: pd.DataFrame, sc: Dict[str, Any]) -> Dict[str, Any]:
    df5 = df5.sort_values("ts").reset_index(drop=True)
    df15 = resample_5m_to_15m(df5)
    df15["ema_fast"] = ema(df15["close"], int(sc.get("ema_len", 20)))
    df15["atr14"] = atr(df15, 14)

    ts5 = df5["ts"].to_numpy()
    mode = sc.get("entry_mode", "rebound_reject")
    lookback_bars = int(sc.get("lookback_bars", 16))
    atr_buf = float(sc.get("atr_buffer_mult", 0.35))
    min_stop_pct = float(sc.get("min_stop_pct", 0.012))
    max_stop_pct = float(sc.get("max_stop_pct", 0.08))
    tp_pct = float(sc.get("tp_pct", 0.04))
    min_net_rr = float(sc.get("min_net_rr", 1.2))
    rt_cost = float(sc.get("round_trip_cost", 0.002))       # 20 bps 默认往返成本
    max_hold_min = int(sc.get("max_hold_minutes", 240))
    trail_act = float(sc.get("trailing_activate_pct", 0.02))
    trail_cb = float(sc.get("trailing_callback_pct", 0.01))
    notional = float(sc.get("notional_usd", 200.0))
    cooldown_bars = int(sc.get("cooldown_15m_bars", 8))

    attempts: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    next_allowed_idx = max(lookback_bars + 10, 35)

    for i in range(next_allowed_idx, len(df15) - 1):
        if i < next_allowed_idx:
            continue

        row = df15.iloc[i]
        # 15m 破位检测（4h = 16 根 15m）
        prior_low = float(df15["low"].iloc[i - lookback_bars:i].min())
        broke_4h = float(row["close"]) < prior_low
        if not broke_4h and mode == "immediate":
            continue

        # 入场过滤
        if mode == "rebound_reject":
            sig = evaluate_rebound_reject_short(
                df15.iloc[: i + 1],
                ema_len=int(sc.get("ema_len", 20)),
                lookback_bars=lookback_bars,
                atr_buffer_mult=atr_buf,
                drop_last_incomplete=False,
            )
            if not sig.ready:
                continue
            stop_raw = sig.structure_stop
            entry_reason = sig.reason
        else:
            atr_v = float(row["atr14"]) if pd.notna(row["atr14"]) else 0.0
            stop_raw = float(df15["high"].iloc[i - 4:i + 1].max()) + atr_buf * atr_v
            entry_reason = "immediate_4h_break"

        # 在 15m 收盘后的第一根 5m 开盘价模拟成交（无未来函数）
        trigger_close_ms = int(row["ts"]) + 15 * 60_000
        fill_pos = int(np.searchsorted(ts5, trigger_close_ms, side="left"))
        if fill_pos >= len(df5):
            continue

        entry_price = float(df5.iloc[fill_pos]["open"])
        fill_ts = int(df5.iloc[fill_pos]["ts"])

        # 夹紧止损并计算目标价
        stop_price = min(max(stop_raw, entry_price * (1.0 + min_stop_pct)), entry_price * (1.0 + max_stop_pct))
        target_price = entry_price * (1.0 - tp_pct)

        if not (target_price < entry_price < stop_price):
            attempts.append({
                "symbol": symbol, "fill_ts": fill_ts, "entry": entry_price,
                "stop": stop_price, "target": target_price, "status": "REJECTED",
                "reason": "INVALID_SHORT_LEVELS",
            })
            continue

        gross_reward = (entry_price - target_price) / entry_price
        gross_risk = (stop_price - entry_price) / entry_price
        net_rr = max(0.0, gross_reward - rt_cost) / max(1e-9, gross_risk + rt_cost)
        if net_rr < min_net_rr:
            attempts.append({
                "symbol": symbol, "fill_ts": fill_ts, "entry": entry_price,
                "stop": stop_price, "target": target_price, "net_rr": round(net_rr, 4),
                "status": "REJECTED", "reason": "INSUFFICIENT_NET_RR",
            })
            continue

        attempts.append({
            "symbol": symbol, "fill_ts": fill_ts, "entry": entry_price,
            "stop": stop_price, "target": target_price, "net_rr": round(net_rr, 4),
            "status": "EXECUTED", "reason": entry_reason,
        })

        # 逐 5m K 线模拟持仓（最坏情况优先：先判止损再判止盈）
        max_bars_5m = max(1, max_hold_min // 5)
        end_pos = min(len(df5) - 1, fill_pos + max_bars_5m - 1)
        cur_stop = stop_price
        best_low = entry_price
        exit_price = float(df5.iloc[end_pos]["close"])
        exit_reason = "TIME"
        exit_ts = int(df5.iloc[end_pos]["ts"])

        for k in range(fill_pos, end_pos + 1):
            b = df5.iloc[k]
            bh, bl = float(b["high"]), float(b["low"])
            if bh >= cur_stop:
                exit_price = cur_stop
                exit_reason = "TRAIL_STOP" if cur_stop < stop_price else "STOP"
                exit_ts = int(b["ts"])
                break
            if bl <= target_price:
                exit_price = target_price
                exit_reason = "TP"
                exit_ts = int(b["ts"])
                break
            if bl < best_low:
                best_low = bl
                peak_profit = (entry_price - best_low) / entry_price
                if trail_act > 0 and peak_profit >= trail_act:
                    cand_stop = best_low * (1.0 + trail_cb)
                    if cand_stop < cur_stop:
                        cur_stop = cand_stop

        gross_ret = (entry_price - exit_price) / entry_price
        net_ret = gross_ret - rt_cost
        trades.append({
            "symbol": symbol,
            "fill_time": pd.to_datetime(fill_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
            "exit_time": pd.to_datetime(exit_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "net_rr": round(net_rr, 4),
            "gross_return": round(gross_ret, 6),
            "net_return": round(net_ret, 6),
            "pnl_usd": round(notional * net_ret, 4),
        })
        next_allowed_idx = i + cooldown_bars

    return {"attempts": attempts, "trades": trades}


def main() -> int:
    ap = argparse.ArgumentParser(description="本地 5m Parquet 离线回放与漏斗审计")
    ap.add_argument("--data-dir", required=True, help="包含 *.parquet 文件的目录")
    ap.add_argument("--scenario", required=True, help="实验场景 YAML 路径")
    ap.add_argument("--out-dir", default="local_bench/outputs/latest", help="输出目录")
    args = ap.parse_args()

    sc = _load_scenario_yaml(args.scenario)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not files:
        print(f"未在 {args.data_dir} 找到 *.parquet 文件", file=sys.stderr)
        return 1

    all_attempts: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    for fp in files:
        sym = os.path.splitext(os.path.basename(fp))[0]
        df5 = pd.read_parquet(fp)
        res = run_symbol_replay(sym, df5, sc)
        all_attempts.extend(res["attempts"])
        all_trades.extend(res["trades"])

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(all_attempts).to_csv(os.path.join(args.out_dir, "route_attempts.csv"), index=False)
    df_t = pd.DataFrame(all_trades)
    df_t.to_csv(os.path.join(args.out_dir, "trades.csv"), index=False)

    rej_counts: Dict[str, int] = {}
    for a in all_attempts:
        if a["status"] == "REJECTED":
            rej_counts[a["reason"]] = rej_counts.get(a["reason"], 0) + 1

    summary = {
        "scenario": sc.get("name", os.path.basename(args.scenario)),
        "symbols_tested": len(files),
        "route_attempts": len(all_attempts),
        "executed_trades": len(all_trades),
        "rejections": rej_counts,
        "win_rate": round(float((df_t["net_return"] > 0).mean()), 4) if len(df_t) else 0.0,
        "mean_net_return": round(float(df_t["net_return"].mean()), 6) if len(df_t) else 0.0,
        "total_pnl_usd": round(float(df_t["pnl_usd"].sum()), 4) if len(df_t) else 0.0,
        "exit_reasons": df_t["exit_reason"].value_counts().to_dict() if len(df_t) else {},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
