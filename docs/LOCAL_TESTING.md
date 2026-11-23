# 本地测试、离线回放与提交流程指南

本文档用于规范 `memeshort` v2 在本地的反复验证流程，确保**代码与实验配置可追溯入库**，而**大体积原始 Parquet 与临时运行输出不污染 Git 历史**。

## 1. 目录约定

| 目录 / 文件 | 用途 | 是否提交 GitHub |
| --- | --- | --- |
| `memeshort/` | 核心策略、风控、选币、入场择时与执行代码 | 是 |
| `tests/` | 离线单元测试（不联网、秒级回归） | 是 |
| `local_bench/replay_parquet.py` | 本地 5m Parquet 回放与漏斗审计脚本 | 是 |
| `local_bench/scenarios/*.yaml` | 多组对照实验参数模板 | 是 |
| `docs/LOCAL_TESTING.md` | 测试规范与实验迭代记录 | 是 |
| `local_bench/data/` | 本地存放的 `*.parquet` 原始 K 线数据 | **否**（已加入 `.gitignore`） |
| `local_bench/outputs/` | 每次回放生成的 `trades.csv` / `route_attempts.csv` / `summary.json` | **否**（已加入 `.gitignore`） |

## 2. 一键运行离线单元测试（每次修改代码必跑）

```bash
python -m unittest discover -s tests -v
```

覆盖 7 项核心断言：
1. 做空选币时，负资金费率（高轧空风险）严格扣分。
2. `PaperBroker` 限价止盈按目标价成交、两档 `0.5 + 0.5` 清仓零残留，`TIME`/`STOP` 市价单含不利方向滑点。
3. 追踪止损触发后自动下移止损位，且 `LiveBroker` 撤旧止损单并按剩余仓位重挂新单。
4. `score_at_event` 正确包含 `T0` 当天已完成的所有 4H K 线。
5. `rollover_day` 跨日重置时自动解除 `halted` 熔断标志。
6. 做空滑点按买盘深度 `bid_depth_usd` 估算，且净盈亏比扣除往返手续费。
7. `entry.py` 的 15m「破位后弱反弹受阻」入场过滤器准确触发并输出结构止损位。

## 3. 本地 Parquet 对照回放

将本地下载的 5m `*.parquet` 文件（如 `WIFUSDT.parquet`、`1000PEPEUSDT.parquet`、`DOGSUSDT.parquet`）放入 `local_bench/data/`，然后依次运行对照场景：

```bash
# 基线对照（直接破位追空 + 120m 超时 + 无追踪止损）
python local_bench/replay_parquet.py   --data-dir local_bench/data   --scenario local_bench/scenarios/baseline.yaml   --out-dir local_bench/outputs/baseline

# 改进方案（15m 破位后弱反弹受阻再空 + 240m 持仓 + 追踪止损）
python local_bench/replay_parquet.py   --data-dir local_bench/data   --scenario local_bench/scenarios/rebound_reject.yaml   --out-dir local_bench/outputs/rebound_reject

# 宽目标瀑布方案（360m 持仓 + 8% 目标 + 追踪止损）
python local_bench/replay_parquet.py   --data-dir local_bench/data   --scenario local_bench/scenarios/trailing_wide.yaml   --out-dir local_bench/outputs/trailing_wide
```

每次运行结束后，检查对应输出目录下的：
- `summary.json`：对比 `route_attempts`、`executed_trades`、`rejections`、`win_rate`、`mean_net_return`、`total_pnl_usd` 和 `exit_reasons`。
- `route_attempts.csv`：排查被拒路线集中卡在 `INSUFFICIENT_NET_RR` 还是 `INVALID_SHORT_LEVELS`。
- `trades.csv`：逐笔核对开平仓时间、退出原因（`STOP` / `TRAIL_STOP` / `TP` / `TIME`）与净收益。

## 4. 覆盖到本地仓库并推送 GitHub

解压 `memeshort-v2.zip` 覆盖到本地 `jump` 仓库根目录后，执行：

```bash
python -m unittest discover -s tests -v
git status
git add .gitignore config.yaml memeshort/ tests/ local_bench/ docs/
git commit -m "feat: upgrade memeshort v2 with bugfixes, rebound-reject entry, trailing stop & local bench"
git push
```
