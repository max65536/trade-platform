Working Memory Snapshot (for new sessions)

- CLI: `trade-cli` supports `fetch/batch/analyze/backtest/mtf/plot`; pipeline at `scripts/pipeline.py`.
- Chan: `analyze()` defaults to full workflow (inclusion→fractals→pens→segments→pivots + signals: turn, buy1/2/3, sell1/2/3). Use `mode="simple"` for legacy.
- Backtest: next-bar entries, TP then SL intrabar, long-only; stats include PF, drawdown, exposure; equity series aligned to bars.
- MTF: use `multiframe.align_htf_to_ltf` and `filter_signals_with_htf_opts(require_htf_breakout, min_htf_run)`.
- WebUI: pages `/`, `/echarts`, `/echarts-mtf`; APIs `/api/ohlcv`, `/api/mtf`.
- Proxies: `TRADE_HTTP_PROXY`, `TRADE_HTTPS_PROXY`, `TRADE_NO_PROXY` supported in ccxt wrapper.

Active Plan (next steps)

1) CI basics: GitHub Actions running `ruff`, `black --check`, `pytest -q` on push/PR.
2) Tests: add Chan divergence (buy3/sell3) and pivot retest (buy2/sell2) edge-case unit tests.
3) Backtest stats: add Sharpe/Sortino/CAGR/Kelly-lite and drawdown recovery; surface in CLI/WebUI.
4) Fetch/batch: incremental append + dedupe, integrity checks, and resumable downloads.
5) Strategy presets: YAML presets (momentum/meanrev/MTF) consumed by CLI and pipeline.
6) WebUI polish: ECharts toggles for signal groups, trade overlay, export current params.

Verification Checklist
- Lint/format: `pdm run lint` and `pdm run format-check` clean.
- Tests: `pdm run pytest -q` green; new cases cover Chan signals and backtest stats.
- Manual: run a small MTF pipeline and plot succeeds; WebUI endpoints return JSON within 1–2s on sample CSV.

TODOs (Development Roadmap)

WebUI · 单周期（/echarts）
- [x] 页面与表单（CSV 或 Exchange 源）
- [x] `GET /api/ohlcv`（OHLCV + 指标 + 缠论）
- [x] 主图 SMA/EMA 叠加
- [x] 子图 MACD/RSI/ATR（可开关）
- [x] 缩放/联动/十字光标
- [ ] 指标参数可调（`macd=12,26,9` / `rsi=14` / `atr=14`）
- [ ] 主题/参数记忆（URL/state 持久化增强）

WebUI · 多周期（/echarts-mtf）
- [x] 页面与表单（LTF/HTF，CSV 或 Exchange）
- [x] `GET /api/mtf`（对齐 HTF→LTF，返回 bands/信号）
- [x] 过滤参数：`require_htf_breakout`、`min_htf_run`
- [ ] MTF 回测叠加：返回 `trades/equity/stats`
- [ ] 资金曲线子图 + 交易点叠加
- [ ] 参数 preset 保存/载入

Backtest（回测引擎）
- [x] 资金曲线（bar 对齐，forward-fill）
- [x] 统计扩展（PF/回撤/持仓/暴露）
- [x] `initial_capital/position_size/slippage_bps`
- [ ] 做空/翻转模式
- [ ] 末尾强制平仓选项
- [ ] Sharpe/Sortino/CAGR（含 `--ppyear`）

CLI / Pipeline
- [x] backtest 新参数（保存 trades/stats）
- [ ] pipeline 接入 slippage/sizing 并可选保存 equity/trades
- [ ] 示例脚本：MTF 回测一体化（拉取→对齐→回测→图表）

Indicators（指标）
- [ ] MACD/RSI/ATR 参数化（CLI + WebAPI + 前端）
- [ ] 新增可选指标（BBands/Stoch/VWAP）

Realtime（可选）
- [ ] WebSocket 推送最新 K 线
- [ ] 前端增量合并 + 窗口滚动

Plotting（绘图增强）
- [x] 主图 SMA/EMA 叠加
- [x] 子图 MACD/RSI/ATR
- [ ] fractals/pivot 标签渲染

Docs & Examples（文档与示例）
- [x] README 增补 WebUI/ECharts/API
- [x] DEVELOP.md 维护 TODO 列表
- [ ] API 使用示例（curl/参数表）

Testing & Quality（测试与质量）
- [x] 单元测试通过（`pdm run pytest -q`）
- [ ] backtest 统计的新增用例（回撤/暴露/持仓时长）
- [ ] WebUI API 冒烟脚本（/api/ohlcv, /api/mtf）

Infra（工程化）
- [ ] pre-commit（ruff/black/pytest -q）
- [ ] 版本与变更日志（CHANGELOG）
- [ ] 样本数据与使用说明（data/）

Notes for Contributors
- Prefer deterministic synthetic data in tests; avoid network.
- Keep modules focused; minimal API surface with type hints and short docstrings.
- When adding features, update AGENTS.md memory and this Active Plan if scope changes.
