# Agent Memory (Persistent)

This section captures stable, high-signal facts so new sessions avoid re-reading the entire repo.

- Purpose: ccxt-based research/trading skeleton with TA, Chan (full), MTF alignment, backtest, CLI + minimal WebUI.
- Data schema: CSV columns `[timestamp, open, high, low, close, volume]` plus `datetime` (ms→UTC naive).
- Analysis default: `chan.analyze()` now runs the full workflow (inclusion→fractals→pens→segments→pivots + signals: turn/buy1/2/3, sell1/2/3). Pass `mode="simple"` for legacy behavior.
- Backtest semantics: next-bar-at-open execution; TP/SL checked intrabar (TP first), long-only flip on sell; equity curve aligned to bars and forward-filled; supports `initial_capital/position_size/slippage_bps`.
- MTF filtering: map HTF `segments/bands` to LTF via datetime; keep buys when `htf_dir>0` (and optional breakout over `htf_pivot_high`), sells when `htf_dir<0` (optional breakout below `htf_pivot_low`); optional `min_htf_run`.
- WebUI: `trade-webui` serves `/`, `/echarts`, `/echarts-mtf`; APIs: `GET /api/ohlcv` and `GET /api/mtf` return OHLCV, indicators, Chan outputs, and MTF context.
- CLI entry: `trade-cli` with subcommands `fetch/batch/analyze/backtest/mtf/plot`; end-to-end script at `scripts/pipeline.py` (`pdm run pipeline -- ...`).
- Proxies: ccxt wrapper honors `TRADE_HTTP_PROXY`, `TRADE_HTTPS_PROXY`, `TRADE_NO_PROXY` if provided.
- Non-goals (current): shorting, portfolio/multi-asset risk, persistent DB; focus on CSV + stateless analysis.

Session Boot Checklist
- Run tests: `pdm run pytest -q` and format/lint if needed.
- Skim `README.md` for latest quickstart and WebUI endpoints.
- For MTF tasks: load LTF/HTF CSVs, call `chan.analyze` on both, then `multiframe.align_htf_to_ltf` and `filter_signals_with_htf_opts`.
- For backtests: ensure indicators via `strategy.ensure_indicators`, then `simple_execute` or `execute_with_risk`.

Preferred Conventions
- Python 3.10+, PEP 8, typed public APIs; modules/functions `snake_case`, classes `PascalCase`.
- Indicators prefer pure functions; avoid network in tests (use synthetic OHLCV in `tests/`).

# Repository Guidelines

## Project Structure & Module Organization
- `trade_platform/`: core package
  - `exchanges.py`: ccxt wrapper for OHLCV
  - `dataio.py`: CSV I/O and schema
  - `indicators.py`: SMA/EMA/RSI/ATR/MACD
  - `chan.py`: Chan (fractals→pens→segments→pivots + signals)
  - `multiframe.py`: HTF→LTF alignment and filters
  - `backtest.py`: basic executor + stats
  - `plotting.py`: matplotlib visualizations
  - `cli.py`: CLI entry (`python -m trade_platform.cli`)
- `README.md`: quickstart and examples
- `data/`: local CSVs (optional)
 - `pyproject.toml`: PDM project config and deps

## Build, Test, and Development Commands (PDM)
- Install runtime deps: `pdm install`
- Install dev deps: `pdm install -G dev`
- CLI (via console script):
  - Fetch: `pdm run trade-cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --output data/BTCUSDT-1h.csv`
  - Batch: `pdm run trade-cli batch --exchange binance --symbols BTC/USDT ETH/USDT --timeframes 4h 1d --output-dir data/spot --name-template {symbol_noslash}-{timeframe}.csv`
  - Analyze: `pdm run trade-cli analyze --input data/BTCUSDT-1h.csv --out data/annotated.csv`
  - Backtest: `pdm run trade-cli backtest --input data/BTCUSDT-1h.csv`
  - MTF (4h+1d): `pdm run trade-cli mtf --lower-input data/BTCUSDT-4h.csv --higher-input data/BTCUSDT-1d.csv --out data/BTCUSDT-4h-mtf.csv --require-htf-breakout --min-htf-run 3 --run-backtest`
  - Plot: `pdm run trade-cli plot --input data/BTCUSDT-4h-mtf.csv --use-mtf-bands --use-mtf-signals --theme dark --save out/plot.png`
- Tests: `pdm run pytest -q`
- Lint: `pdm run lint` (ruff)
- Format: `pdm run format` (black) / `pdm run format-check`
- Preset runs:
  - Momentum 1h: `pdm run bt_momentum`
  - Mean-reversion 1h: `pdm run bt_meanrev`
  - MTF 4h+1d momentum: `pdm run mtf_momentum_4h_1d`
  - MTF 4h+1d mean-reversion: `pdm run mtf_meanrev_4h_1d`
- End-to-end pipeline:
  - `pdm run pipeline -- --exchange binance --symbols BTC/USDT ETH/USDT --lower-tf 4h --higher-tf 1d --out-dir runs --require-htf-breakout --min-htf-run 3`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation
- Type hints and concise docstrings on public APIs
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`
- Keep modules focused; indicators/analysis prefer pure functions
- Optional tools: `black trade_platform` and `ruff check trade_platform`

## Testing Guidelines
- Framework: `pytest`
- Location: `tests/` with `test_*.py`
- Scope: indicators, Chan pipeline (fractals/pens/segments/pivots), MTF alignment, backtest math
- Run: `pytest -q`
- Prefer deterministic synthetic OHLCV over network calls

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects; group related changes; rationale in body
- PRs: description, scope, before/after evidence; link issues; include sample commands/output
- Checks: run analyze/mtf/backtest on a small CSV; no secrets; format/lint if available

## Security & Configuration Tips
- No hardcoded API keys; use env vars (e.g., `API_KEY`, `API_SECRET`) or local `.env` (gitignored)
- Avoid committing large CSVs; keep small samples for repro
