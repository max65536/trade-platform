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
