# Repository Guidelines

## Project Structure & Module Organization
- `trade_platform/`: core package
  - `exchanges.py`: ccxt wrapper for OHLCV
  - `dataio.py`: CSV I/O and schema
  - `indicators.py`: SMA/EMA/RSI/ATR/MACD
  - `chan.py`: simplified Chan (fractals→pens→segments)
  - `backtest.py`: basic executor + stats
  - `cli.py`: CLI entry (`python -m trade_platform.cli`)
- `requirements.txt`: Python deps
- `README.md`: quickstart and examples
- `data/`: suggested local folder for downloaded CSVs (not required)

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run CLI examples:
  - Fetch: `python -m trade_platform.cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --output data/BTCUSDT-1h.csv`
  - Analyze: `python -m trade_platform.cli analyze --input data/BTCUSDT-1h.csv --out data/annotated.csv`
  - Backtest: `python -m trade_platform.cli backtest --input data/BTCUSDT-1h.csv`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation
- Use type hints and clear docstrings for public functions
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`
- Keep modules small and focused; prefer pure functions for indicators/analysis
- Optional (if installed): `black trade_platform` and `ruff check trade_platform`

## Testing Guidelines
- Framework: `pytest` (add to a local env if needed)
- Location: `tests/` with files named `test_*.py`
- Scope: unit tests for indicators, fractal/pen/segment builders, and backtest math
- Run: `pytest -q`
- Prefer small, deterministic fixtures (synthetic OHLCV) over network calls

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; group related changes; include rationale in body
- PRs: clear description, scope, before/after notes; link related issues; include sample commands/output
- Checks: run analyze/backtest on a small CSV; ensure no secrets added; format/lint if available

## Security & Configuration Tips
- Do not hardcode API keys; use env vars (e.g., `API_KEY`, `API_SECRET`) or a local `.env` not committed
- Avoid committing large data files; prefer small samples for repro
