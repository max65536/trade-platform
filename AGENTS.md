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
- `requirements.txt`: Python deps (matplotlib included)
- `README.md`: quickstart and examples
- `data/`: local CSVs (optional)

## Build, Test, and Development Commands
- Env: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- CLI:
  - Fetch: `python -m trade_platform.cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --output data/BTCUSDT-1h.csv`
  - Analyze: `python -m trade_platform.cli analyze --input data/BTCUSDT-1h.csv --out data/annotated.csv`
  - Backtest: `python -m trade_platform.cli backtest --input data/BTCUSDT-1h.csv`
  - MTF (4h+1d): `python -m trade_platform.cli mtf --lower-input data/BTCUSDT-4h.csv --higher-input data/BTCUSDT-1d.csv --out data/BTCUSDT-4h-mtf.csv --require-htf-breakout --min-htf-run 3 --run-backtest`
  - Plot: `python -m trade_platform.cli plot --input data/BTCUSDT-4h-mtf.csv --use-mtf-bands --use-mtf-signals --theme dark --save out/plot.png`

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
