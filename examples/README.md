Examples

This folder contains small, runnable examples showing how to use the core features without network access. They generate deterministic synthetic OHLCV CSVs and then run analysis/backtests/plots using the in-repo modules or CLI.

Contents
- generate_synthetic.py: writes sample CSVs under examples/data.
- backtest_example.py: single-timeframe analysis + backtest on 1h synthetic.
- mtf_example.py: multi-timeframe alignment (4h + 1d) + optional backtest.
- plot_example.py: render a plot with signals and optional trade overlay.

Quickstart
1) Create synthetic data files
   - `pdm run python examples/generate_synthetic.py`

2) Run backtest (Python API)
   - `pdm run python examples/backtest_example.py`

3) Run MTF example (Python API)
   - `pdm run python examples/mtf_example.py`

4) Produce a plot
   - `pdm run python examples/plot_example.py`
   - Output PNG saved to `examples/out/plot.png`

CLI alternatives (after generating synthetic CSVs)
- Single TF backtest:
  - `pdm run trade-cli backtest --input examples/data/BTCUSDT-1h.csv --ppyear 8760 --stop-pct 0.02 --tp-pct 0.04 --save-trades examples/out/trades.csv --save-stats examples/out/stats.json`

- Multi TF (4h + 1d):
  - `pdm run trade-cli mtf --lower-input examples/data/BTCUSDT-4h.csv --higher-input examples/data/BTCUSDT-1d.csv --out examples/data/BTCUSDT-4h-mtf.csv --require-htf-breakout --min-htf-run 2 --run-backtest`

- Plot (using MTF CSV produced above):
  - `pdm run trade-cli plot --input examples/data/BTCUSDT-4h-mtf.csv --use-mtf-bands --use-mtf-signals --theme dark --save examples/out/plot.png`

Notes
- These examples avoid network calls and use synthetic data only.
- Matplotlib is required for plotting: `pip install matplotlib` if not already available.

PDM script aliases
- `pdm run ex_gen`: generate synthetic CSVs
- `pdm run ex_backtest`: run single-timeframe backtest example
- `pdm run ex_mtf`: run multi-timeframe alignment + backtest example
- `pdm run ex_plot`: render a PNG chart to examples/out/plot.png
- `pdm run ex_all`: run all the above in sequence
- `pdm run examples`: interactive menu (or `--list`, `--run gen|backtest|mtf|plot|all`)
