Trade Platform (ccxt + TA + Chan)

一个基于 ccxt 的最小可用交易/研究骨架：
- 统一拉取并存储K线数据（CSV）
- 常见技术指标（SMA/EMA/RSI/ATR/MACD）
- 简化版缠论分析（分型/笔/线段）
- 简单策略与回测器
- 命令行工具

快速开始
- Python 3.10+
- pip install -r requirements.txt

命令行用法
- 拉取K线：
  `python -m trade_platform.cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --limit 2000 --output data/BTCUSDT-1h.csv`

- 运行缠论分析：
  `python -m trade_platform.cli analyze --input data/BTCUSDT-1h.csv --out data/BTCUSDT-1h-analyzed.csv`

- 回测简化缠论策略：
  `python -m trade_platform.cli backtest --input data/BTCUSDT-1h.csv --start 2023-01-01 --end 2024-01-01`

目录结构
- trade_platform/
  - exchanges.py       ccxt封装
  - dataio.py          CSV读取/写入
  - indicators.py      技术指标
  - chan.py            缠论核心（简化版）
  - backtest.py        回测引擎
  - cli.py             命令行入口

说明
- 缠论实现为工程化简化版本，涵盖：分型→笔→线段 的抽象，便于在实盘/回测中稳定使用。
- 如需更严格/完整的缠论定义（如顶底分型判定、笔破坏/延伸、线段重构、中枢/走势类型等），可在当前基础上逐步细化。
- 回测为事件驱动、下根K线开盘成交的近似执行模型，主要用于快速验证策略逻辑。

