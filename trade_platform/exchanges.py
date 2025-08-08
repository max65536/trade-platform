from __future__ import annotations

import time
from typing import List, Optional

import ccxt  # type: ignore


class ExchangeClient:
    """Lightweight ccxt wrapper for unified OHLCV fetching with rate-limit handling."""

    def __init__(self, exchange: str, api_key: Optional[str] = None, secret: Optional[str] = None, password: Optional[str] = None):
        ex_class = getattr(ccxt, exchange)
        self.exchange = ex_class({
            "apiKey": api_key or "",
            "secret": secret or "",
            "password": password or "",
            "enableRateLimit": True,
            "timeout": 30000,
        })

    def load_markets(self):
        return self.exchange.load_markets()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> List[List[float]]:
        """Fetch a batch of OHLCV. Returns list of [ts, open, high, low, close, volume]."""
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def fetch_ohlcv_all(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
        max_bars: Optional[int] = None,
        sleep_sec: float = 0.2,
    ) -> List[List[float]]:
        """Paginate OHLCV fetching until no more data or max_bars reached."""
        results: List[List[float]] = []
        last_ts = since
        while True:
            batch = self.fetch_ohlcv(symbol, timeframe, since=last_ts, limit=limit)
            if not batch:
                break
            # ccxt returns candles including 'since' bar; avoid duplication on next loop
            if results and batch and batch[0][0] == results[-1][0]:
                batch = batch[1:]
            results.extend(batch)
            if max_bars and len(results) >= max_bars:
                results = results[:max_bars]
                break
            last_ts = results[-1][0]
            time.sleep(sleep_sec)
            if len(batch) < limit:
                break
        return results

