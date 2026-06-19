"""Historical & current daily bars, with a CSV cache.

Live/paper data comes from IBKR via ``fetch_history`` (uses an ``ib_async`` IB
instance). For offline work — backtests on a machine without a gateway, or this
sandbox — bars are read from / written to a CSV cache, and a synthetic generator
is provided purely so the engine can be exercised end-to-end without a broker.

Bar DataFrames have a sorted ``DatetimeIndex`` and columns:
open, high, low, close, volume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{symbol}.csv"


def save_cache(df: pd.DataFrame, cache_dir: str | Path, symbol: str) -> None:
    p = cache_path(cache_dir, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index_label="date")


def load_cache(cache_dir: str | Path, symbol: str) -> pd.DataFrame | None:
    p = cache_path(cache_dir, symbol)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df[_COLUMNS]


def bars_to_dataframe(bars) -> pd.DataFrame:
    """Convert ib_async historical bars into the canonical frame."""
    if not bars:
        return pd.DataFrame(columns=_COLUMNS)
    rows = []
    idx = []
    for b in bars:
        idx.append(pd.Timestamp(b.date))
        rows.append([b.open, b.high, b.low, b.close, getattr(b, "volume", 0) or 0])
    df = pd.DataFrame(rows, columns=_COLUMNS, index=pd.DatetimeIndex(idx))
    return df.sort_index()


def fetch_history(
    ib,
    contract,
    duration: str = "2 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> pd.DataFrame:
    """Request historical daily bars from IBKR. Requires a connected IB instance.

    ``endDateTime=""`` means "up to now"; IBKR returns *completed* bars, so the
    last row is the most recent finished daily bar (no look-ahead).
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
    )
    return bars_to_dataframe(bars)


def get_bars(
    symbol: str,
    *,
    ib=None,
    contract=None,
    cache_dir: str | Path = "data/cache",
    duration: str = "2 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    refresh: bool = True,
) -> pd.DataFrame:
    """Return daily bars, fetching from IBKR when possible and caching to CSV.

    Falls back to the CSV cache when no IB connection is supplied or the fetch
    fails — so a backtest can run from cached data with no gateway.
    """
    if ib is not None and contract is not None and refresh:
        try:
            df = fetch_history(ib, contract, duration, bar_size, what_to_show, use_rth)
            if len(df):
                save_cache(df, cache_dir, symbol)
                return df
        except Exception:
            pass  # fall through to cache
    cached = load_cache(cache_dir, symbol)
    if cached is not None:
        return cached
    raise FileNotFoundError(
        f"No data for {symbol}: no IB connection and no cache at "
        f"{cache_path(cache_dir, symbol)}"
    )


def generate_synthetic(
    symbol: str, n: int = 600, seed: int | None = None, start: str = "2022-01-03"
) -> pd.DataFrame:
    """Synthetic daily OHLC via geometric Brownian motion.

    For OFFLINE demonstration / smoke-testing of the engine only — never a
    substitute for real market data. Deterministic given a seed.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))
    mu, sigma = 0.08 / 252, 0.20 / np.sqrt(252)
    rets = rng.normal(mu, sigma, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    # Build plausible OHLC around the close path.
    intraday = np.abs(rng.normal(0, sigma, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )
