"""Technical indicators — pure functions over pandas Series/DataFrame.

Deliberately dependency-light (pandas/numpy only) and side-effect free so they
can be unit-tested against known values. RSI and ATR both use Wilder's
smoothing (the textbook definition), which is what most charting platforms and
IBKR display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` observations."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns values in [0, 100]. The first ``period`` values are NaN. When there
    are no losses over the window the RSI is 100 (and 0 when there are no gains).
    """
    if period <= 0:
        raise ValueError("period must be positive")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing == EMA with alpha = 1/period, seeded by the SMA of the
    # first ``period`` values. ewm(alpha=1/period, adjust=False) after a simple
    # mean seed reproduces this exactly.
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 & avg_gain > 0  -> rs = +inf -> out = 100 (formula handles it).
    # avg_loss == 0 & avg_gain == 0 -> rs = 0/0 = NaN -> flat market, define as 50.
    flat = (avg_gain == 0) & (avg_loss == 0)
    out = out.mask(flat, 50.0)
    # Preserve the warm-up NaNs (first `period` values have no defined average).
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-prevClose|, |L-prevClose|)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, period)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: simple-mean seed over the first ``period`` *valid*
    values, then the recursive update ``avg_t = avg_{t-1} + (x_t - avg_{t-1})/N``.

    Leading NaNs are skipped before seeding. This matters because the inputs warm
    up at different offsets: RSI's gain/loss come from ``diff()`` (index 0 is
    NaN, so the seed spans changes 1..period and the first output lands at index
    ``period``), whereas ATR's true range is valid from index 0 (first output at
    index ``period-1``). Both are the textbook positions.
    """
    values = series.to_numpy(dtype="float64")
    n = len(values)
    out = np.full(n, np.nan, dtype="float64")

    # First index with a real value.
    valid = np.where(~np.isnan(values))[0]
    if valid.size < period:
        return pd.Series(out, index=series.index)
    start = int(valid[0])
    seed_end = start + period - 1
    if seed_end >= n:
        return pd.Series(out, index=series.index)

    out[seed_end] = values[start : seed_end + 1].mean()
    alpha = 1.0 / period
    for i in range(seed_end + 1, n):
        prev = out[i - 1]
        cur = values[i]
        out[i] = prev if np.isnan(cur) else prev + alpha * (cur - prev)
    return pd.Series(out, index=series.index)
