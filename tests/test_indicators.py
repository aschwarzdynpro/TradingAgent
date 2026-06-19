"""Indicator tests against known/independently-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import atr, rsi, sma, true_range


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    out = sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_invalid_period():
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), 0)


# Canonical Wilder / StockCharts RSI(14) reference series and expected values.
# Source: J. Welles Wilder example as reproduced by StockCharts.
_WILDER_CLOSE = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]
_WILDER_RSI_FROM_IDX14 = [
    70.46, 66.25, 66.48, 69.35, 66.29, 57.92, 62.88, 63.20, 56.02, 62.34,
    54.67, 50.39, 39.98, 41.46, 41.87, 45.46, 37.30, 33.10, 37.77,
]


def test_rsi_wilder_reference():
    close = pd.Series(_WILDER_CLOSE, dtype="float64")
    out = rsi(close, 14)
    # First 14 values (indices 0..13) are warm-up NaNs; first RSI at index 14.
    assert out.iloc[:14].isna().all()
    got = out.iloc[14 : 14 + len(_WILDER_RSI_FROM_IDX14)].to_numpy()
    expected = np.array(_WILDER_RSI_FROM_IDX14)
    # Allow small rounding tolerance vs the published 2-decimal table.
    assert np.allclose(got, expected, atol=0.15)


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 40, dtype="float64"))
    out = rsi(close, 14)
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    close = pd.Series(np.arange(40, 1, -1, dtype="float64"))
    out = rsi(close, 14)
    assert out.dropna().iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_is_50():
    close = pd.Series([100.0] * 40)
    out = rsi(close, 14)
    # No gains and no losses -> defined as a neutral 50.
    assert out.dropna().iloc[-1] == pytest.approx(50.0)


def test_true_range():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 9.5, 11.0])
    close = pd.Series([9.5, 10.5, 11.5])
    tr = true_range(high, low, close)
    # Bar 0: no prev close -> just H-L = 1.0
    assert tr.iloc[0] == pytest.approx(1.0)
    # Bar 1: max(11-9.5=1.5, |11-9.5|=1.5, |9.5-9.5|=0) = 1.5
    assert tr.iloc[1] == pytest.approx(1.5)
    # Bar 2: max(12-11=1, |12-10.5|=1.5, |11-10.5|=0.5) = 1.5
    assert tr.iloc[2] == pytest.approx(1.5)


def test_atr_constant_range():
    # Each bar has identical true range of 2.0 -> ATR converges to 2.0.
    n = 30
    high = pd.Series([12.0] * n)
    low = pd.Series([10.0] * n)
    close = pd.Series([11.0] * n)
    out = atr(high, low, close, 14)
    assert out.iloc[:13].isna().all()
    assert out.iloc[13] == pytest.approx(2.0)
    assert out.iloc[-1] == pytest.approx(2.0)
