"""Cross-sectional momentum engine tests (synthetic panel, no cache needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.xsec import run_xsec


def _panel(n=400):
    """6 names: 3 clear up-trenders, 3 down-trenders — momentum must pick the ups."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n)
    cols = {}
    for i in range(3):
        cols[f"UP{i}"] = 100.0 * np.exp((0.0010 + 0.0001 * i) * t)
    for i in range(3):
        cols[f"DN{i}"] = 100.0 * np.exp((-0.0005 - 0.0001 * i) * t)
    return pd.DataFrame(cols, index=idx)


def test_run_xsec_selects_top_momentum_and_compounds():
    panel = _panel()
    equity, info = run_xsec(panel, top_k=3, lookback=60, skip=5, rebalance=20,
                            cost_bps=0.0, starting_cash=10_000)
    assert info["avg_names"] == 3                       # holds top_k names
    assert info["rebalances"] > 0
    assert equity.iloc[0] > 0
    # Up-trenders are selected -> equity grows over the window.
    assert equity.iloc[-1] > 10_000
    assert info["sharpe"] == info["sharpe"]             # not NaN


def test_run_xsec_regime_off_holds_cash():
    panel = _panel()
    # A regime series that is risk-off everywhere -> never invests -> flat equity.
    regime = pd.Series(False, index=panel.index)
    equity, info = run_xsec(panel, top_k=3, lookback=60, skip=5, rebalance=20,
                            cost_bps=0.0, starting_cash=10_000, regime=regime)
    assert info["cash_periods"] == info["rebalances"]   # every period in cash
    assert equity.iloc[-1] == pytest.approx(10_000)     # no positions -> no P&L


def test_run_xsec_costs_reduce_return():
    panel = _panel()
    eq_free, _ = run_xsec(panel, top_k=3, lookback=60, skip=5, rebalance=20,
                          cost_bps=0.0, starting_cash=10_000)
    eq_costly, _ = run_xsec(panel, top_k=3, lookback=60, skip=5, rebalance=20,
                            cost_bps=50.0, starting_cash=10_000)
    assert eq_costly.iloc[-1] < eq_free.iloc[-1]
