"""Backtest regression + cost-model tests.

The regression test locks a known result on *fixed* CSV fixtures
(``tests/fixtures/cache/*.csv``) so any future refactor that silently changes
strategy, risk or execution behaviour fails loudly. The fixtures are committed
deterministic synthetic bars — they are NOT regenerated at test time.

If you intentionally change behaviour, re-run the backtest against the fixtures
and update the locked values below (and say so in the commit).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.backtest import Backtester, benchmark_metrics
from src.config import (
    AccountConfig,
    AppConfig,
    BacktestConfig,
    DataConfig,
    EnvSettings,
    RiskConfig,
    SizingConfig,
    StrategyConfig,
    SymbolSpec,
    YamlConfig,
)
from src.data import load_cache

FIXTURES = Path(__file__).parent / "fixtures" / "cache"
SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def _cfg(benchmark: str | None = None, **bt_overrides) -> AppConfig:
    """A fixed, fixture-pointing config — independent of the repo's config.yaml."""
    bt = BacktestConfig(
        commission_per_share=0.005, min_commission=1.0, max_commission_pct=0.01,
        slippage_bps=2.0, benchmark=benchmark, **bt_overrides,
    )
    yaml = YamlConfig(
        account=AccountConfig(base_currency="USD", total_capital=10_000),
        sizing=SizingConfig(per_trade_notional=2_000, allow_fractional=True),
        strategy=StrategyConfig(sma_trend=200, rsi_period=14, rsi_entry=30,
                                rsi_exit=55, atr_period=14, atr_mult=3.0),
        risk=RiskConfig(max_open_positions=4, one_trade_per_symbol=True,
                        cooldown_days=3, daily_loss_limit_pct=0.03,
                        cash_buffer_pct=0.05),
        backtest=bt,
        data=DataConfig(cache_dir=str(FIXTURES)),
        active_universe="reg",
        universe={"reg": [SymbolSpec(symbol=s, currency="USD") for s in SYMBOLS]},
    )
    return AppConfig(env=EnvSettings(), cfg=yaml)


def _load(symbols=SYMBOLS) -> dict:
    return {s: load_cache(FIXTURES, s) for s in symbols}


# ── Regression: locked end-to-end result ─────────────────────────────────────
def test_backtest_regression_metrics():
    res = Backtester(_cfg(), starting_cash=10_000).run(_load())
    m = res.metrics

    # Exact structural locks — these change only if the trade logic changes.
    assert m["trading_days"] == 600
    assert m["num_trades"] == 5
    assert m["start_equity"] == pytest.approx(10_000.0)

    # Value locks (tight tolerance; a behaviour change shifts these far more).
    assert m["end_equity"] == pytest.approx(10_016.261586, rel=1e-6)
    assert m["total_return"] == pytest.approx(0.00162616, rel=1e-5)
    assert m["cagr"] == pytest.approx(0.00068266, rel=1e-5)
    assert m["max_drawdown"] == pytest.approx(-0.01720949, rel=1e-5)
    assert m["sharpe"] == pytest.approx(0.06416229, rel=1e-5)
    assert m["win_rate"] == pytest.approx(0.4)
    assert m["avg_hold_days"] == pytest.approx(12.4)
    assert m["total_trade_pnl"] == pytest.approx(26.261586, rel=1e-5)


def test_backtest_regression_trade_sequence():
    res = Backtester(_cfg(), starting_cash=10_000).run(_load())
    # (symbol, entry_date, exit_date, reason) — the full decision trail, locked.
    expected = [
        ("CCC", date(2022, 11, 10), date(2022, 11, 29), "atr_trailing_stop"),
        ("CCC", date(2022, 12, 7), date(2022, 12, 22), "rsi_exit"),
        ("CCC", date(2023, 4, 5), date(2023, 4, 18), "rsi_exit"),
        ("DDD", date(2023, 6, 20), date(2023, 6, 29), "trend_break"),
        ("AAA", date(2024, 4, 9), date(2024, 4, 15), "trend_break"),
    ]
    got = [(t.symbol, t.entry_date, t.exit_date, t.reason) for t in res.trades]
    assert got == expected


# ── Cost model (IBKR fixed: per-share, min per order, capped at % of value) ───
def test_commission_per_share_above_minimum():
    bt = Backtester(_cfg(), starting_cash=10_000)
    # 1000 sh * $0.005 = $5.00; well above the $1 min and below the 1% cap.
    assert bt._commission(1_000, 50.0) == pytest.approx(5.0)


def test_commission_hits_minimum():
    bt = Backtester(_cfg(), starting_cash=10_000)
    # 50 sh * $0.005 = $0.25 -> raised to the $1.00 per-order minimum.
    # 1% of 50*100 = $50 cap does not bind.
    assert bt._commission(50, 100.0) == pytest.approx(1.0)


def test_commission_max_pct_cap_overrides_minimum():
    bt = Backtester(_cfg(), starting_cash=10_000)
    # Tiny notional: 1 sh @ $20 -> per-share $0.005, min would be $1, but the
    # 1% cap ($0.20) overrides the minimum.
    assert bt._commission(1, 20.0) == pytest.approx(0.20)


def test_commission_zero_for_no_shares():
    bt = Backtester(_cfg(), starting_cash=10_000)
    assert bt._commission(0, 100.0) == 0.0


# ── Benchmark / alpha ─────────────────────────────────────────────────────────
def test_benchmark_and_alpha():
    cfg = _cfg()
    res = Backtester(cfg, starting_cash=10_000).run(_load())
    bench = load_cache(FIXTURES, "BENCH")
    bench.attrs["symbol"] = "BENCH"
    bm = benchmark_metrics(bench, res.equity_curve.index, 10_000,
                           cfg.cfg.backtest, res.metrics)

    assert bm["benchmark_symbol"] == "BENCH"
    assert bm["benchmark_total_return"] == pytest.approx(0.399563, rel=1e-5)
    assert bm["benchmark_cagr"] == pytest.approx(0.15164, rel=1e-4)
    # Alpha = strategy − benchmark; the trend strategy badly trails this drift.
    assert bm["alpha_cagr"] == pytest.approx(
        res.metrics["cagr"] - bm["benchmark_cagr"], rel=1e-9)
    assert bm["alpha_total_return"] == pytest.approx(
        res.metrics["total_return"] - bm["benchmark_total_return"], rel=1e-9)


def test_benchmark_insufficient_overlap_returns_empty():
    cfg = _cfg()
    bench = load_cache(FIXTURES, "BENCH").iloc[:1]  # only one bar -> no overlap window
    bench.attrs["symbol"] = "BENCH"
    assert benchmark_metrics(bench, bench.index, 10_000, cfg.cfg.backtest, {}) == {}
