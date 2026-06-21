"""Walk-forward sweep harness tests.

Fast + deterministic: they reuse the committed backtest fixtures (AAA..DDD, 600
bars each) and only ever run tiny windows / tiny grids, so the suite stays quick.
The point is to lock the *mechanics* of the walk-forward (fold layout, warm-up
scoping, equity renormalisation, in-sample selection) — not strategy numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
from src.sweep import (
    GRIDS,
    ParamSet,
    build_fold_indices,
    cfg_with_params,
    evaluate_window,
    grid_params,
    objective_value,
    stitch_oos,
    walk_forward,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cache"
SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def _cfg() -> AppConfig:
    yaml = YamlConfig(
        account=AccountConfig(base_currency="USD", total_capital=10_000),
        sizing=SizingConfig(per_trade_notional=2_000, allow_fractional=True),
        strategy=StrategyConfig(sma_trend=100, rsi_period=14, rsi_entry=30,
                                rsi_exit=55, atr_period=14, atr_mult=3.0),
        risk=RiskConfig(max_open_positions=4, one_trade_per_symbol=True,
                        cooldown_days=3, daily_loss_limit_pct=0.03, cash_buffer_pct=0.05),
        backtest=BacktestConfig(benchmark=None),
        data=DataConfig(cache_dir=str(FIXTURES)),
        active_universe="reg",
        universe={"reg": [SymbolSpec(symbol=s, currency="USD") for s in SYMBOLS]},
    )
    return AppConfig(env=EnvSettings(), cfg=yaml)


def _data() -> dict:
    return {s: load_cache(FIXTURES, s) for s in SYMBOLS}


# ── grid + param application ─────────────────────────────────────────────────
def test_grid_params_count_and_type():
    params = grid_params(GRIDS["coarse"])
    g = GRIDS["coarse"]
    expected = (len(g["sma_trend"]) * len(g["rsi_entry"]) * len(g["rsi_exit"])
                * len(g["atr_mult"]) * len(g["cooldown_days"]))
    assert len(params) == expected == 81
    assert all(isinstance(p, ParamSet) for p in params)
    assert len(set(params)) == len(params)  # frozen dataclass -> hashable + unique


def test_momentum_grid_builds_trend_momentum_paramsets():
    params = grid_params(GRIDS["momentum"])
    g = GRIDS["momentum"]
    assert len(params) == (len(g["sma_trend"]) * len(g["mom_lookback"])
                           * len(g["atr_mult"]) * len(g["cooldown_days"]))
    assert all(p.mode == "trend_momentum" for p in params)
    assert {p.mom_lookback for p in params} == set(g["mom_lookback"])


def test_cfg_with_params_applies_mode_and_momentum():
    cfg = _cfg()
    p = ParamSet(sma_trend=150, atr_mult=4.0, cooldown_days=3,
                 mode="trend_momentum", mom_lookback=126)
    c2 = cfg_with_params(cfg, p)
    assert c2.cfg.strategy.mode == "trend_momentum"
    assert c2.cfg.strategy.mom_lookback == 126
    assert c2.cfg.strategy.sma_trend == 150
    assert cfg.cfg.strategy.mode == "mean_reversion"  # original untouched


def test_cfg_with_params_applies_without_mutating_original():
    cfg = _cfg()
    p = ParamSet(sma_trend=150, rsi_entry=25, rsi_exit=60, atr_mult=2.0, cooldown_days=5)
    c2 = cfg_with_params(cfg, p)
    assert (c2.cfg.strategy.sma_trend, c2.cfg.strategy.rsi_entry, c2.cfg.strategy.rsi_exit,
            c2.cfg.strategy.atr_mult, c2.cfg.risk.cooldown_days) == (150, 25, 60, 2.0, 5)
    # Original untouched (deep copy).
    assert cfg.cfg.strategy.sma_trend == 100
    assert cfg.cfg.risk.cooldown_days == 3


# ── fold layout ──────────────────────────────────────────────────────────────
def test_build_fold_indices_rolling_nonoverlapping():
    folds = build_fold_indices(n=2000, warmup=260, train_bars=500, test_bars=250, max_folds=99)
    assert folds[0] == (260, 759, 760, 1009)        # first train starts at warmup
    assert len(folds) == 4                            # last would overrun n=2000
    for _tr_s, tr_e, te_s, te_e in folds:
        assert tr_e == te_s - 1                       # train ends right before test
        assert te_e - te_s + 1 <= 250                 # test no longer than test_bars
    for a, b in zip(folds, folds[1:], strict=False):  # consecutive pairs: contiguous tests
        assert a[3] + 1 == b[2]


def test_build_fold_indices_respects_max_folds():
    assert len(build_fold_indices(2000, 260, 500, 250, max_folds=2)) == 2


# ── objective / selection ────────────────────────────────────────────────────
def test_objective_value_rejects_thin_samples():
    assert objective_value({"num_trades": 2, "total_return": 0.9}, "total_return", 3) == float("-inf")
    assert objective_value({"num_trades": 5, "total_return": 0.1}, "total_return", 3) == pytest.approx(0.1)
    assert objective_value({}, "total_return", 3) == float("-inf")


# ── warm-up-scoped windowed evaluation ───────────────────────────────────────
def test_evaluate_window_is_scoped_and_renormalised():
    cfg, data = _cfg(), _data()
    calendar = sorted(set().union(*[set(df.index) for df in data.values()]))
    p = ParamSet(100, 30, 55, 3.0, 3)
    w_start_idx, w_end_idx, warmup = 300, 450, 150
    m, trades, eq = evaluate_window(cfg, p, data, calendar, w_start_idx, w_end_idx, warmup, 10_000)

    # Equity is renormalised to the starting cash at the window's first scored day.
    assert m["start_equity"] == pytest.approx(10_000.0)
    # Only the window itself is scored (not the warm-up lead).
    assert m["trading_days"] == w_end_idx - w_start_idx + 1
    assert eq.index[0] == calendar[w_start_idx]
    # Trades are attributed by entry date inside the window.
    for t in trades:
        assert calendar[w_start_idx].date() <= t.entry_date <= calendar[w_end_idx].date()


# ── end-to-end walk-forward on fixtures ──────────────────────────────────────
def test_walk_forward_end_to_end_small_grid():
    cfg, data = _cfg(), _data()
    params = [ParamSet(100, 30, 55, 3.0, 3), ParamSet(100, 25, 60, 2.0, 3)]
    folds = walk_forward(
        cfg, data, params, train_years=0.5, test_years=0.5, max_folds=99,
        warmup=150, objective="total_return", min_trades=0, starting_cash=10_000)

    assert folds, "expected at least one fold from 600-bar fixtures"
    for f in folds:
        assert f.best in params                       # chose a grid member
        assert f.train_end < f.test_start             # no look-ahead across the split
        if not f.oos_equity.empty:
            assert f.oos_metrics["start_equity"] == pytest.approx(10_000.0)

    stitched = stitch_oos(folds, 10_000)
    assert not stitched.empty
    assert float(stitched.iloc[0]) > 0
    assert stitched.index.is_monotonic_increasing     # folds stitched in time order


def test_walk_forward_raises_when_history_too_short():
    cfg, data = _cfg(), _data()
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward(cfg, data, [ParamSet(100, 30, 55, 3.0, 3)],
                     train_years=2.0, test_years=2.0, max_folds=99, warmup=260,
                     objective="total_return", min_trades=0, starting_cash=10_000)
