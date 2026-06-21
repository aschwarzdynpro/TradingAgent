"""Risk-layer tests: sizing/rounding, caps, cooldown, kill-switch, buffers."""

from __future__ import annotations

from datetime import date

import pytest

from src.config import RiskConfig, SizingConfig
from src.risk import AccountSnapshot, RiskManager, compute_quantity, compute_quantity_risk


def make_rm(**risk_overrides):
    params = {
        "max_open_positions": 12,
        "one_trade_per_symbol": True,
        "cooldown_days": 3,
        "daily_loss_limit_pct": 0.03,
        "cash_buffer_pct": 0.05,
    }
    params.update(risk_overrides)
    risk = RiskConfig(**params)
    sizing = SizingConfig(per_trade_notional=500, allow_fractional=True)
    return RiskManager(risk, sizing, base_currency="EUR")


def rich_snap(**kw):
    base = {"equity": 10_000, "buying_power": 10_000, "cash": 10_000, "daily_pnl": 0.0,
            "open_positions": set(), "cooldowns": {}}
    base.update(kw)
    return AccountSnapshot(**base)


# ── sizing / rounding ─────────────────────────────────────────────────────────
def test_compute_quantity_fractional():
    # 500 EUR * 1.0 fx / 50 price = 10 shares
    assert compute_quantity(500, 50, 1.0, True) == pytest.approx(10.0)


def test_compute_quantity_fx_conversion():
    # 500 EUR * 1.10 = 550 USD / 100 = 5.5 shares
    assert compute_quantity(500, 100, 1.10, True) == pytest.approx(5.5)


def test_compute_quantity_floor_when_no_fractional():
    # 500 / 70 = 7.14 -> floor 7
    assert compute_quantity(500, 70, 1.0, False) == 7


def test_compute_quantity_rounds_to_zero():
    # 500 / 600 = 0.83 -> floor 0 when fractional disabled
    assert compute_quantity(500, 600, 1.0, False) == 0


def test_entry_quantity_zero_is_rejected():
    rm = make_rm()
    rm.sizing = SizingConfig(per_trade_notional=500, allow_fractional=False)
    d = rm.evaluate_entry("AAPL", price=600, asof=date(2024, 1, 10), snap=rich_snap())
    assert not d.approved
    assert d.reason == "quantity_rounds_to_zero"


# ── risk_per_trade sizing (Phase 5.2) ─────────────────────────────────────────
def test_compute_quantity_risk_basic():
    # risk 1% of 10k = 100 base; stop 4 -> 100 * 1.0 / 4 = 25 shares
    assert compute_quantity_risk(10_000, 0.01, 4.0, 1.0, True) == pytest.approx(25.0)


def test_compute_quantity_risk_fx_and_floor():
    # fx 1.08: 100 * 1.08 / 4 = 27 shares
    assert compute_quantity_risk(10_000, 0.01, 4.0, 1.08, True) == pytest.approx(27.0)
    # 100 / 3 = 33.33 -> floor 33 without fractional
    assert compute_quantity_risk(10_000, 0.01, 3.0, 1.0, False) == 33


def test_compute_quantity_risk_guards():
    assert compute_quantity_risk(10_000, 0.01, 0.0, 1.0, True) == 0.0   # no stop distance
    assert compute_quantity_risk(0, 0.01, 4.0, 1.0, True) == 0.0        # no equity


def _rm_risk(**sizing_kw):
    rm = make_rm()
    base = {"method": "risk_per_trade", "risk_per_trade_pct": 0.01,
            "max_position_pct": 0.20, "allow_fractional": True}
    base.update(sizing_kw)
    rm.sizing = SizingConfig(**base)
    return rm


def test_entry_risk_per_trade_sizes_to_stop():
    # equity 10k, risk 1% = 100; stop_distance 5 -> 20 shares; cost 1000 < buffers.
    rm = _rm_risk()
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=rich_snap(),
                          stop_distance=5.0)
    assert d.approved
    assert d.quantity == pytest.approx(20.0)


def test_entry_risk_per_trade_caps_position():
    # Tiny stop wants 200 shares (cost 10k, would breach the buffer); cap at
    # 20% of equity = 2000 / 50 = 40 shares keeps it approved.
    rm = _rm_risk()
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=rich_snap(),
                          stop_distance=0.5)
    assert d.approved
    assert d.quantity == pytest.approx(40.0)


def test_entry_risk_per_trade_falls_back_without_stop():
    # No stop_distance -> fixed notional (per_trade_notional 500 / 50 = 10 shares).
    rm = _rm_risk(per_trade_notional=500)
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=rich_snap())
    assert d.approved
    assert d.quantity == pytest.approx(10.0)


# ── caps ──────────────────────────────────────────────────────────────────────
def test_max_open_positions_cap():
    rm = make_rm(max_open_positions=2)
    snap = rich_snap(open_positions={"AAA", "BBB"})
    d = rm.evaluate_entry("CCC", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "max_open_positions"


def test_one_trade_per_symbol():
    rm = make_rm()
    snap = rich_snap(open_positions={"AAPL"})
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "already_in_position"


# ── cooldown ──────────────────────────────────────────────────────────────────
def test_cooldown_blocks_recent_reentry():
    rm = make_rm(cooldown_days=3)
    snap = rich_snap(cooldowns={"AAPL": date(2024, 1, 8)})
    # 2 days later -> still in cooldown
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "cooldown_active"


def test_cooldown_expired_allows_entry():
    rm = make_rm(cooldown_days=3)
    snap = rich_snap(cooldowns={"AAPL": date(2024, 1, 8)})
    # 3 days later -> cooldown elapsed
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 11), snap=snap)
    assert d.approved


# ── kill switch ───────────────────────────────────────────────────────────────
def test_kill_switch_trips_below_limit():
    rm = make_rm(daily_loss_limit_pct=0.03)
    snap = rich_snap(daily_pnl=-301)  # < -3% of 10_000
    assert rm.kill_switch_active(snap)
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "kill_switch_active"


def test_kill_switch_not_tripped_within_limit():
    rm = make_rm(daily_loss_limit_pct=0.03)
    snap = rich_snap(daily_pnl=-299)  # > -3% of 10_000
    assert not rm.kill_switch_active(snap)


# ── cash buffer / buying power ────────────────────────────────────────────────
def test_cash_buffer_breached():
    rm = make_rm(cash_buffer_pct=0.05)
    # equity 10k -> min cash 500. cash 600, order ~500 -> would leave 100 < 500.
    snap = rich_snap(equity=10_000, cash=600, buying_power=10_000)
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "cash_buffer_breached"


def test_insufficient_buying_power():
    rm = make_rm(cash_buffer_pct=0.0)
    snap = rich_snap(equity=10_000, cash=10_000, buying_power=100)
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=snap)
    assert not d.approved
    assert d.reason == "insufficient_buying_power"


def test_happy_path_entry_approved():
    rm = make_rm()
    d = rm.evaluate_entry("AAPL", price=50, asof=date(2024, 1, 10), snap=rich_snap())
    assert d.approved
    assert d.quantity == pytest.approx(10.0)  # 500/50
    assert d.reason == "approved"
