"""Backtest engine over historical daily bars.

Uses the SAME ``strategy.evaluate`` and the SAME ``risk`` sizing/caps as the live
agent — there is no separate backtest strategy. Look-ahead is avoided by the
execution model: a signal is produced from the *close* of day ``t`` and filled at
the *open* of day ``t+1``.

Currency: the backtest runs in a single currency (fx = 1.0). The v1 (USD) and v2
(EUR) universes are each single-currency, so ``per_trade_notional`` is interpreted
in that universe's currency. Multi-currency FX simulation is out of scope here.

Run:
    python -m src.backtest                 # from CSV cache (data/cache/*.csv)
    python -m src.backtest --synthetic     # generate synthetic data (offline demo)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .config import AppConfig, BacktestConfig, load_config
from .data import generate_synthetic, load_cache
from .risk import AccountSnapshot, RiskManager
from .strategy import PositionState, SignalType, compute_indicators, evaluate


@dataclass
class BTPosition:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: date
    highest_high: float


@dataclass
class Trade:
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: float
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def hold_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: AppConfig, starting_cash: float,
                 costs: BacktestConfig | None = None):
        self.cfg = cfg
        self.sp = cfg.cfg.strategy
        self.rm = RiskManager(cfg.cfg.risk, cfg.cfg.sizing, cfg.cfg.account.base_currency)
        self.starting_cash = starting_cash
        self.costs = costs or cfg.cfg.backtest
        self.slip = self.costs.slippage_bps / 10_000.0

    def _commission(self, shares: float, price: float) -> float:
        """IBKR-style per-share commission with a per-order minimum, capped at a
        percentage of trade value (the cap overrides the minimum for tiny orders)."""
        if shares <= 0:
            return 0.0
        notional = shares * price
        fee = max(self.costs.min_commission, self.costs.commission_per_share * shares)
        return min(fee, self.costs.max_commission_pct * notional)

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResult:
        # Precompute indicators per symbol (causal -> no look-ahead).
        ind = {s: compute_indicators(df, self.sp) for s, df in data.items() if len(df)}
        # Unified trading calendar across all symbols.
        all_dates = sorted(set().union(*[set(df.index) for df in ind.values()]))

        cash = self.starting_cash
        positions: dict[str, BTPosition] = {}
        cooldowns: dict[str, date] = {}
        trades: list[Trade] = []
        pending_entries: dict[str, float] = {}  # symbol -> qty to buy at next open
        pending_exits: dict[str, str] = {}      # symbol -> exit reason at next open

        equity_points: list[tuple[pd.Timestamp, float]] = []
        prev_equity = self.starting_cash

        for ts in all_dates:
            d = ts.date()

            # 1) Fill yesterday's queued orders at TODAY's open.
            for sym, reason in list(pending_exits.items()):
                if sym in positions and ts in ind[sym].index:
                    px = float(ind[sym].loc[ts, "open"]) * (1 - self.slip)  # sell: worse = lower
                    pos = positions.pop(sym)
                    cash += pos.quantity * px - self._commission(pos.quantity, px)
                    trades.append(Trade(sym, pos.entry_date, d, pos.entry_price, px,
                                        pos.quantity, reason))
                    cooldowns[sym] = d
            pending_exits.clear()

            for sym, qty in list(pending_entries.items()):
                if sym not in positions and ts in ind[sym].index:
                    px = float(ind[sym].loc[ts, "open"]) * (1 + self.slip)  # buy: worse = higher
                    cost = qty * px + self._commission(qty, px)
                    if cost <= cash and qty > 0:
                        cash -= cost
                        positions[sym] = BTPosition(sym, qty, px, d,
                                                    highest_high=float(ind[sym].loc[ts, "high"]))
            pending_entries.clear()

            # 2) Update trailing highs for open positions using today's high.
            for sym, pos in positions.items():
                if ts in ind[sym].index:
                    pos.highest_high = max(pos.highest_high, float(ind[sym].loc[ts, "high"]))

            # 3) Mark-to-market equity at today's close.
            pos_value = 0.0
            for sym, pos in positions.items():
                if ts in ind[sym].index:
                    pos_value += pos.quantity * float(ind[sym].loc[ts, "close"])
            equity = cash + pos_value
            equity_points.append((ts, equity))
            daily_pnl = equity - prev_equity
            prev_equity = equity

            # 4) Generate signals from TODAY's close -> queue for tomorrow's open.
            snap = AccountSnapshot(
                equity=equity, buying_power=cash, cash=cash, daily_pnl=daily_pnl,
                open_positions=set(positions.keys()), cooldowns=cooldowns,
            )
            for sym, df in ind.items():
                if ts not in df.index:
                    continue
                pos = positions.get(sym)
                window = df.loc[:ts]
                window.attrs["symbol"] = sym
                pstate = (
                    PositionState(sym, pos.quantity, pos.entry_price, pos.entry_date,
                                  pos.highest_high)
                    if pos else None
                )
                res = evaluate(window, pstate, self.sp)
                if res.signal is SignalType.EXIT and sym in positions:
                    pending_exits[sym] = res.reason
                elif res.signal is SignalType.ENTER_LONG and sym not in positions:
                    decision = self.rm.evaluate_entry(sym, res.price, d, snap, fx_base_to_instrument=1.0)
                    if decision.approved:
                        pending_entries[sym] = decision.quantity
                        # Reserve a slot/cash so multiple same-day entries respect caps.
                        snap.open_positions.add(sym)
                        snap.cash -= decision.quantity * res.price

        eq = pd.Series(dict(equity_points)).sort_index()
        metrics = compute_metrics(eq, trades, self.starting_cash)
        return BacktestResult(equity_curve=eq, trades=trades, metrics=metrics)


def compute_metrics(equity: pd.Series, trades: list[Trade], starting_cash: float) -> dict:
    if equity.empty:
        return {}
    rets = equity.pct_change().dropna()
    n_days = len(equity)
    years = max(n_days / 252.0, 1e-9)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if equity.iloc[0] > 0 else float("nan")

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else float("nan")

    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) if trades else float("nan")
    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else float("nan")
    total_pnl = sum(t.pnl for t in trades)

    return {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "num_trades": len(trades),
        "win_rate": float(win_rate) if win_rate == win_rate else float("nan"),
        "avg_hold_days": avg_hold,
        "total_trade_pnl": float(total_pnl),
        "trading_days": n_days,
    }


def benchmark_metrics(
    bench: pd.DataFrame, equity_index: pd.Index, starting_cash: float,
    costs: BacktestConfig, strategy_metrics: dict,
) -> dict:
    """Buy-and-hold the benchmark over the strategy's date range and compute
    alpha (excess CAGR / total return vs. the strategy).

    Fully invests ``starting_cash`` at the first overlapping open (one commission)
    and marks to market at each close, so the comparison uses the same cost model
    and the same trading calendar as the strategy.
    """
    overlap = bench.index.intersection(equity_index)
    if len(overlap) < 2:
        return {}
    b = bench.loc[overlap].sort_index()
    first_open = float(b.iloc[0]["open"])
    if first_open <= 0:
        return {}
    fee = max(costs.min_commission, costs.commission_per_share *
              (starting_cash / first_open))
    shares = (starting_cash - fee) / first_open
    eq = shares * b["close"].astype(float)
    bm = compute_metrics(eq, [], starting_cash)
    out = {f"benchmark_{k}": v for k, v in bm.items()}
    out["benchmark_symbol"] = bench.attrs.get("symbol", "")
    if "cagr" in strategy_metrics and "cagr" in bm:
        out["alpha_cagr"] = float(strategy_metrics["cagr"] - bm["cagr"])
    if "total_return" in strategy_metrics and "total_return" in bm:
        out["alpha_total_return"] = float(
            strategy_metrics["total_return"] - bm["total_return"])
    return out


def _print_report(result: BacktestResult) -> None:
    m = result.metrics
    print("\n" + "=" * 56)
    print(" BACKTEST RESULTS")
    print("=" * 56)
    if not m:
        print(" No data / no equity curve produced.")
        return
    print(f" Period (trading days) : {m['trading_days']}")
    print(f" Start equity          : {m['start_equity']:,.2f}")
    print(f" End equity            : {m['end_equity']:,.2f}")
    print(f" Total return          : {m['total_return']*100:,.2f}%")
    print(f" CAGR                  : {m['cagr']*100:,.2f}%")
    print(f" Max drawdown          : {m['max_drawdown']*100:,.2f}%")
    print(f" Sharpe (daily, ann.)  : {m['sharpe']:,.2f}")
    print(f" Number of trades      : {m['num_trades']}")
    wr = m["win_rate"]
    print(f" Win rate              : {wr*100:,.1f}%" if wr == wr else " Win rate              : n/a")
    ah = m["avg_hold_days"]
    print(f" Avg hold (days)       : {ah:,.1f}" if ah == ah else " Avg hold (days)       : n/a")
    print(f" Total trade P&L       : {m['total_trade_pnl']:,.2f}")
    if "benchmark_total_return" in m:
        sym = m.get("benchmark_symbol") or "benchmark"
        print("-" * 56)
        print(f" Benchmark ({sym}) buy & hold:")
        print(f"   Total return        : {m['benchmark_total_return']*100:,.2f}%")
        print(f"   CAGR                : {m['benchmark_cagr']*100:,.2f}%")
        print(f"   Max drawdown        : {m['benchmark_max_drawdown']*100:,.2f}%")
        if "alpha_cagr" in m:
            print(f" Alpha (CAGR)          : {m['alpha_cagr']*100:,.2f}%")
        if "alpha_total_return" in m:
            print(f" Alpha (total return)  : {m['alpha_total_return']*100:,.2f}%")
    print("=" * 56 + "\n")


def load_universe_data(cfg: AppConfig, synthetic: bool) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for spec in cfg.cfg.symbols:
        if synthetic:
            data[spec.symbol] = generate_synthetic(spec.symbol)
        else:
            df = load_cache(cfg.cfg.data.cache_dir, spec.symbol)
            if df is not None and len(df):
                data[spec.symbol] = df
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the trading strategy.")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate synthetic data (offline demo; not real data)")
    parser.add_argument("--cash", type=float, default=None,
                        help="starting cash (default: account.total_capital)")
    parser.add_argument("--out", default="data/backtest_equity.csv",
                        help="path to write the equity curve CSV")
    parser.add_argument("--trades-out", default="data/backtest_trades.csv",
                        help="path to write the trades CSV")
    args = parser.parse_args(argv)

    cfg = load_config()
    starting_cash = args.cash if args.cash is not None else cfg.cfg.account.total_capital

    data = load_universe_data(cfg, synthetic=args.synthetic)
    if not data:
        print("No data available. Provide CSV caches in "
              f"'{cfg.cfg.data.cache_dir}' or run with --synthetic.")
        return 1

    bt = Backtester(cfg, starting_cash=starting_cash)
    result = bt.run(data)

    # Benchmark (buy-and-hold) + alpha, if configured and data is available.
    bench_sym = cfg.cfg.backtest.benchmark
    if bench_sym and not result.equity_curve.empty:
        bench = (generate_synthetic(bench_sym) if args.synthetic
                 else load_cache(cfg.cfg.data.cache_dir, bench_sym))
        if bench is not None and len(bench):
            bench.attrs["symbol"] = bench_sym
            result.metrics.update(benchmark_metrics(
                bench, result.equity_curve.index, starting_cash,
                cfg.cfg.backtest, result.metrics))
        else:
            print(f"(benchmark '{bench_sym}' unavailable — skipping alpha)")

    _print_report(result)

    result.equity_curve.to_csv(args.out, index_label="date", header=["equity"])
    if result.trades:
        pd.DataFrame([t.__dict__ | {"pnl": t.pnl, "hold_days": t.hold_days} for t in result.trades]) \
            .to_csv(args.trades_out, index=False)
    print(f"Equity curve -> {args.out}")
    print(f"Trades       -> {args.trades_out if result.trades else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
