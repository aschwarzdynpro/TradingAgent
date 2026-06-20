"""Walk-forward parameter sweep — Phase 1.2.

The strategy thresholds in ``config.yaml`` are explicit placeholders. This module
sweeps them, but does so *honestly*: it never reports an in-sample optimum as if
it were the strategy's real edge. Instead it runs a rolling **walk-forward**:

    [ ---- train (in-sample) ---- ][ -- test (out-of-sample) -- ]
                                   ^ params chosen on train are
                                     evaluated on the next, unseen window

Each test window is data the optimiser never saw. Stitching the test windows
gives a true out-of-sample equity curve. If the stitched OOS performance tracks
the in-sample performance, the parameters generalise; if OOS collapses while IS
looks great, that is overfitting — exactly what the split is here to expose.

Warm-up: every measured window is run with ``warmup`` extra bars *before* it so
the SMA/RSI/ATR are valid (and the book is warm) at the window's first day; only
the window itself is scored.

    python -m src.sweep                       # coarse grid, default folds
    python -m src.sweep --grid fine --folds 6
    python -m src.sweep --objective sharpe --min-trades 5
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd

from .backtest import Backtester, benchmark_metrics, compute_metrics
from .config import AppConfig, load_config
from .data import load_cache

# ── parameter grids ──────────────────────────────────────────────────────────
# Only ``sma_trend`` changes the indicator columns; the rest are signal/exit
# thresholds. Keep the coarse grid small enough to walk-forward in minutes.
GRIDS: dict[str, dict[str, list]] = {
    # Mean-reversion (v1) grids.
    "coarse": {
        "sma_trend": [100, 150, 200],
        "rsi_entry": [25, 30, 35],
        "rsi_exit": [50, 55, 60],
        "atr_mult": [2.0, 3.0, 4.0],
        "cooldown_days": [3],
    },
    "fine": {
        "sma_trend": [50, 100, 150, 200],
        "rsi_entry": [20, 25, 30, 35],
        "rsi_exit": [50, 55, 60, 65],
        "atr_mult": [2.0, 2.5, 3.0, 4.0],
        "cooldown_days": [1, 3, 5],
    },
    # Trend/momentum (Phase 5). Deliberately SMALL + economically motivated to
    # resist overfitting: a long-MA trend gate x a 6mo/12mo momentum lookback x
    # the ATR trail width.
    "momentum": {
        "sma_trend": [100, 150, 200],
        "mom_lookback": [126, 252],
        "atr_mult": [3.0, 4.0],
        "cooldown_days": [3],
        "mode": ["trend_momentum"],
    },
}

_OBJECTIVES = ("total_return", "cagr", "sharpe")


@dataclass(frozen=True)
class ParamSet:
    # sma_trend is always swept; the rest carry mode-appropriate defaults so a
    # grid only needs to list the keys it actually varies.
    sma_trend: int
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    atr_mult: float = 3.0
    cooldown_days: int = 3
    mode: str = "mean_reversion"
    mom_lookback: int = 252
    mom_skip: int = 21
    mom_threshold: float = 0.0

    def label(self) -> str:
        if self.mode == "trend_momentum":
            return (f"mom{self.mom_lookback} sma{self.sma_trend} "
                    f"atr{self.atr_mult:g} cd{self.cooldown_days}")
        return (f"sma{self.sma_trend} rsi{self.rsi_entry:g}/{self.rsi_exit:g} "
                f"atr{self.atr_mult:g} cd{self.cooldown_days}")


def grid_params(grid: dict[str, list]) -> list[ParamSet]:
    """Cartesian product of a grid. The grid lists only the keys it varies;
    every key must be a ``ParamSet`` field and ``sma_trend`` must be present."""
    keys = list(grid.keys())
    return [ParamSet(**dict(zip(keys, combo, strict=True)))
            for combo in itertools.product(*(grid[k] for k in keys))]


def cfg_with_params(cfg: AppConfig, p: ParamSet) -> AppConfig:
    """A deep copy of the config with the swept parameters applied."""
    c = cfg.model_copy(deep=True)
    s = c.cfg.strategy
    s.mode = p.mode
    s.sma_trend = p.sma_trend
    s.rsi_entry = p.rsi_entry
    s.rsi_exit = p.rsi_exit
    s.atr_mult = p.atr_mult
    s.mom_lookback = p.mom_lookback
    s.mom_skip = p.mom_skip
    s.mom_threshold = p.mom_threshold
    c.cfg.risk.cooldown_days = p.cooldown_days
    return c


# ── windowed evaluation (warm-up aware) ──────────────────────────────────────
def _slice(data: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {s: df.loc[start:end] for s, df in data.items() if len(df.loc[start:end])}


def evaluate_window(
    cfg: AppConfig, p: ParamSet, data: dict[str, pd.DataFrame], calendar: list[pd.Timestamp],
    w_start_idx: int, w_end_idx: int, warmup: int, starting_cash: float,
) -> tuple[dict, list, pd.Series]:
    """Run the backtest over ``[w_start - warmup, w_end]`` but score only the
    window ``[w_start, w_end]`` (equity renormalised to ``starting_cash`` at the
    window open, trades attributed by entry date). Returns (metrics, trades, eq)."""
    data_start = calendar[max(0, w_start_idx - warmup)]
    w_start, w_end = calendar[w_start_idx], calendar[w_end_idx]
    sliced = _slice(data, data_start, w_end)
    res = Backtester(cfg_with_params(cfg, p), starting_cash=starting_cash).run(sliced)

    eq = res.equity_curve.loc[w_start:w_end]
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return {}, [], eq
    eq = eq / eq.iloc[0] * starting_cash
    trades = [t for t in res.trades if w_start.date() <= t.entry_date <= w_end.date()]
    return compute_metrics(eq, trades, starting_cash), trades, eq


def objective_value(metrics: dict, objective: str, min_trades: int) -> float:
    """Score for picking the in-sample best. Combos with too few trades are
    rejected (-inf) — too little evidence to trust, and a magnet for overfitting."""
    if not metrics or metrics.get("num_trades", 0) < min_trades:
        return float("-inf")
    v = metrics.get(objective)
    return float(v) if v is not None and v == v else float("-inf")


# ── walk-forward driver ──────────────────────────────────────────────────────
@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best: ParamSet
    is_metrics: dict
    oos_metrics: dict
    oos_trades: list
    oos_equity: pd.Series


def build_fold_indices(n: int, warmup: int, train_bars: int, test_bars: int, max_folds: int):
    """Rolling, non-overlapping test windows. Train is the ``train_bars`` window
    immediately before each test; the first train has ``warmup`` bars before it."""
    folds = []
    test_start = warmup + train_bars
    while test_start + test_bars <= n and len(folds) < max_folds:
        folds.append((test_start - train_bars, test_start - 1, test_start,
                      min(test_start + test_bars - 1, n - 1)))
        test_start += test_bars
    return folds


def walk_forward(
    cfg: AppConfig, data: dict[str, pd.DataFrame], params: list[ParamSet],
    *, train_years: float, test_years: float, max_folds: int, warmup: int,
    objective: str, min_trades: int, starting_cash: float,
    progress=lambda *_: None,
) -> list[Fold]:
    calendar = sorted(set().union(*[set(df.index) for df in data.values()]))
    n = len(calendar)
    train_bars, test_bars = int(train_years * 252), int(test_years * 252)
    idx_folds = build_fold_indices(n, warmup, train_bars, test_bars, max_folds)
    if not idx_folds:
        raise ValueError(
            f"Not enough history for walk-forward: have {n} bars, need "
            f">= {warmup + train_bars + test_bars}. Lower --train-years/--test-years "
            "or fetch more history.")

    folds: list[Fold] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(idx_folds):
        # In-sample sweep.
        best_p, best_obj, best_m = None, float("-inf"), {}
        for j, p in enumerate(params):
            m, _, _ = evaluate_window(cfg, p, data, calendar, tr_s, tr_e, warmup, starting_cash)
            obj = objective_value(m, objective, min_trades)
            if obj > best_obj:
                best_p, best_obj, best_m = p, obj, m
            progress(i, len(idx_folds), j, len(params))
        if best_p is None:  # no combo cleared min_trades — fall back to config defaults
            s = cfg.cfg.strategy
            best_p = ParamSet(sma_trend=s.sma_trend, rsi_entry=s.rsi_entry, rsi_exit=s.rsi_exit,
                              atr_mult=s.atr_mult, cooldown_days=cfg.cfg.risk.cooldown_days,
                              mode=s.mode, mom_lookback=s.mom_lookback, mom_skip=s.mom_skip,
                              mom_threshold=s.mom_threshold)
            best_m, _, _ = evaluate_window(cfg, best_p, data, calendar, tr_s, tr_e, warmup, starting_cash)

        # Out-of-sample evaluation of the chosen params.
        oos_m, oos_trades, oos_eq = evaluate_window(
            cfg, best_p, data, calendar, te_s, te_e, warmup, starting_cash)
        folds.append(Fold(i, calendar[tr_s], calendar[tr_e], calendar[te_s], calendar[te_e],
                          best_p, best_m, oos_m, oos_trades, oos_eq))
    return folds


def stitch_oos(folds: list[Fold], starting_cash: float) -> pd.Series:
    """Compound each fold's (renormalised) OOS equity into one continuous curve."""
    pieces, level = [], starting_cash
    for f in folds:
        if f.oos_equity.empty:
            continue
        norm = f.oos_equity / f.oos_equity.iloc[0]
        scaled = norm * level
        level = float(scaled.iloc[-1])
        pieces.append(scaled)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces)


# ── reporting ────────────────────────────────────────────────────────────────
def _pct(x) -> str:
    return f"{x*100:,.2f}%" if isinstance(x, (int, float)) and x == x else "n/a"


def print_report(folds: list[Fold], stitched: pd.Series, agg: dict, objective: str) -> None:
    print("\n" + "=" * 78)
    print(" WALK-FORWARD SWEEP - per fold (params chosen IN-SAMPLE, scored OUT-OF-SAMPLE)")
    print("=" * 78)
    print(f" {'#':>2}  {'test window':<23} {'chosen params':<30} "
          f"{'IS '+objective:>12} {'OOS '+objective:>12} {'OOS trades':>10}")
    print(" " + "-" * 76)
    for f in folds:
        win = f"{f.test_start.date()}->{f.test_end.date()}"
        isv = f.is_metrics.get(objective, float('nan'))
        oosv = f.oos_metrics.get(objective, float('nan'))
        fmt = _pct if objective in ("total_return", "cagr") else (lambda v: f"{v:,.2f}" if v == v else "n/a")
        print(f" {f.index:>2}  {win:<23} {f.best.label():<30} "
              f"{fmt(isv):>12} {fmt(oosv):>12} {f.oos_metrics.get('num_trades', 0):>10}")

    print("\n" + "=" * 78)
    print(" STITCHED OUT-OF-SAMPLE PERFORMANCE (the honest number)")
    print("=" * 78)
    if not agg:
        print(" No OOS equity produced.")
        return
    print(f" OOS period (trading days) : {agg['trading_days']}")
    print(f" Total return              : {_pct(agg['total_return'])}")
    print(f" CAGR                      : {_pct(agg['cagr'])}")
    print(f" Max drawdown              : {_pct(agg['max_drawdown'])}")
    print(f" Sharpe (daily, ann.)      : {agg['sharpe']:,.2f}" if agg['sharpe'] == agg['sharpe'] else " Sharpe: n/a")
    print(f" Number of trades          : {agg['num_trades']}")
    wr = agg.get("win_rate")
    print(f" Win rate                  : {_pct(wr)}" if wr == wr else " Win rate                  : n/a")
    if "benchmark_total_return" in agg:
        print(" " + "-" * 76)
        print(f" Benchmark ({agg.get('benchmark_symbol','?')}) buy & hold over same OOS span:")
        print(f"   Total return            : {_pct(agg['benchmark_total_return'])}")
        print(f"   CAGR                    : {_pct(agg['benchmark_cagr'])}")
        print(f" Alpha (CAGR)              : {_pct(agg.get('alpha_cagr'))}")
        print(f" Alpha (total return)      : {_pct(agg.get('alpha_total_return'))}")

    # Overfitting gap: mean IS vs mean OOS objective.
    is_vals = [f.is_metrics.get(objective) for f in folds if f.is_metrics.get(objective) is not None]
    oos_vals = [f.oos_metrics.get(objective) for f in folds if f.oos_metrics.get(objective) is not None]
    if is_vals and oos_vals:
        mean_is = sum(is_vals) / len(is_vals)
        mean_oos = sum(oos_vals) / len(oos_vals)
        print(" " + "-" * 76)
        lab = _pct if objective in ("total_return", "cagr") else (lambda v: f"{v:,.2f}")
        print(f" Mean IS {objective:<16}: {lab(mean_is)}")
        print(f" Mean OOS {objective:<15}: {lab(mean_oos)}   "
              f"(large IS>>OOS drop => overfitting)")
    print("=" * 78 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward parameter sweep (Phase 1.2).")
    parser.add_argument("--grid", choices=list(GRIDS), default="coarse")
    parser.add_argument("--objective", choices=_OBJECTIVES, default="total_return",
                        help="in-sample selection metric (default: total_return)")
    parser.add_argument("--min-trades", type=int, default=3,
                        help="reject in-sample combos with fewer trades than this")
    parser.add_argument("--train-years", type=float, default=4.0)
    parser.add_argument("--test-years", type=float, default=1.5)
    parser.add_argument("--folds", type=int, default=99, help="max number of folds")
    parser.add_argument("--cash", type=float, default=None,
                        help="starting cash per window (default: account.total_capital)")
    parser.add_argument("--notional", type=float, default=None,
                        help="override sizing.per_trade_notional (deployment experiment)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="override risk.max_open_positions (deployment experiment)")
    parser.add_argument("--out", default="data/walkforward_oos_equity.csv")
    args = parser.parse_args(argv)

    cfg = load_config()
    cash = args.cash if args.cash is not None else cfg.cfg.account.total_capital
    if args.notional is not None:
        cfg.cfg.sizing.per_trade_notional = args.notional
    if args.max_positions is not None:
        cfg.cfg.risk.max_open_positions = args.max_positions

    data: dict[str, pd.DataFrame] = {}
    for spec in cfg.cfg.symbols:
        df = load_cache(cfg.cfg.data.cache_dir, spec.symbol)
        if df is not None and len(df):
            data[spec.symbol] = df
    if not data:
        print(f"No cached data in '{cfg.cfg.data.cache_dir}'. Run `python -m src.fetch` first.")
        return 1

    params = grid_params(GRIDS[args.grid])
    g = GRIDS[args.grid]
    warmup = max(max(g["sma_trend"]), max(g.get("mom_lookback", [0]))) + 60
    print(f"Walk-forward sweep: grid '{args.grid}' = {len(params)} combos/fold, "
          f"objective={args.objective}, min_trades={args.min_trades}, "
          f"train={args.train_years}y test={args.test_years}y, warmup={warmup} bars.")
    print(f"  sizing: per_trade_notional={cfg.cfg.sizing.per_trade_notional:g} "
          f"(~{cfg.cfg.sizing.per_trade_notional / cash * 100:.0f}% of {cash:g}), "
          f"max_open_positions={cfg.cfg.risk.max_open_positions}.")

    last = [-1]
    def progress(i, nf, j, nc):
        pct = int((j + 1) / nc * 100)
        if i != last[0] or j + 1 == nc:
            last[0] = i
            print(f"\r  fold {i+1}/{nf}: optimising {j+1}/{nc} ({pct}%)   ", end="", flush=True)

    folds = walk_forward(
        cfg, data, params, train_years=args.train_years, test_years=args.test_years,
        max_folds=args.folds, warmup=warmup, objective=args.objective,
        min_trades=args.min_trades, starting_cash=cash, progress=progress)
    print()

    stitched = stitch_oos(folds, cash)
    agg = compute_metrics(stitched, [t for f in folds for t in f.oos_trades], cash) if not stitched.empty else {}

    # Benchmark over the stitched OOS span.
    bench_sym = cfg.cfg.backtest.benchmark
    if agg and bench_sym:
        bench = load_cache(cfg.cfg.data.cache_dir, bench_sym)
        if bench is not None and len(bench):
            bench.attrs["symbol"] = bench_sym
            agg.update(benchmark_metrics(bench, stitched.index, cash, cfg.cfg.backtest, agg))

    print_report(folds, stitched, agg, args.objective)
    if not stitched.empty:
        stitched.to_csv(args.out, index_label="date", header=["equity"])
        print(f"Stitched OOS equity -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
