"""Cross-sectional momentum backtest (Phase 6).

Goal: beat SPY on a *risk-adjusted* basis (Sharpe > the market's). A single-name
mean-reversion/momentum signal on a handful of tickers can't — cross-sectional
equity alpha lives in ranking a *broad* universe and holding the strongest names.

This is a portfolio engine, deliberately separate from the per-symbol daily event
engine in ``backtest.py``:

  every ``rebalance`` trading days: rank the universe by 12-1 momentum, hold the
  top-K equal-weight (long only), optionally stand down to cash when the market
  regime is risk-off. Costs are charged on turnover.

It produces a daily equity curve so the metrics (annualised on 252) are directly
comparable to SPY and to the other backtests in this repo.

    python -m src.xsec --universe xsec_us --top-k 12 --regime
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .backtest import compute_metrics
from .config import load_config
from .data import load_cache
from .strategy import compute_regime


def load_panel(cache_dir: str, symbols: list[str]) -> pd.DataFrame:
    """Daily close panel (index=dates, columns=symbols). Names with different
    start dates simply carry NaN until they have history."""
    series = {}
    for s in symbols:
        df = load_cache(cache_dir, s)
        if df is not None and len(df):
            series[s] = df["close"].astype(float)
    panel = pd.DataFrame(series).sort_index()
    return panel.ffill(limit=5)  # bridge the odd missing print, not long gaps


def ann_sharpe(daily_returns: pd.Series) -> float:
    sd = daily_returns.std()
    return float(np.sqrt(252) * daily_returns.mean() / sd) if sd > 0 else float("nan")


def run_xsec(
    panel: pd.DataFrame, *, top_k: int, lookback: int, skip: int, rebalance: int,
    cost_bps: float, starting_cash: float,
    regime: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    """Return (daily equity curve, info). ``regime`` is a risk-on bool series; when
    risk-off at a rebalance, that period is held in cash."""
    momentum = panel.shift(skip) / panel.shift(lookback) - 1.0
    dates = panel.index
    rebal_pos = range(lookback, len(dates), rebalance)
    rebal_dates = [dates[i] for i in rebal_pos]
    if not rebal_dates:
        raise ValueError("Not enough history for one rebalance — lower --lookback.")

    weights = pd.DataFrame(0.0, index=rebal_dates, columns=panel.columns)
    n_cash = 0
    for d in rebal_dates:
        risk_on = True if regime is None or d not in regime.index else bool(regime.loc[d])
        if not risk_on:
            n_cash += 1
            continue  # weights stay 0 -> cash for this period
        scores = momentum.loc[d].dropna()
        scores = scores[panel.loc[d, scores.index].notna()]  # price must exist too
        if scores.empty:
            continue
        picks = scores.nlargest(min(top_k, len(scores))).index
        weights.loc[d, picks] = 1.0 / len(picks)

    # Daily weights: hold each rebalance's targets until the next; apply yesterday's
    # decided weights to today's return (no look-ahead). Costs charged on turnover.
    daily_w = weights.reindex(dates, method="ffill").fillna(0.0)
    eff_w = daily_w.shift(1).fillna(0.0)
    daily_rets = panel.pct_change().fillna(0.0)
    port = (eff_w * daily_rets).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # initial deployment
    cost = pd.Series(0.0, index=dates)
    cost.loc[turnover.index] = turnover * (cost_bps / 1e4)
    port = port - cost

    active = port.loc[rebal_dates[0]:]
    equity = starting_cash * (1.0 + active).cumprod()
    info = {
        "rebalances": len(rebal_dates),
        "cash_periods": n_cash,
        "avg_turnover": float(turnover.mean()),
        "avg_names": float((weights > 0).sum(axis=1).replace(0, np.nan).mean()),
        "sharpe": ann_sharpe(active),
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
    }
    return equity, info


def spy_returns(cache_dir: str, bench: str, index: pd.Index) -> pd.Series:
    """SPY daily returns aligned to ``index`` (for the per-fold comparison)."""
    df = load_cache(cache_dir, bench)
    if df is None or not len(df):
        return pd.Series(dtype=float)
    return df["close"].astype(float).reindex(index).ffill().pct_change()


def fold_report(strat_rets: pd.Series, spy_rets: pd.Series, n_folds: int) -> None:
    """Per-sub-period strat-vs-SPY Sharpe + total return — does the edge persist
    in time, or is one lucky stretch carrying the full-period number?"""
    idx = strat_rets.index
    bounds = [int(round(i * len(idx) / n_folds)) for i in range(n_folds + 1)]
    print(f"\n Per-fold persistence ({n_folds} contiguous sub-periods):")
    print(f" {'window':<25}{'strat Sh':>10}{'SPY Sh':>9}{'strat ret':>11}{'SPY ret':>10}{'win?':>6}")
    print(" " + "-" * 65)
    wins = 0
    for a, b in zip(bounds, bounds[1:], strict=False):
        sl = idx[a:b]
        sr, br = strat_rets.loc[sl].dropna(), spy_rets.loc[sl].dropna()
        if len(sr) < 2:
            continue
        s_sh, b_sh = ann_sharpe(sr), ann_sharpe(br)
        s_ret = (1 + sr).prod() - 1
        b_ret = (1 + br).prod() - 1
        win = s_sh > b_sh
        wins += win
        print(f" {str(sl[0].date())+'->'+str(sl[-1].date()):<25}{s_sh:>10.2f}{b_sh:>9.2f}"
              f"{s_ret*100:>10.1f}%{b_ret*100:>9.1f}%{'  Y' if win else '  n':>6}")
    print(f" => strategy wins {wins}/{n_folds} folds on Sharpe")


def spy_stats(cache_dir: str, bench: str, index: pd.Index, starting_cash: float) -> dict:
    df = load_cache(cache_dir, bench)
    if df is None or not len(df):
        return {}
    close = df["close"].astype(float).reindex(index).ffill()
    rets = close.pct_change().dropna()
    eq = starting_cash * (close / close.iloc[0])
    m = compute_metrics(eq, [], starting_cash)
    return {"cagr": m.get("cagr"), "max_drawdown": m.get("max_drawdown"),
            "sharpe": ann_sharpe(rets), "total_return": m.get("total_return")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cross-sectional momentum backtest (Phase 6).")
    p.add_argument("--universe", default="xsec_us", help="named universe to rank over")
    p.add_argument("--top-k", type=int, default=12, help="number of names held")
    p.add_argument("--lookback", type=int, default=252, help="momentum lookback (bars)")
    p.add_argument("--skip", type=int, default=21, help="skip most-recent N bars (reversal)")
    p.add_argument("--rebalance", type=int, default=21, help="rebalance every N trading days")
    p.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit turnover, bps")
    p.add_argument("--regime", action="store_true", help="hold cash when SPY < its SMA")
    p.add_argument("--regime-sma", type=int, default=200)
    p.add_argument("--exclude", nargs="+", default=[], metavar="SYM",
                   help="drop these symbols (survivorship-bias robustness check)")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--folds", type=int, default=1,
                   help="split the period into N contiguous sub-periods and report strat vs SPY per fold")
    p.add_argument("--out", default="data/xsec_equity.csv")
    args = p.parse_args(argv)

    cfg = load_config()
    cache_dir = cfg.cfg.data.cache_dir
    if args.universe not in cfg.cfg.universe:
        print(f"Universe '{args.universe}' not found. Available: {list(cfg.cfg.universe)}")
        return 1
    excluded = {s.upper() for s in args.exclude}
    symbols = [s.symbol for s in cfg.cfg.universe[args.universe] if s.symbol.upper() not in excluded]
    if excluded:
        print(f"Excluding {sorted(excluded)} ({len(symbols)} names remain).")
    panel = load_panel(cache_dir, symbols)
    if panel.shape[1] < args.top_k:
        print(f"Only {panel.shape[1]} symbols cached for '{args.universe}' — "
              f"run `python -m src.fetch --universe {args.universe}` first.")
        return 1

    regime = None
    if args.regime:
        bench = load_cache(cache_dir, cfg.cfg.backtest.benchmark or "SPY")
        if bench is not None and len(bench):
            regime = compute_regime(bench["close"].astype(float), args.regime_sma)

    equity, info = run_xsec(
        panel, top_k=args.top_k, lookback=args.lookback, skip=args.skip,
        rebalance=args.rebalance, cost_bps=args.cost_bps, starting_cash=args.cash,
        regime=regime)
    m = compute_metrics(equity, [], args.cash)
    spy = spy_stats(cache_dir, cfg.cfg.backtest.benchmark or "SPY", equity.index, args.cash)

    print("\n" + "=" * 60)
    print(" CROSS-SECTIONAL MOMENTUM (Phase 6)")
    print("=" * 60)
    print(f" Universe / held          : {panel.shape[1]} names, top-{args.top_k}")
    print(f" Momentum / rebalance     : {args.lookback}-{args.skip} / every {args.rebalance}d")
    regime_desc = f"ON (SPY<SMA{args.regime_sma} -> cash)" if args.regime else "off"
    print(f" Regime filter            : {regime_desc}")
    print(f" Period                   : {info['start']} -> {info['end']}")
    print(f" Rebalances (cash periods): {info['rebalances']} ({info['cash_periods']})")
    print(f" Avg held / turnover      : {info['avg_names']:.1f} names / {info['avg_turnover']*100:.0f}% per rebal")
    print("-" * 60)
    print(f" {'metric':<20}{'Strategy':>12}{'SPY B&H':>12}")
    print(f" {'CAGR':<20}{m['cagr']*100:>11.2f}%{(spy.get('cagr') or 0)*100:>11.2f}%")
    print(f" {'Sharpe':<20}{info['sharpe']:>12.2f}{spy.get('sharpe', float('nan')):>12.2f}")
    print(f" {'Max drawdown':<20}{m['max_drawdown']*100:>11.2f}%{(spy.get('max_drawdown') or 0)*100:>11.2f}%")
    print(f" {'Total return':<20}{m['total_return']*100:>11.2f}%{(spy.get('total_return') or 0)*100:>11.2f}%")
    print("-" * 60)
    beat = info["sharpe"] > spy.get("sharpe", float("inf"))
    print(f" => {'STRATEGY beats SPY on Sharpe' if beat else 'does NOT beat SPY on Sharpe yet'}")
    print("=" * 60)

    if args.folds > 1:
        spy_r = spy_returns(cache_dir, cfg.cfg.backtest.benchmark or "SPY", equity.index)
        fold_report(equity.pct_change(), spy_r, args.folds)
    print()

    equity.to_csv(args.out, index_label="date", header=["equity"])
    print(f"Equity curve -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
