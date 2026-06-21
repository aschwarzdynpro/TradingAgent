"""Fetch-only: populate the CSV cache with real daily bars from IBKR.

Phase 1.1 of the roadmap — *trust the numbers before trusting the agent*. This
connects to the (paper) gateway, qualifies every symbol in the active universe
(plus the backtest benchmark), pulls completed daily bars and writes them to the
CSV cache via the same ``data.get_bars`` path the live agent uses.

It deliberately does **nothing else**: no strategy evaluation, no risk checks,
no orders, no store writes. The build order is data -> backtest -> paper -> live,
and this is the data step. Run it, then ``python -m src.backtest`` on real bars.

    python -m src.fetch                  # active universe + benchmark
    python -m src.fetch --skip-benchmark # universe only
    python -m src.fetch --symbols AAPL MSFT
"""

from __future__ import annotations

import argparse
import sys

from .broker import IBKRBroker
from .config import SymbolSpec, load_config
from .data import get_bars
from .log import configure_logging, get_logger

log = get_logger(__name__)

# Polite spacing between historical-data requests to stay clear of IBKR pacing
# limits (it throttles bursts of historical requests).
_PACING_SLEEP_S = 1.0


def _resolve_specs(cfg, symbols: list[str] | None, skip_benchmark: bool) -> list[SymbolSpec]:
    """Universe specs (optionally filtered) plus the benchmark, de-duplicated."""
    universe = cfg.cfg.symbols
    if symbols:
        wanted = {s.upper() for s in symbols}
        specs = [s for s in universe if s.symbol.upper() in wanted]
        missing = wanted - {s.symbol.upper() for s in universe}
        if missing:
            log.warning("symbols_not_in_universe", missing=sorted(missing))
    else:
        specs = list(universe)

    seen = {s.symbol.upper() for s in specs}
    bench = cfg.cfg.backtest.benchmark
    if not skip_benchmark and bench and bench.upper() not in seen and not symbols:
        # Benchmark is a plain US-equity ETF (e.g. SPY); the backtest loads it by
        # symbol from the same cache dir.
        specs.append(SymbolSpec(symbol=bench, exchange="SMART", currency="USD"))
    return specs


def fetch_all(cfg, broker: IBKRBroker, specs: list[SymbolSpec]) -> dict[str, str]:
    """Fetch + cache each spec. Returns symbol -> status line. One failure does
    not abort the rest."""
    results: dict[str, str] = {}
    for i, spec in enumerate(specs):
        try:
            contract = broker.qualify(spec)
            df = get_bars(
                spec.symbol,
                ib=broker.ib,
                contract=contract,
                cache_dir=cfg.cfg.data.cache_dir,
                duration=cfg.cfg.data.history_duration,
                bar_size=cfg.cfg.strategy.timeframe,
                what_to_show=cfg.cfg.data.what_to_show,
                use_rth=cfg.cfg.data.use_rth,
                refresh=True,
            )
            if len(df):
                start = df.index[0].date()
                end = df.index[-1].date()
                status = f"OK   {len(df):>4} bars  {start} -> {end}"
                log.info("fetched", symbol=spec.symbol, bars=len(df), start=str(start), end=str(end))
            else:
                status = "EMPTY (0 bars returned)"
                log.warning("fetched_empty", symbol=spec.symbol)
            results[spec.symbol] = status
        except Exception as e:
            results[spec.symbol] = f"FAIL {e}"
            log.error("fetch_failed", symbol=spec.symbol, error=str(e))
        if i < len(specs) - 1:
            broker.ib.sleep(_PACING_SLEEP_S)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real daily bars from IBKR into the CSV cache (no trading).")
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="only fetch these symbols (must be in the active universe)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="do not also fetch the backtest benchmark (e.g. SPY)")
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_logging(cfg.env.log_level)

    if cfg.env.live_trading:
        log.error("fetch_refuses_live", msg="fetch-only is a paper/data step; set LIVE_TRADING=false")
        print("Refusing to run a data fetch with LIVE_TRADING=true. This is a paper/data step.")
        return 2

    specs = _resolve_specs(cfg, args.symbols, args.skip_benchmark)
    if not specs:
        print("No symbols to fetch (check --symbols against the active universe).")
        return 1

    log.info("fetch_start", mode=cfg.env.mode, host=cfg.env.ib_host, port=cfg.env.port,
             universe=cfg.cfg.active_universe, count=len(specs), cache_dir=cfg.cfg.data.cache_dir)
    print(f"Fetching {len(specs)} symbols into '{cfg.cfg.data.cache_dir}' "
          f"({cfg.cfg.data.history_duration}, {cfg.cfg.strategy.timeframe}, "
          f"{cfg.cfg.data.what_to_show}) via {cfg.env.mode} gateway "
          f"{cfg.env.ib_host}:{cfg.env.port} ...\n")

    broker = IBKRBroker(cfg)
    try:
        # Data-only: connect read-only so this works even when the gateway's API
        # is configured "Read-Only", and never needs order privileges.
        broker.connect(readonly=True)
    except Exception as e:
        log.error("connect_failed", error=str(e))
        print(f"\nCould not connect to IB Gateway at {cfg.env.ib_host}:{cfg.env.port}: {e}\n"
              "Is the gateway running on the PAPER port (4002) with the API enabled "
              "and 127.0.0.1 in Trusted IPs?")
        return 1

    try:
        results = fetch_all(cfg, broker, specs)
    finally:
        broker.disconnect()

    print("\nResults")
    print("-------")
    width = max(len(s) for s in results)
    ok = 0
    for sym, status in results.items():
        print(f"  {sym:<{width}}  {status}")
        if status.startswith("OK"):
            ok += 1
    print(f"\n{ok}/{len(results)} symbols cached to '{cfg.cfg.data.cache_dir}'.")
    if ok < len(results):
        print("Some fetches failed/were empty — check market-data permissions and the log above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
