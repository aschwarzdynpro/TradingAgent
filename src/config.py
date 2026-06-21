"""Typed configuration.

Loads two sources and keeps them strictly separated:

* ``.env``       -> connection details + the LIVE_TRADING master switch (secrets).
* ``config.yaml`` -> strategy / risk / sizing / universe parameters (no secrets).

Strategy and risk parameters live ONLY in the YAML config — never hard-coded in
the logic modules. This is the single source of truth used by backtest, paper
and live alike, which is what keeps backtest and live behaviour identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# .env — connection + the live switch
# ─────────────────────────────────────────────────────────────────────────────
class EnvSettings(BaseSettings):
    """Environment / secret settings, read from ``.env`` (or the real env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    live_trading: bool = False

    ib_host: str = "127.0.0.1"
    ib_paper_port: int = 4002
    ib_live_port: int = 4001
    ib_client_id: int = 17
    ib_timeout: float = 20.0
    ib_account: str = ""

    db_path: str = "data/trading_agent.sqlite"
    config_path: str = "config.yaml"
    log_level: str = "INFO"

    @property
    def port(self) -> int:
        """The port to actually connect to, derived from the live switch."""
        return self.ib_live_port if self.live_trading else self.ib_paper_port

    @property
    def mode(self) -> Literal["LIVE", "PAPER"]:
        return "LIVE" if self.live_trading else "PAPER"


# ─────────────────────────────────────────────────────────────────────────────
# config.yaml — strategy / risk / universe
# ─────────────────────────────────────────────────────────────────────────────
class AccountConfig(BaseModel):
    base_currency: str = "EUR"
    total_capital: float = 10_000.0


class SizingConfig(BaseModel):
    # "fixed_notional" = v1 (per_trade_notional per trade). "risk_per_trade" =
    # size so a stop-out loses risk_per_trade_pct of equity (volatility/equity
    # scaled), capped at max_position_pct of equity.
    method: Literal["fixed_notional", "risk_per_trade"] = "fixed_notional"
    per_trade_notional: float = 500.0
    allow_fractional: bool = True
    risk_per_trade_pct: float = 0.01   # risk_per_trade mode: fraction of equity risked to the stop
    max_position_pct: float = 0.20     # risk_per_trade mode: cap one position at this fraction of equity


class StrategyConfig(BaseModel):
    # "mean_reversion" = v1 (RSI dip in an uptrend). "trend_momentum" = ride the
    # trend while in an uptrend with positive 12-1 momentum (Phase 5 candidate).
    mode: Literal["mean_reversion", "trend_momentum"] = "mean_reversion"
    timeframe: str = "1 day"
    sma_trend: int = 200
    rsi_period: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    atr_period: int = 14
    atr_mult: float = 3.0
    # Momentum (trend_momentum mode only): return from mom_lookback bars ago to
    # mom_skip bars ago, entered when > mom_threshold.
    mom_lookback: int = 252
    mom_skip: int = 21
    mom_threshold: float = 0.0
    # Market-regime filter: when on, no new entries while the regime symbol trades
    # below its SMA (risk-off); with regime_exit, open positions are also closed.
    use_regime_filter: bool = False
    regime_symbol: str = "SPY"
    regime_sma: int = 200
    regime_exit: bool = False

    @field_validator("rsi_entry", "rsi_exit")
    @classmethod
    def _rsi_bounds(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("RSI thresholds must be within [0, 100]")
        return v

    @field_validator("mom_lookback")
    @classmethod
    def _mom_window(cls, v: int, info) -> int:
        if v <= info.data.get("mom_skip", 0):
            raise ValueError("mom_lookback must be greater than mom_skip")
        return v


class RiskConfig(BaseModel):
    max_open_positions: int = 12
    one_trade_per_symbol: bool = True
    cooldown_days: int = 3
    daily_loss_limit_pct: float = 0.03
    auto_flatten_on_kill: bool = False
    cash_buffer_pct: float = 0.05


class ExecutionConfig(BaseModel):
    order_type: Literal["LMT", "MOO", "MOC", "MKT"] = "LMT"
    limit_offset_pct: float = 0.002
    tif: str = "DAY"
    outside_rth: bool = False


class SchedulerConfig(BaseModel):
    timezone: str = "Europe/Berlin"
    run_time: str = "23:30"
    run_on_weekends: bool = False


class BacktestConfig(BaseModel):
    """Backtest cost/benchmark model. Defaults track IBKR US-equity *fixed* pricing:
    $0.005/share, $1.00 minimum per order, capped at 1% of trade value."""

    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission_pct: float = 0.01
    slippage_bps: float = 2.0
    benchmark: str | None = "SPY"  # buy-and-hold reference for alpha; null to disable


class DataConfig(BaseModel):
    history_duration: str = "2 Y"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    cache_dir: str = "data/cache"


class SymbolSpec(BaseModel):
    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str | None = Field(default=None, alias="primaryExchange")

    model_config = SettingsConfigDict(populate_by_name=True)


class YamlConfig(BaseModel):
    account: AccountConfig = AccountConfig()
    sizing: SizingConfig = SizingConfig()
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()
    active_universe: str = "v1_us_usd"
    universe: dict[str, list[SymbolSpec]] = Field(default_factory=dict)

    @property
    def symbols(self) -> list[SymbolSpec]:
        """The symbol list for the currently active universe."""
        if self.active_universe not in self.universe:
            raise ValueError(
                f"active_universe '{self.active_universe}' not found in universe; "
                f"available: {list(self.universe)}"
            )
        return self.universe[self.active_universe]


class AppConfig(BaseModel):
    """The fully assembled configuration handed to every component."""

    env: EnvSettings
    cfg: YamlConfig


def _load_yaml(path: str | Path) -> YamlConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file '{p}' not found. Copy/edit config.yaml in the repo root."
        )
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return YamlConfig.model_validate(raw)


def load_config(env: EnvSettings | None = None) -> AppConfig:
    """Load env + yaml into a single typed config object."""
    env = env or EnvSettings()
    cfg = _load_yaml(env.config_path)
    return AppConfig(env=env, cfg=cfg)
