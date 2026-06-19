"""Daily scheduler (timezone-aware, Europe/Berlin by default).

Runs the agent once per day at the configured time on completed daily bars. The
default run time is after the US close so the latest daily bar is final; orders
are placed for the next session.
"""

from __future__ import annotations

from collections.abc import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .agent import TradingAgent
from .broker import IBKRBroker
from .config import AppConfig
from .log import get_logger

log = get_logger(__name__)


def _run_cycle(cfg: AppConfig, confirm_live: Callable[[], bool]) -> None:
    """One scheduled cycle: fresh connection, run, disconnect."""
    broker = IBKRBroker(cfg, confirm_live=confirm_live)
    agent = TradingAgent(cfg, broker=broker)
    try:
        agent.run_once()
    except Exception as e:  # never let one failed day kill the scheduler
        log.error("scheduled_run_failed", error=str(e))
        agent.store.record_event("RUN_FAILED", str(e), level="ERROR")
    finally:
        agent.shutdown()


def run_scheduler(cfg: AppConfig, confirm_live: Callable[[], bool] | None = None) -> None:
    confirm_live = confirm_live or (lambda: False)
    sc = cfg.cfg.scheduler
    hour, minute = (int(x) for x in sc.run_time.split(":"))

    day_of_week = "mon-sun" if sc.run_on_weekends else "mon-fri"
    scheduler = BlockingScheduler(timezone=sc.timezone)
    scheduler.add_job(
        _run_cycle,
        trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=sc.timezone),
        args=[cfg, confirm_live],
        id="daily_cycle",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )
    log.info("scheduler_started", timezone=sc.timezone, run_time=sc.run_time,
             day_of_week=day_of_week, mode=cfg.env.mode)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler_stopped")
