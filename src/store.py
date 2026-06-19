"""SQLite persistence: state + full audit trail.

Every decision the agent makes is recorded here. For an autonomous real-money
system the audit trail is not optional — it is how you reconstruct *why* a
position exists. Tables: signals, orders, fills, positions, equity_curve, events.

The DB file is gitignored; it never leaves the machine.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    asof        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    signal      TEXT NOT NULL,
    reason      TEXT,
    price       REAL,
    indicators  TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    action        TEXT NOT NULL,           -- BUY / SELL
    order_type    TEXT NOT NULL,
    quantity      REAL NOT NULL,
    limit_price   REAL,
    tif           TEXT,
    broker_order_id INTEGER,
    status        TEXT NOT NULL,           -- submitted/filled/partially/cancelled/rejected/paper
    mode          TEXT NOT NULL,           -- PAPER / LIVE / BACKTEST
    reason        TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    action        TEXT NOT NULL,
    quantity      REAL NOT NULL,
    price         REAL NOT NULL,
    commission    REAL,
    broker_order_id INTEGER,
    broker_exec_id  TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    symbol        TEXT PRIMARY KEY,
    quantity      REAL NOT NULL,
    avg_price     REAL NOT NULL,
    entry_date    TEXT NOT NULL,
    highest_high  REAL NOT NULL,
    currency      TEXT,
    updated_ts    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS closed_positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    quantity      REAL NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL NOT NULL,
    entry_date    TEXT NOT NULL,
    exit_date     TEXT NOT NULL,
    pnl           REAL,
    reason        TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts            TEXT PRIMARY KEY,
    equity        REAL NOT NULL,
    cash          REAL,
    positions_value REAL,
    mode          TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    level         TEXT NOT NULL,
    kind          TEXT NOT NULL,           -- e.g. KILL_SWITCH, CONNECT, ORPHAN_CLEANUP
    message       TEXT,
    payload       TEXT
);
"""


@dataclass
class StoredPosition:
    symbol: str
    quantity: float
    avg_price: float
    entry_date: date
    highest_high: float
    currency: str | None = None


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _d(d: date | str | None) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        return d
    return d.isoformat()


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    # ── signals ──────────────────────────────────────────────────────────────
    def record_signal(
        self, asof: date, symbol: str, signal: str, reason: str, price: float, indicators: dict
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO signals(ts, asof, symbol, signal, reason, price, indicators) "
                "VALUES(?,?,?,?,?,?,?)",
                (_now(), _d(asof), symbol, signal, reason, price, json.dumps(indicators, default=str)),
            )

    # ── orders / fills ───────────────────────────────────────────────────────
    def record_order(
        self,
        symbol: str,
        action: str,
        order_type: str,
        quantity: float,
        limit_price: float | None,
        tif: str | None,
        status: str,
        mode: str,
        broker_order_id: int | None = None,
        reason: str | None = None,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO orders(ts, symbol, action, order_type, quantity, limit_price, tif, "
                "broker_order_id, status, mode, reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), symbol, action, order_type, quantity, limit_price, tif,
                 broker_order_id, status, mode, reason),
            )
            return int(cur.lastrowid)

    def update_order_status(self, order_id: int, status: str, broker_order_id: int | None = None) -> None:
        with self._tx() as conn:
            if broker_order_id is not None:
                conn.execute(
                    "UPDATE orders SET status=?, broker_order_id=? WHERE id=?",
                    (status, broker_order_id, order_id),
                )
            else:
                conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))

    def record_fill(
        self, symbol: str, action: str, quantity: float, price: float,
        commission: float | None = None, broker_order_id: int | None = None,
        broker_exec_id: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO fills(ts, symbol, action, quantity, price, commission, "
                "broker_order_id, broker_exec_id) VALUES(?,?,?,?,?,?,?,?)",
                (_now(), symbol, action, quantity, price, commission, broker_order_id, broker_exec_id),
            )

    # ── positions ────────────────────────────────────────────────────────────
    def upsert_position(self, p: StoredPosition) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO positions(symbol, quantity, avg_price, entry_date, highest_high, currency, updated_ts) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, avg_price=excluded.avg_price, "
                "entry_date=excluded.entry_date, highest_high=excluded.highest_high, "
                "currency=excluded.currency, updated_ts=excluded.updated_ts",
                (p.symbol, p.quantity, p.avg_price, _d(p.entry_date), p.highest_high, p.currency, _now()),
            )

    def update_highest_high(self, symbol: str, highest_high: float) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE positions SET highest_high=?, updated_ts=? WHERE symbol=?",
                (highest_high, _now(), symbol),
            )

    def get_position(self, symbol: str) -> StoredPosition | None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        return _row_to_position(row) if row else None

    def get_positions(self) -> dict[str, StoredPosition]:
        with self._tx() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
        return {r["symbol"]: _row_to_position(r) for r in rows}

    def close_position(
        self, symbol: str, exit_price: float, exit_date: date, reason: str
    ) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
            if row:
                pnl = (exit_price - row["avg_price"]) * row["quantity"]
                conn.execute(
                    "INSERT INTO closed_positions(symbol, quantity, entry_price, exit_price, "
                    "entry_date, exit_date, pnl, reason) VALUES(?,?,?,?,?,?,?,?)",
                    (symbol, row["quantity"], row["avg_price"], exit_price,
                     row["entry_date"], _d(exit_date), pnl, reason),
                )
                conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    def last_exit_dates(self) -> dict[str, date]:
        """Most recent exit date per symbol — used for the re-entry cooldown."""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(exit_date) AS d FROM closed_positions GROUP BY symbol"
            ).fetchall()
        return {r["symbol"]: date.fromisoformat(r["d"]) for r in rows if r["d"]}

    # ── equity / events ──────────────────────────────────────────────────────
    def record_equity(
        self, equity: float, cash: float | None = None,
        positions_value: float | None = None, mode: str | None = None,
        ts: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO equity_curve(ts, equity, cash, positions_value, mode) VALUES(?,?,?,?,?) "
                "ON CONFLICT(ts) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
                "positions_value=excluded.positions_value, mode=excluded.mode",
                (ts or _now(), equity, cash, positions_value, mode),
            )

    def record_event(
        self, kind: str, message: str, level: str = "INFO", payload: dict[str, Any] | None = None
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO events(ts, level, kind, message, payload) VALUES(?,?,?,?,?)",
                (_now(), level, kind, message, json.dumps(payload, default=str) if payload else None),
            )


def _row_to_position(row: sqlite3.Row) -> StoredPosition:
    return StoredPosition(
        symbol=row["symbol"],
        quantity=row["quantity"],
        avg_price=row["avg_price"],
        entry_date=date.fromisoformat(row["entry_date"]),
        highest_high=row["highest_high"],
        currency=row["currency"],
    )
