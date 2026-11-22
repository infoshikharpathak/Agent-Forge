from __future__ import annotations

"""
SQLite persistence for agent-forge run traces.

Zero config, local by default (``./agent_forge_traces.db``, override via
AGENT_FORGE_DB_PATH). Traces are append-only — insert_run() is the only
write path; there is deliberately no update/delete function.
"""

import json
import os
import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path("agent_forge_traces.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT,
    timestamp TEXT NOT NULL,
    task TEXT NOT NULL,
    routing_tier TEXT NOT NULL,
    app_id TEXT,
    framework_used TEXT,
    agents_spawned TEXT NOT NULL,
    guardrails_triggered TEXT NOT NULL,
    total_latency_ms REAL NOT NULL,
    total_cost_usd REAL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    error TEXT,
    report TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL
);
"""


def _db_path() -> Path:
    override = os.getenv("AGENT_FORGE_DB_PATH", "").strip()
    return Path(override) if override else _DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    # CREATE TABLE IF NOT EXISTS doesn't alter an already-existing table —
    # migrate older DB files that predate later columns.
    for ddl in (
        "ALTER TABLE runs ADD COLUMN report TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE runs ADD COLUMN trace_id TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    _allow_null_cost(conn)
    return conn


def _allow_null_cost(conn: sqlite3.Connection) -> None:
    """Older DB files declare total_cost_usd NOT NULL, from when unpriced
    models were stored as $0. SQLite can't drop a NOT NULL constraint in
    place, so rebuild the table once with the current schema."""
    cols = {r["name"]: r for r in conn.execute("PRAGMA table_info(runs)")}
    if not cols["total_cost_usd"]["notnull"]:
        return
    names = ", ".join(cols)
    with conn:
        conn.execute("ALTER TABLE runs RENAME TO runs_old")
        conn.execute(_SCHEMA)
        conn.execute(f"INSERT INTO runs ({names}) SELECT {names} FROM runs_old")
        conn.execute("DROP TABLE runs_old")


def insert_run(row: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, trace_id, timestamp, task, routing_tier, app_id, framework_used,
                agents_spawned, guardrails_triggered, total_latency_ms, total_cost_usd,
                total_input_tokens, total_output_tokens, outcome, error, report, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["run_id"], row.get("trace_id"), row["timestamp"], row["task"], row["routing_tier"],
                row.get("app_id"), row.get("framework_used"),
                json.dumps(row["agents_spawned"]), json.dumps(row["guardrails_triggered"]),
                row["total_latency_ms"], row["total_cost_usd"],
                row["total_input_tokens"], row["total_output_tokens"],
                row["outcome"], row.get("error"), row.get("report", ""),
                json.dumps(row.get("metadata", {})),
            ),
        )


def _deserialize(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["agents_spawned"] = json.loads(d["agents_spawned"])
    d["guardrails_triggered"] = json.loads(d["guardrails_triggered"])
    d["metadata"] = json.loads(d["metadata"])
    return d


def query_runs(
    *,
    limit: int = 10,
    app_id: str | None = None,
    status: str | None = None,
    since: str | None = None,
    cost_above: float | None = None,
    trace_id: str | None = None,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if trace_id is not None:
        clauses.append("trace_id = ?")
        params.append(trace_id)
    if app_id is not None:
        clauses.append("app_id = ?")
        params.append(app_id)
    if status is not None:
        clauses.append("outcome = ?")
        params.append(status)
    if since is not None:
        clauses.append("timestamp >= ?")
        params.append(since)
    if cost_above is not None:
        clauses.append("total_cost_usd > ?")
        params.append(cost_above)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY timestamp DESC LIMIT ?", params
        ).fetchall()

    return [_deserialize(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _deserialize(row) if row is not None else None
