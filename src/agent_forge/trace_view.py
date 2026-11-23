"""Rows for showing run traces to people: the Streamlit Traces tab (app.py)
and ``agent-forge traces`` use these, so a cost reads the same everywhere.

A cost of None means the model has no price in agent_forge.pricing. It is
shown as "unknown", never as None, a dash or $0. A run total that leaves out
unpriced agents (metadata["unpriced_models"]) is marked "+ unknown".
"""

from __future__ import annotations

from typing import Any

UNKNOWN = "unknown"


def format_cost(value: float | None, *, partial: bool = False) -> str:
    """USD to 6 decimals; "unknown" for None; "<cost> + unknown" when the
    known part leaves out unpriced models."""
    if value is None:
        return UNKNOWN
    text = f"{value:.6f}"
    return f"{text} + {UNKNOWN}" if partial else text


def run_cost(run: dict[str, Any]) -> str:
    unpriced = (run.get("metadata") or {}).get("unpriced_models") or []
    return format_cost(run.get("total_cost_usd"), partial=bool(unpriced))


def run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Run History table."""
    return [
        {
            "run_id": r["run_id"][:8],
            "timestamp": r["timestamp"][:19],
            "tier": r["routing_tier"],
            "app_id": r.get("app_id") or "-",
            "framework": r.get("framework_used") or "-",
            "outcome": r["outcome"],
            "latency_ms": r["total_latency_ms"],
            "cost_usd": run_cost(r),
            "tokens_in": r["total_input_tokens"],
            "tokens_out": r["total_output_tokens"],
        }
        for r in runs
    ]


def agent_rows(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A run's "Agents spawned" table (without the agents' output)."""
    rows = []
    for a in agents:
        row = {k: v for k, v in a.items() if k != "content"}
        if "cost_usd" in row:
            row["cost_usd"] = format_cost(row["cost_usd"])
        rows.append(row)
    return rows
