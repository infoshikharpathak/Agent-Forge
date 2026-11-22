from __future__ import annotations

"""
agent-forge CLI.

Run:
    agent-forge run                      — demo pipeline (hardcoded goal, autogen only)
    agent-forge traces --last 10         — query run history from the local SQLite store
    python -m agent_forge [args]         — same, via module invocation (no subcommand = run)
"""

import argparse
import asyncio
import logging

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


GOAL = (
    "Analyse the impact of rising US interest rates on emerging market equities "
    "and produce a concise investment brief."
)

# One task per agent — keyed by agent name as returned by the orchestrator.
TASKS: dict[str, str] = {
    # The orchestrator may return different agent names depending on the goal.
    # These tasks are matched to agents by name at runtime; unmatched agents
    # receive the goal itself as their task.
}


async def _run_demo() -> None:
    orchestrator = Orchestrator(provider="openai")
    factory = AutoGenFactory(provider="openai")
    manager = AgentManager(factory)

    print("\n=== agent-forge demo ===\n")
    print(f"Goal: {GOAL}\n")

    # --- Plan ---
    print("Orchestrator planning agents...\n")
    specs = await orchestrator.plan(GOAL)

    for spec in specs:
        print(f"  [{spec.name}] {spec.role_description}")
    print()

    # --- Spawn ---
    agents = []
    for spec in specs:
        agent = await manager.spawn(
            role=spec.role_description,
            name=spec.name,
            system_message=spec.system_prompt,
        )
        agents.append((spec, agent))

    print(f"Active agents: {[a.name for _, a in agents]}\n")

    # --- Run ---
    for spec, agent in agents:
        task = TASKS.get(spec.name, GOAL)
        result = await manager.run_task(agent.agent_id, task)
        print(f"[{spec.name}]\n{result}\n")

    # --- Kill ---
    await manager.shutdown()
    print("All agents shut down cleanly.")


def _cmd_run(_args: argparse.Namespace) -> None:
    asyncio.run(_run_demo())


def _cost(value: float | None) -> str:
    return "—" if value is None else str(value)  # None: no price for the model


def _format_traces_table(rows: list[dict]) -> str:
    if not rows:
        return "No traces found."
    headers = [
        "run_id", "timestamp", "tier", "app_id", "framework",
        "outcome", "latency_ms", "cost_usd", "tokens_in", "tokens_out",
    ]
    col_widths = [max(len(h), 8) for h in headers]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, col_widths))]
    for r in rows:
        values = [
            r["run_id"][:8], r["timestamp"][:19], r["routing_tier"], r.get("app_id") or "-",
            r.get("framework_used") or "-", r["outcome"],
            str(r["total_latency_ms"]), _cost(r["total_cost_usd"]),
            str(r["total_input_tokens"]), str(r["total_output_tokens"]),
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(values, col_widths)))
    return "\n".join(lines)


def _cmd_traces(args: argparse.Namespace) -> None:
    from agent_forge import db

    rows = db.query_runs(
        limit=args.last, app_id=args.app, status=args.status,
        since=args.since, cost_above=args.cost_above,
    )
    print(_format_traces_table(rows))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-forge", description="agent-forge CLI")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the hardcoded demo pipeline (autogen only).")
    run_p.set_defaults(func=_cmd_run)

    traces_p = sub.add_parser("traces", help="Query run traces from the local SQLite store.")
    traces_p.add_argument("--last", type=int, default=10, help="Max rows to show (default 10).")
    traces_p.add_argument("--app", dest="app", default=None, help="Filter by app_id.")
    traces_p.add_argument(
        "--status", default=None, choices=["success", "partial", "failed"],
        help="Filter by outcome.",
    )
    traces_p.add_argument("--since", default=None, help="Filter by ISO-8601 timestamp lower bound.")
    traces_p.add_argument(
        "--cost-above", dest="cost_above", type=float, default=None,
        help="Filter by total_cost_usd > value.",
    )
    traces_p.set_defaults(func=_cmd_traces)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    # No subcommand → preserve today's `python -m agent_forge` behaviour (runs the demo).
    func = getattr(args, "func", _cmd_run)
    func(args)


if __name__ == "__main__":
    main()
