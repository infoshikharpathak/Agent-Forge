from __future__ import annotations

"""
Run tracing for agent-forge.

One RunTracer is created per pipeline execution (see api/app.py). It collects
per-agent usage/latency as agents run, plus which guardrails actually fired,
then finish() persists a single append-only row to SQLite via db.insert_run().

Scope note: v1 only tokenizes spawned-agent calls (agents_spawned). The
Orchestrator's own planning/guardrail LLM calls are not yet captured, so
total_cost_usd / total_tokens understate true spend — a known gap, not
silently rounded into the totals.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_forge import db
from agent_forge.core.agent import AgentRunResult
from agent_forge.core.context import RunContext

# Approximate $ per 1K tokens as (input_rate, output_rate). Unknown models cost
# $0 — this table is illustrative/editable, not a source of billing truth.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.00025, 0.001),
    "gpt-5.4-nano": (0.00005, 0.0002),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-6": (0.015, 0.075),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    "gemini-2.5-flash": (0.000075, 0.0003),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
    return (input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate


@dataclass
class RunTracer:
    task: str
    routing_tier: str
    app_id: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    """Caller-supplied correlation ID. Defaults to run_id when not given."""
    agents_spawned: list[dict] = field(default_factory=list, init=False)
    guardrails_triggered: list[str] = field(default_factory=list, init=False)
    hook_denials: list[dict] = field(default_factory=list, init=False)
    _started_at: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = self.run_id

    def run_context(self) -> RunContext:
        """Base RunContext for agents executed under this tracer."""
        return RunContext(trace_id=self.trace_id or self.run_id, run_id=self.run_id)

    def record_agent(
        self, agent_id: str, result: AgentRunResult, latency_ms: float, status: str
    ) -> None:
        self.agents_spawned.append({
            "agent_id": agent_id,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": round(latency_ms, 1),
            "status": status,
            "content": result.content,
        })

    def record_guardrail(self, name: str) -> None:
        self.guardrails_triggered.append(name)

    def record_hook_denial(self, label: str, message: str, *, hook: str) -> None:
        """A hook denied something: record it as a guardrail, and keep the
        message (persisted under metadata["hook_denials"])."""
        self.record_guardrail(label)
        self.hook_denials.append({"guardrail": label, "hook": hook, "message": message})

    async def finish(
        self, outcome: str, *, error: str | None = None, framework_used: str | None = None,
        report: str = "",
    ) -> None:
        total_latency_ms = (time.perf_counter() - self._started_at) * 1000
        total_input = sum(a["input_tokens"] for a in self.agents_spawned)
        total_output = sum(a["output_tokens"] for a in self.agents_spawned)
        total_cost = sum(
            _estimate_cost(a["model"], a["input_tokens"], a["output_tokens"])
            for a in self.agents_spawned
        )
        row = {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "timestamp": self._timestamp,
            "task": self.task,
            "routing_tier": self.routing_tier,
            "app_id": self.app_id,
            "framework_used": framework_used,
            "agents_spawned": self.agents_spawned,
            "guardrails_triggered": self.guardrails_triggered,
            "total_latency_ms": round(total_latency_ms, 1),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "outcome": outcome,
            "error": error,
            "report": report,
            "metadata": {"hook_denials": self.hook_denials} if self.hook_denials else {},
        }
        await asyncio.to_thread(db.insert_run, row)
