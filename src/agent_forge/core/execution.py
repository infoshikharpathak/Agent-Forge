from __future__ import annotations

"""
The single path every agent run goes through.

Both ``AgentManager.run_task`` (conversation path) and ``GraphRunner``
(graph path) call ``execute_agent``, so tracing and the RunContext behave
the same no matter which strategy the Orchestrator picked.
"""

import time
from typing import TYPE_CHECKING, Any

from agent_forge.core.agent import AgentRunResult
from agent_forge.core.context import RunContext, current_run_context, use_run_context

if TYPE_CHECKING:
    from agent_forge.core.agent import BaseAgent
    from agent_forge.tracing import RunTracer


def base_run_context(tracer: RunTracer | None) -> RunContext:
    """RunContext an agent run starts from: the tracer's IDs, else whatever
    context the caller already set, else a fresh one."""
    if tracer is not None:
        return tracer.run_context()
    return current_run_context() or RunContext.new()


async def execute_agent(
    agent: BaseAgent,
    task: str,
    label: str,
    tracer: RunTracer | None,
    **run_kwargs: Any,
) -> AgentRunResult:
    """Run ``agent.run(task)`` once with the RunContext set, recording
    latency, usage and status to *tracer* (if any) under *label*.

    Exceptions from the agent propagate after being recorded as "failed".
    """
    ctx = base_run_context(tracer).for_agent(agent.name, agent.role)
    start = time.perf_counter()
    with use_run_context(ctx):
        try:
            result = await agent.run(task, **run_kwargs)
        except Exception:
            if tracer is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                tracer.record_agent(label, AgentRunResult(content="", model=""), latency_ms, status="failed")
            raise
    if tracer is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        tracer.record_agent(label, result, latency_ms, status="success")
    return result
