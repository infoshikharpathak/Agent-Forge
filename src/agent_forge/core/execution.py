from __future__ import annotations

"""
The single path every agent run goes through.

Both ``AgentManager.run_task`` (conversation path) and ``GraphRunner``
(graph path) call ``execute_agent``, so tracing and the RunContext behave
the same no matter which strategy the Orchestrator picked.
"""

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_forge.core.agent import AgentRunResult
from agent_forge.core.context import RunContext, current_run_context, use_run_context
from agent_forge.core.hooks import Hooks, run_hooks

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
    hooks: Hooks | None = None,
    **run_kwargs: Any,
) -> AgentRunResult:
    """Run ``agent.run(task)`` once with the RunContext set, recording
    latency, usage and status to *tracer* (if any) under *label*.

    Hooks (in order) may block the run, replace the task, withhold the
    output or replace it. A blocked run never calls ``agent.run()`` and
    returns the hook's message with status "blocked"; a withheld output is
    replaced by the message with status "withheld". Exceptions from the
    agent itself propagate after being recorded as "failed".
    """
    ctx = base_run_context(tracer).for_agent(agent.name, agent.role)
    with use_run_context(ctx):
        before = await run_hooks(hooks, "before_agent_run", ctx, task, subject=agent.name, tracer=tracer)
        if not before.allowed:
            result = AgentRunResult(content=before.value, model="", status="blocked")
            if tracer is not None:
                tracer.record_agent(label, result, 0.0, status="blocked")
            return result
        task = before.value

        start = time.perf_counter()
        try:
            result = await agent.run(task, **run_kwargs)
        except Exception:
            if tracer is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                tracer.record_agent(label, AgentRunResult(content="", model=""), latency_ms, status="failed")
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        after = await run_hooks(
            hooks, "after_agent_run", ctx, result.content, subject=agent.name, tracer=tracer
        )
        result = replace(
            result,
            content=after.value,
            status="success" if after.allowed else "withheld",
        )

    if tracer is not None:
        tracer.record_agent(label, result, latency_ms, status=result.status)
    return result
