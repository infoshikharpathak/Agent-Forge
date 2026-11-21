from __future__ import annotations

"""
Run context for agent-forge.

A ``RunContext`` describes the agent run currently executing: which trace
it belongs to, which pipeline run, and which agent. It lives in a
``ContextVar``, so it follows asyncio tasks. Two concurrent runs never see
each other's context, and anything called from inside ``agent.run()``
(tools, tool sources, hooks) can read it without it being threaded through
every backend's signature.

``execute_agent`` (core/execution.py) sets it for the duration of each
agent run. Code outside a run sees ``None``.
"""

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True)
class RunContext:
    trace_id: str
    """Caller-supplied correlation ID (e.g. from an upstream request). Defaults to run_id."""
    run_id: str
    """This pipeline run's ID (the persisted trace row's run_id)."""
    agent_name: str | None = None
    agent_role: str | None = None
    app: str | None = None
    """Reserved for app namespacing; always None for now."""

    def for_agent(self, name: str, role: str | None) -> RunContext:
        return replace(self, agent_name=name, agent_role=role)

    @classmethod
    def new(cls, trace_id: str | None = None) -> RunContext:
        run_id = str(uuid.uuid4())
        return cls(trace_id=trace_id or run_id, run_id=run_id)


_current: ContextVar[RunContext | None] = ContextVar("agent_forge_run_context", default=None)


def current_run_context() -> RunContext | None:
    """The RunContext of the agent run executing in this task, if any."""
    return _current.get()


@contextmanager
def use_run_context(ctx: RunContext) -> Iterator[RunContext]:
    """Set *ctx* as the current RunContext for the duration of the block."""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
