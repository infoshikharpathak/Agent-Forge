from __future__ import annotations

"""
Agent hooks: policy checks around every agent run and handoff.

Subclass ``AgentHooks`` and override any of the three async methods; the
defaults allow everything unchanged. Pass hooks as an ordered list to
``AgentManager``, ``GraphRunner`` or ``create_app`` (there is no global
registry). Hooks run in list order: each sees the previous hook's
replacement, and the first deny wins.

    class NoSecrets(AgentHooks):
        async def before_agent_run(self, ctx, task):
            if "password" in task.lower():
                return HookDecision.deny("Task mentions a password.")
            return HookDecision.allow()

Semantics (see core/execution.py, graph_runner.py, conversation.py):

- before_agent_run deny  -> agent.run() is not called; the result's content
  is the hook's message and its status is "blocked". Allow with content
  replaces the task.
- after_agent_run deny   -> the output is withheld and replaced by the
  message; status "withheld". Allow with content replaces the output.
- on_handoff deny        -> that branch stops: the graph routes to END, a
  conversation ends. Allow with content replaces the payload handed over.
- A hook that raises is a deny (fail closed), never an exception out of
  the pipeline.
- Every deny is recorded on the tracer as
  "hook:<event>:<blocked|withheld|denied>:<agent or from->to>".
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from agent_forge.core.context import RunContext
    from agent_forge.tracing import RunTracer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookDecision:
    allowed: bool = True
    content: str | None = None
    """Replacement for the task / output / payload (allow only)."""
    message: str | None = None
    """Explanation shown in place of the blocked content (deny only)."""

    @classmethod
    def allow(cls, content: str | None = None) -> HookDecision:
        return cls(allowed=True, content=content)

    @classmethod
    def deny(cls, message: str) -> HookDecision:
        return cls(allowed=False, message=message)


class AgentHooks:
    """Base class with pass-through defaults. Override what you need."""

    async def before_agent_run(self, ctx: RunContext, task: str) -> HookDecision:
        return HookDecision.allow()

    async def after_agent_run(self, ctx: RunContext, output: str) -> HookDecision:
        return HookDecision.allow()

    async def on_handoff(
        self, ctx: RunContext, from_agent: str, to_agent: str, payload: str
    ) -> HookDecision:
        return HookDecision.allow()


Hooks = Sequence[AgentHooks]

# Outcome word recorded for a deny of each event.
_OUTCOME = {
    "before_agent_run": "blocked",
    "after_agent_run": "withheld",
    "on_handoff": "denied",
}


@dataclass(frozen=True)
class HookOutcome:
    allowed: bool
    value: str
    """The (possibly replaced) task/output/payload if allowed, else the deny message."""


async def run_hooks(
    hooks: Hooks | None,
    event: str,
    ctx: RunContext,
    value: str,
    *,
    subject: str,
    tracer: RunTracer | None,
    handoff: tuple[str, str] | None = None,
) -> HookOutcome:
    """Run *event* through every hook in order. Never raises."""
    for hook in hooks or ():
        name = type(hook).__name__
        try:
            method = getattr(hook, event)
            if handoff is not None:
                decision = await method(ctx, handoff[0], handoff[1], value)
            else:
                decision = await method(ctx, value)
            if not isinstance(decision, HookDecision):
                raise TypeError(f"returned {type(decision).__name__}, not HookDecision")
        except Exception as exc:  # fail closed
            log.exception("[hook] %s.%s raised", name, event)
            decision = HookDecision.deny(f"Hook {name}.{event} failed: {exc}")

        if not decision.allowed:
            message = decision.message or f"Blocked by hook {name}.{event}."
            record_denial(tracer, event, subject, message, hook=name)
            return HookOutcome(allowed=False, value=message)
        if decision.content is not None:
            value = decision.content
    return HookOutcome(allowed=True, value=value)


def record_denial(
    tracer: RunTracer | None, event: str, subject: str, message: str, *, hook: str
) -> None:
    label = f"hook:{event}:{_OUTCOME[event]}:{subject}"
    log.warning("[%s] %s", label, message)
    if tracer is not None:
        tracer.record_hook_denial(label, message, hook=hook)

