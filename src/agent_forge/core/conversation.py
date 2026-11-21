from __future__ import annotations

import logging
import json as _json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, Any

log = logging.getLogger("uvicorn.error")

_MIN_OUTPUT_LENGTH = 20  # lower threshold for debate agents vs graph nodes

from agent_forge.core.execution import base_run_context
from agent_forge.core.hooks import run_hooks


def _is_valid_output(content: str) -> bool:
    return len((content or "").strip()) >= _MIN_OUTPUT_LENGTH

if TYPE_CHECKING:
    from agent_forge.core.agent import BaseAgent
    from agent_forge.core.manager import AgentManager
    from agent_forge.core.orchestrator import Orchestrator


# ── Position block helpers ────────────────────────────────────────────────────

_POSITION_TAG = re.compile(r"\[POSITION\]\s*(.*?)\s*\[/POSITION\]", re.DOTALL)

_POSITION_INSTRUCTION = (
    "\n\nAt the end of your response, append a position block in this exact format "
    "(do not skip this — it is required):\n"
    "[POSITION]\n"
    "{\"position\": \"<your stance in 2-5 words>\", "
    "\"confidence\": <0.0-1.0>, "
    "\"reasoning_summary\": \"<one sentence summarising your core argument>\"}\n"
    "[/POSITION]"
)


def _extract_position(content: str) -> dict | None:
    """Parse the [POSITION]...[/POSITION] block from an agent response. Returns None if absent or malformed."""
    m = _POSITION_TAG.search(content)
    if not m:
        return None
    try:
        return _json.loads(m.group(1))
    except Exception:
        return None


def _strip_position(content: str) -> str:
    """Remove the [POSITION]...[/POSITION] block before display or context passing."""
    return _POSITION_TAG.sub("", content).strip()


@dataclass
class ConversationMessage:
    agent: str
    content: str
    round: int


@dataclass
class StopSignal:
    reason: str
    stopped_by: str  # "orchestrator" | "max_rounds" | "graph" | "hook"


class AgentConversation:
    """
    Runs a multi-round conversation between agents.

    Each agent sees the full conversation history before responding, so
    it can genuinely react to what others said. After each complete round,
    the orchestrator judges whether agents have converged. The conversation
    stops when the orchestrator says so, or when max_rounds is reached —
    whichever comes first.

    Usage:
        conv = AgentConversation(manager, agents, orchestrator, max_rounds=5)
        async for item in conv.run_stream(goal):
            if isinstance(item, ConversationMessage):
                print(f"[{item.agent}] {item.content}")
            elif isinstance(item, StopSignal):
                print(f"Stopped: {item.reason}")
    """

    def __init__(
        self,
        manager: AgentManager,
        agents: list[BaseAgent],
        orchestrator: Orchestrator,
        max_rounds: int = 5,
    ) -> None:
        self._manager = manager
        self._agents = agents
        self._orchestrator = orchestrator
        self._max_rounds = max_rounds
        self._history: list[ConversationMessage] = []

    def history(self) -> list[ConversationMessage]:
        return list(self._history)

    def to_context_text(self, last_override: str | None = None) -> str:
        """Format the full conversation history as a plain-text block.

        *last_override* replaces the latest message's content (a handoff
        hook's replacement payload) without changing the stored history."""
        if not self._history:
            return ""
        lines = ["=== Conversation so far ==="]
        current_round = 0
        for i, msg in enumerate(self._history):
            if msg.round != current_round:
                current_round = msg.round
                lines.append(f"\n-- Round {current_round} --")
            content = msg.content
            if last_override is not None and i == len(self._history) - 1:
                content = last_override
            lines.append(f"[{msg.agent}]: {content}")
        lines.append("=== End of conversation ===")
        return "\n\n".join(lines)

    def _build_task(self, goal: str, last_override: str | None = None) -> str:
        """Build the full task string for the next agent turn."""
        if not self._history:
            return goal + _POSITION_INSTRUCTION
        return (
            f"Goal: {goal}\n\n"
            f"{self.to_context_text(last_override)}\n\n"
            "Based on the conversation above, continue your contribution. "
            "Build on prior points, challenge weak reasoning, or add new insights. "
            "Be direct and specific."
            + _POSITION_INSTRUCTION
        )

    async def run_stream(
        self, goal: str
    ) -> AsyncGenerator[ConversationMessage | StopSignal, None]:
        """
        Async generator. Yields ConversationMessage as each agent responds,
        then StopSignal when the conversation ends.

        Position blocks ([POSITION]...[/POSITION]) are extracted from each
        agent response and stripped before the message is yielded or stored
        in history — so they never appear in the UI or synthesis context.
        Extracted positions are passed to the hybrid convergence check.
        """
        prev_positions: list[dict] = []

        for round_num in range(1, self._max_rounds + 1):
            current_positions: list[dict] = []

            for agent in self._agents:
                # Passing the conversation to the next agent is a handoff from
                # the previous speaker. A deny ends the conversation (it is a
                # single chain); an allow with content replaces what is handed over.
                override: str | None = None
                if self._manager.hooks and self._history and self._history[-1].agent != agent.name:
                    prev = self._history[-1]
                    ctx = base_run_context(self._manager.tracer).for_agent(prev.agent, None)
                    outcome = await run_hooks(
                        self._manager.hooks, "on_handoff", ctx, prev.content,
                        subject=f"{prev.agent}->{agent.name}", tracer=self._manager.tracer,
                        handoff=(prev.agent, agent.name),
                    )
                    if not outcome.allowed:
                        yield StopSignal(reason=outcome.value, stopped_by="hook")
                        return
                    if outcome.value != prev.content:
                        override = outcome.value

                task = self._build_task(goal, override)
                run = await self._manager.run_task_result(agent.agent_id, task)
                result = run.content
                # Hook-blocked/withheld results carry the hook's message; never retry them.
                if run.status == "success" and not _is_valid_output(result):
                    log.warning("[agent] %s invalid output (len=%d) — retrying", agent.name, len(result or ""))
                    run = await self._manager.run_task_result(agent.agent_id, task)
                    result = run.content
                    if run.status == "success" and not _is_valid_output(result):
                        log.error("[agent] %s failed after retry — using fallback", agent.name)
                        result = f"[{agent.name}: failed to produce a valid response]"

                # Extract structured position, strip tag from content
                position = _extract_position(result)
                clean_content = _strip_position(result)

                if position:
                    position["agent"] = agent.name
                    current_positions.append(position)

                msg = ConversationMessage(
                    agent=agent.name, content=clean_content, round=round_num
                )
                self._history.append(msg)
                yield msg

            # Hybrid convergence check
            converged, reason = await self._orchestrator.convergence_check(
                goal=goal,
                round_num=round_num,
                current_positions=current_positions,
                previous_positions=prev_positions,
                history_text=self.to_context_text(),
            )
            if converged:
                yield StopSignal(reason=reason, stopped_by="orchestrator")
                return

            prev_positions = current_positions

        yield StopSignal(
            reason=f"Reached the maximum of {self._max_rounds} rounds.",
            stopped_by="max_rounds",
        )
