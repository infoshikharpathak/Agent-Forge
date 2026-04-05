from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, Any

if TYPE_CHECKING:
    from agent_forge.core.agent import BaseAgent
    from agent_forge.core.manager import AgentManager
    from agent_forge.core.orchestrator import Orchestrator


@dataclass
class ConversationMessage:
    agent: str
    content: str
    round: int


@dataclass
class StopSignal:
    reason: str
    stopped_by: str  # "orchestrator" | "max_rounds"


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

    def to_context_text(self) -> str:
        """Format the full conversation history as a plain-text block."""
        if not self._history:
            return ""
        lines = ["=== Conversation so far ==="]
        current_round = 0
        for msg in self._history:
            if msg.round != current_round:
                current_round = msg.round
                lines.append(f"\n-- Round {current_round} --")
            lines.append(f"[{msg.agent}]: {msg.content}")
        lines.append("=== End of conversation ===")
        return "\n\n".join(lines)

    def _build_task(self, goal: str) -> str:
        """Build the full task string for the next agent turn."""
        if not self._history:
            return goal
        return (
            f"Goal: {goal}\n\n"
            f"{self.to_context_text()}\n\n"
            "Based on the conversation above, continue your contribution. "
            "Build on prior points, challenge weak reasoning, or add new insights. "
            "Be direct and specific."
        )

    async def run_stream(
        self, goal: str
    ) -> AsyncGenerator[ConversationMessage | StopSignal, None]:
        """
        Async generator. Yields ConversationMessage as each agent responds,
        then StopSignal when the conversation ends.
        """
        for round_num in range(1, self._max_rounds + 1):
            for agent in self._agents:
                task = self._build_task(goal)
                result = await self._manager.run_task(agent.agent_id, task)
                msg = ConversationMessage(
                    agent=agent.name, content=result, round=round_num
                )
                self._history.append(msg)
                yield msg

            # After each full round, ask orchestrator if agents have converged
            converged, reason = await self._orchestrator.should_stop(
                goal, self.to_context_text()
            )
            if converged:
                yield StopSignal(reason=reason, stopped_by="orchestrator")
                return

        yield StopSignal(
            reason=f"Reached the maximum of {self._max_rounds} rounds.",
            stopped_by="max_rounds",
        )
