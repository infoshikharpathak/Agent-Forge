from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CLOSED = "closed"


@dataclass
class AgentRunResult:
    """Result of a single BaseAgent.run() call, including token usage for tracing."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "success"
    """"success", or "blocked" / "withheld" when a hook stopped the run or its output."""


class BaseAgent(ABC):
    """
    Abstract base for all agents regardless of backend.

    Every backend (AutoGen, LangGraph, Anthropic, ...) wraps its
    native agent object inside a concrete subclass of this interface.
    """

    def __init__(self, agent_id: str, name: str, role: str) -> None:
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> AgentRunResult:
        """Execute a task and return the result, including token usage."""

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this agent."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self.status != AgentStatus.CLOSED

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id!r} role={self.role!r} status={self.status}>"
