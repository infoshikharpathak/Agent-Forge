from __future__ import annotations

"""LangGraph-backed agent for agent-forge."""

import inspect
from typing import Any

from agent_forge.core.agent import AgentStatus, BaseAgent


class LangGraphAgent(BaseAgent):
    """
    BaseAgent wrapper that calls an LLM via LangChain's ChatOpenAI.

    Tools (if any) are bound to the model at creation time via
    ``llm.bind_tools()``.  The ``run()`` method handles the full
    tool-calling loop internally, so callers just get back a plain string.

    Args:
        agent_id:       Unique identifier assigned by the factory.
        name:           Display name used in conversation logs.
        role:           Short role label (e.g. ``"analyst"``).
        llm:            A LangChain chat model, optionally with tools bound.
        system_message: System prompt written by the Orchestrator.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        llm: Any,
        system_message: str,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        self._llm = llm
        self._system_message = system_message

    async def run(self, task: str, **kwargs: Any) -> str:
        """Run the agent on *task*, handling any tool calls, and return the result."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from agent_forge.tools import get_tool

        messages: list[Any] = [
            SystemMessage(content=self._system_message),
            HumanMessage(content=task),
        ]

        self.status = AgentStatus.RUNNING

        response = await self._llm.ainvoke(messages)

        # Tool-calling loop
        while getattr(response, "tool_calls", None):
            messages.append(response)
            for tc in response.tool_calls:
                try:
                    fn = get_tool(tc["name"])
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(**tc["args"])
                    else:
                        result = fn(**tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
                messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content=str(result),
                ))
            response = await self._llm.ainvoke(messages)

        self.status = AgentStatus.IDLE
        return response.content

    async def close(self) -> None:
        """LangChain clients are stateless; just mark as closed."""
        self.status = AgentStatus.CLOSED
