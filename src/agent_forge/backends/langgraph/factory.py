from __future__ import annotations

"""LangGraph agent factory for agent-forge."""

import inspect
import uuid
from typing import Any

from agent_forge.backends.langgraph.agent import LangGraphAgent
from agent_forge.config.settings import ProviderConfig, Settings
from agent_forge.core.factory import AgentFactory


class LangGraphFactory(AgentFactory):
    """
    Creates LangGraphAgents backed by a LangChain ChatOpenAI model.

    Tools are resolved from the agent-forge tool registry and bound to
    the model via ``llm.bind_tools()`` so the agent can call them during
    its internal tool-calling loop.

    Args:
        provider: LLM provider name (only ``"openai"`` is wired today).
        model:    Override the default model for this provider.
    """

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        super().__init__()
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)

    async def create(
        self,
        role: str = "agent",
        name: str = "agent",
        *,
        system_message: str,
        tools: list[str] | None = None,
        **kwargs: Any,
    ) -> LangGraphAgent:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import StructuredTool
        from agent_forge.tools import get_tools

        kwargs_llm: dict[str, Any] = {
            "model": self._config.model,
            "api_key": self._config.api_key,
        }
        if self._config.base_url:
            kwargs_llm["base_url"] = self._config.base_url

        llm: Any = ChatOpenAI(**kwargs_llm)

        if tools:
            resolved = get_tools(tools)
            lc_tools = [
                StructuredTool.from_function(
                    coroutine=fn if inspect.iscoroutinefunction(fn) else None,
                    func=fn if not inspect.iscoroutinefunction(fn) else None,
                    name=getattr(fn, "_tool_name", fn.__name__),
                    description=getattr(fn, "_tool_description", "") or fn.__doc__ or "",
                )
                for fn in resolved
            ]
            llm = llm.bind_tools(lc_tools)

        agent_id = str(uuid.uuid4())
        agent = LangGraphAgent(
            agent_id=agent_id,
            name=name,
            role=role,
            llm=llm,
            system_message=system_message,
        )
        self._agents[agent_id] = agent
        return agent
