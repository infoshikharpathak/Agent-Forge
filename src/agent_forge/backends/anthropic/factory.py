from __future__ import annotations

import uuid
from typing import Any

from anthropic import AsyncAnthropic

from agent_forge.backends.anthropic.agent import AnthropicAgent
from agent_forge.config.settings import Settings
from agent_forge.core.factory import AgentFactory


class AnthropicFactory(AgentFactory):
    """Creates agents backed by the Anthropic Messages API."""

    def __init__(self, *, model: str | None = None) -> None:
        super().__init__()
        self._config = Settings.for_provider("anthropic", model=model)
        self._client = AsyncAnthropic(api_key=self._config.api_key)

    async def create(
        self,
        role: str = "general_assistant",
        name: str = "agent",
        *,
        system_message: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AnthropicAgent:
        resolved_model = model or self._config.model

        anthropic_tools: list[dict] = []
        tool_handlers: dict[str, Any] = {}

        if tools:
            from agent_forge.tools import get_tools, to_openai_schema
            resolved = get_tools(tools)
            for fn in resolved:
                tool_name = getattr(fn, "_tool_name", fn.__name__)
                oai_schema = to_openai_schema(fn)
                anthropic_tools.append({
                    "name": tool_name,
                    "description": oai_schema["function"]["description"],
                    "input_schema": oai_schema["function"]["parameters"],
                })
                tool_handlers[tool_name] = fn

        agent_id = str(uuid.uuid4())
        agent = AnthropicAgent(
            agent_id=agent_id,
            name=name,
            role=role,
            client=self._client,
            system_message=system_message or f"You are {name}, a {role}.",
            model=resolved_model,
            tools=anthropic_tools,
            tool_handlers=tool_handlers,
        )
        self._agents[agent_id] = agent
        return agent
