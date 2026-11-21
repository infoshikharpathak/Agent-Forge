from __future__ import annotations

import uuid
from typing import Any

from openai import AsyncOpenAI

from agent_forge.backends.openai.agent import OpenAIAgent
from agent_forge.config.settings import Settings
from agent_forge.core.factory import AgentFactory


class OpenAIFactory(AgentFactory):
    """Creates agents backed by the OpenAI Chat Completions API."""

    def __init__(self, *, model: str | None = None) -> None:
        super().__init__()
        self._config = Settings.for_provider("openai", model=model)
        kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        self._client = AsyncOpenAI(**kwargs)

    async def create(
        self,
        role: str = "general_assistant",
        name: str = "agent",
        *,
        system_message: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> OpenAIAgent:
        resolved_model = model or self._config.model

        openai_tools: list[dict] = []
        tool_handlers: dict[str, Any] = {}

        if tools:
            from agent_forge.tools import get_tools, to_openai_schema
            resolved = get_tools(tools)
            for fn in resolved:
                tool_name = getattr(fn, "_tool_name", fn.__name__)
                schema = to_openai_schema(fn)
                openai_tools.append(schema)
                tool_handlers[tool_name] = fn

        agent_id = str(uuid.uuid4())
        agent = OpenAIAgent(
            agent_id=agent_id,
            name=name,
            role=role,
            client=self._client,
            system_message=system_message or f"You are {name}, a {role}.",
            model=resolved_model,
            tools=openai_tools,
            tool_handlers=tool_handlers,
        )
        self._agents[agent_id] = agent
        return agent
