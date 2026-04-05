from __future__ import annotations

import uuid
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agent_forge.backends.autogen.agent import AutoGenAgent
from agent_forge.config.settings import ProviderConfig, Settings
from agent_forge.core.factory import AgentFactory


class AutoGenFactory(AgentFactory):
    """
    Creates AutoGen AssistantAgents backed by an OpenAI-compatible model client.

    Agents are fully defined by the system_message passed at creation time —
    typically written by an Orchestrator for the specific task at hand.

    Args:
        provider: LLM provider name (only 'openai' is wired today).
        model:    Override the default model for this provider.
    """

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        super().__init__()
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)

    def _build_model_client(self) -> OpenAIChatCompletionClient:
        if self._config.provider != "openai":
            raise NotImplementedError(
                f"AutoGenFactory only supports 'openai' today, got {self._config.provider!r}."
            )
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "api_key": self._config.api_key,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        return OpenAIChatCompletionClient(**kwargs)

    async def create(
        self,
        role: str = "agent",
        name: str = "agent",
        *,
        system_message: str,
        tools: list[str] | None = None,
        **kwargs: Any,
    ) -> AutoGenAgent:
        from agent_forge.tools import get_tools

        agent_id = str(uuid.uuid4())
        resolved_tools = get_tools(tools) if tools else []

        native = AssistantAgent(
            name=name,
            model_client=self._build_model_client(),
            system_message=system_message,
            tools=resolved_tools,
        )

        agent = AutoGenAgent(agent_id=agent_id, name=name, role=role, native=native)
        self._agents[agent_id] = agent
        return agent
