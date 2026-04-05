from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from agent_forge.config.settings import ProviderConfig, Settings

if TYPE_CHECKING:
    from agent_forge.core.shared_thread import SharedThread


class AgentSpec(BaseModel):
    """Specification for a single agent, written by the Orchestrator."""

    name: str
    role_description: str
    system_prompt: str
    tools: list[str] = []


class Orchestrator:
    """
    Uses an LLM to plan what agents are needed for a given goal and
    writes detailed system prompts for each one.

    Usage:
        orchestrator = Orchestrator(provider="openai")
        specs = await orchestrator.plan("Analyse AAPL earnings and write a short report.")

        for spec in specs:
            agent = await manager.spawn(
                name=spec.name,
                role=spec.role_description,
                system_message=spec.system_prompt,
            )
    """

    _SYSTEM = (
        "You are an agent orchestration planner. Given a high-level goal, decide what "
        "specialized AI agents are needed to accomplish it and write detailed system "
        "prompts for each one.\n\n"
        "Return a JSON object with an 'agents' array. Each element must have:\n"
        "  - name: short snake_case identifier (e.g. 'data_analyst')\n"
        "  - role_description: one sentence describing this agent's purpose\n"
        "  - system_prompt: a detailed, specific system prompt tailored exactly to the "
        "task — not generic. Include persona, constraints, output format expectations, "
        "and any domain knowledge the agent needs.\n"
        "  - tools: list of tool names this agent needs (empty list if none)\n\n"
        "Only create agents that are genuinely necessary. Prefer fewer, more capable "
        "agents over many narrow ones."
    )

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)

    def _client(self) -> AsyncOpenAI:
        """Create a fresh client per call so it binds to the current event loop."""
        kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        return AsyncOpenAI(**kwargs)

    async def plan(self, goal: str) -> list[AgentSpec]:
        """
        Given a goal, return a list of AgentSpecs needed to accomplish it.
        The LLM decides how many agents are needed and writes their system prompts.
        """
        response = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return [AgentSpec(**spec) for spec in data["agents"]]

    async def plan_stream(self, goal: str):
        """
        Async generator for live UI display.
        Yields str chunks while the LLM is writing the plan, then yields
        list[AgentSpec] as the final item once the response is complete.
        """
        stream = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            stream=True,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
        )
        chunks: list[str] = []
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                chunks.append(text)
                yield text

        data = json.loads("".join(chunks))
        yield [AgentSpec(**spec) for spec in data["agents"]]

    _SYNTHESIZE_SYSTEM = (
        "You are a senior analyst synthesizing the findings from multiple AI agents. "
        "Given the original goal and the full agent conversation, produce a single, clean, "
        "well-structured final response for the user. Eliminate redundancy, resolve any "
        "contradictions, and present the most important insights clearly and concisely."
    )

    _STOP_SYSTEM = (
        "You are a conversation moderator. Given the goal and the conversation history "
        "between AI agents, decide if the agents have converged enough to stop.\n\n"
        "Return JSON: {\"converged\": true/false, \"reason\": \"brief explanation\"}\n\n"
        "Stop if: agents have reached agreement or complementary conclusions, the goal "
        "is adequately addressed, or further rounds would add little value.\n"
        "Continue if: there are unresolved disagreements worth exploring, important "
        "aspects of the goal are uncovered, or agents have raised new questions."
    )

    async def synthesize_stream(self, goal: str, context: SharedThread | str):
        """
        Async generator. Reads the full context (SharedThread or pre-formatted
        conversation text) and streams a clean final synthesis back to the user.
        """
        context_text = context if isinstance(context, str) else context.to_context()
        stream = await self._client().chat.completions.create(
            model=self._config.model,
            stream=True,
            messages=[
                {"role": "system", "content": self._SYNTHESIZE_SYSTEM},
                {"role": "user", "content": f"Goal: {goal}\n\n{context_text}"},
            ],
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    async def should_stop(self, goal: str, history_text: str) -> tuple[bool, str]:
        """
        After each conversation round, judge whether agents have converged.
        Returns (converged: bool, reason: str).
        """
        response = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._STOP_SYSTEM},
                {"role": "user", "content": f"Goal: {goal}\n\n{history_text}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return data["converged"], data["reason"]
