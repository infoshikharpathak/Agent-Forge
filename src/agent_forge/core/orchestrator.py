from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict

from agent_forge.config.settings import ProviderConfig, Settings

if TYPE_CHECKING:
    from agent_forge.core.shared_thread import SharedThread


@dataclass
class OrchestratorToolCall:
    """Emitted during plan_stream when the orchestrator calls a tool."""
    tool: str
    args: dict
    result: str


class AgentSpec(BaseModel):
    """Specification for a single agent in an AutoGen debate team."""

    name: str
    role_description: str
    system_prompt: str
    tools: list[str] = []


class GraphNode(BaseModel):
    """A single node in a LangGraph execution graph."""

    name: str
    role_description: str
    system_prompt: str
    task_prompt: str
    tools: list[str] = []


class GraphEdge(BaseModel):
    """A directed edge between two nodes in a LangGraph execution graph."""

    model_config = ConfigDict(populate_by_name=True)
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")


class GraphSpec(BaseModel):
    """Full specification for a LangGraph execution graph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    entry: str


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

    _SYSTEM_BASE = """\
You are an agent orchestration planner. Given a goal, choose the best execution \
strategy and plan a team of AI agents to accomplish it.

STRATEGY CHOICE — default to autogen unless langgraph is clearly better:
- autogen: Use for the vast majority of goals — analysis, research, tool use, \
comparisons, opinions, or any problem where agents benefit from seeing each \
other's output and building on it. Agents debate and challenge each other until \
they converge on a good answer.
- langgraph: Use ONLY when tasks are truly independent and parallel — e.g. \
simultaneously analysing 5 separate companies where each agent works in isolation \
and does NOT need to see what the others found. Or when stages are pure \
transformations with no benefit from shared context (translate → reformat). \
Do NOT use langgraph just because a task sounds "sequential" — if agents would \
benefit from seeing each other's research or analysis, use autogen.

RETURN a JSON object — choose exactly one of these formats:

For autogen:
{{
  "strategy": "autogen",
  "agents": [
    {{
      "name": "snake_case_name",
      "role_description": "one sentence",
      "system_prompt": "detailed, specific system prompt",
      "tools": ["tool_name"]
    }}
  ]
}}

For langgraph:
{{
  "strategy": "langgraph",
  "nodes": [
    {{
      "name": "snake_case_name",
      "role_description": "one sentence",
      "system_prompt": "detailed persona, constraints, and domain knowledge for this node",
      "task_prompt": "the specific task this node must perform — tailored to its exact job, not the generic goal",
      "tools": ["tool_name"]
    }}
  ],
  "edges": [
    {{"from": "node_a", "to": "node_b"}}
  ],
  "entry": "first_node_name"
}}

Rules:
- For langgraph: if nodes are fully independent (e.g. each analyses a different company
  with no need to see each other's output), leave "edges" as an empty array []. Each
  node will receive only the goal — no shared context. If stages must build on each
  other (e.g. researcher feeds analyst), define edges explicitly.
- tools: choose ONLY from the available tools listed below. Empty list if none needed.
- Only create agents/nodes that are genuinely necessary.

Available tools:
{tools_list}"""

    def _build_system_prompt(self) -> str:
        from agent_forge.tools import list_tools
        tools_str = "\n".join(f"  - {t}" for t in list_tools())
        return self._SYSTEM_BASE.format(tools_list=tools_str)

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
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return [AgentSpec(**spec) for spec in data["agents"]]

    _RESEARCH_SYSTEM = (
        "You are a research assistant helping an AI orchestration planner stay current. "
        "Use the available tools to gather a small amount of targeted, relevant context "
        "for the goal — current date, recent news, or key facts. "
        "Make 1-3 focused tool calls, then stop."
    )

    # Tools the orchestrator is allowed to use
    _ORCHESTRATOR_TOOLS = ["web_search", "get_datetime"]

    async def _research(self, goal: str) -> tuple[list[OrchestratorToolCall], str]:
        """
        Run a tool-calling loop to gather current context for the goal.
        Returns (tool_calls_made, research_summary_text).
        """
        from agent_forge.tools import get_tool, get_openai_schemas

        tool_schemas = get_openai_schemas(self._ORCHESTRATOR_TOOLS)
        messages: list[dict] = [
            {"role": "system", "content": self._RESEARCH_SYSTEM},
            {"role": "user", "content": f"Gather current context for: {goal}"},
        ]
        tool_calls_made: list[OrchestratorToolCall] = []

        for _ in range(4):  # safety cap on rounds
            response = await self._client().chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                fn = get_tool(tc.function.name)
                args = json.loads(tc.function.arguments)
                import inspect as _inspect
                if _inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = fn(**args)
                tool_calls_made.append(
                    OrchestratorToolCall(tool=tc.function.name, args=args, result=result)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        research_text = "\n\n".join(
            f"[{tc.tool}({', '.join(f'{k}={v}' for k, v in tc.args.items())})]\n{tc.result}"
            for tc in tool_calls_made
        )
        return tool_calls_made, research_text

    async def plan_stream(self, goal: str):
        """
        Async generator for live UI display.
        Phase 1 — yields OrchestratorToolCall as the orchestrator researches.
        Phase 2 — yields str chunks as the plan JSON streams.
        Phase 3 — yields list[AgentSpec] as the final item.
        """
        # Phase 1: research
        tool_calls, research_text = await self._research(goal)
        for tc in tool_calls:
            yield tc

        # Phase 2 + 3: stream plan with research context injected
        user_content = f"Goal: {goal}"
        if research_text:
            user_content += f"\n\nCurrent context from research:\n{research_text}"

        stream = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            stream=True,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_content},
            ],
        )
        chunks: list[str] = []
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                chunks.append(text)
                yield text

        data = json.loads("".join(chunks))
        if data.get("strategy") == "langgraph":
            yield GraphSpec(
                nodes=[GraphNode(**n) for n in data["nodes"]],
                edges=[GraphEdge.model_validate(e) for e in data["edges"]],
                entry=data["entry"],
            )
        else:
            yield [AgentSpec(**spec) for spec in data.get("agents", [])]

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
