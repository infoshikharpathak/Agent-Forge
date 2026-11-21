"""Scripted agents/factories for tests: no LLM calls."""

from __future__ import annotations

from typing import Any, Callable

from agent_forge.core.agent import AgentRunResult, AgentStatus, BaseAgent
from agent_forge.core.context import current_run_context
from agent_forge.core.factory import AgentFactory
from agent_forge.core.orchestrator import GraphEdge, GraphNode, GraphSpec

LONG = " — this sentence pads the output past the minimum valid-output length."


class FakeAgent(BaseAgent):
    """Replies via *respond(task) -> str* (sync or async) and records each
    task it received plus the RunContext visible during run()."""

    def __init__(self, name: str, role: str = "tester", respond: Callable | None = None) -> None:
        super().__init__(agent_id=f"id-{name}", name=name, role=role)
        self._respond = respond or (lambda task: f"{name} says hello{LONG}")
        self.tasks: list[str] = []
        self.contexts: list[Any] = []

    async def run(self, task: str, **kwargs: Any) -> AgentRunResult:
        self.tasks.append(task)
        self.contexts.append(current_run_context())
        out = self._respond(task)
        if hasattr(out, "__await__"):
            out = await out
        return AgentRunResult(content=out, model="fake-model", input_tokens=3, output_tokens=5)

    async def close(self) -> None:
        self.status = AgentStatus.CLOSED


class FakeFactory(AgentFactory):
    """Creates FakeAgents; *responders* maps agent name -> respond callable."""

    def __init__(self, responders: dict[str, Callable] | None = None) -> None:
        super().__init__()
        self._responders = responders or {}
        self.created: dict[str, FakeAgent] = {}

    async def create(self, role: str = "tester", name: str = "agent", **kwargs: Any) -> FakeAgent:
        agent = FakeAgent(name, role, self._responders.get(name))
        self._agents[agent.agent_id] = agent
        self.created[name] = agent
        return agent


def node(name: str) -> GraphNode:
    return GraphNode(name=name, role_description=f"{name}-role", system_prompt="sys", task_prompt=f"do {name}")


def pipeline(*names: str) -> GraphSpec:
    edges = [GraphEdge(from_node=a, to_node=b) for a, b in zip(names, names[1:])]
    return GraphSpec(nodes=[node(n) for n in names], edges=edges, entry=names[0])


def parallel(*names: str) -> GraphSpec:
    return GraphSpec(nodes=[node(n) for n in names], edges=[], entry=names[0])


async def collect(stream) -> list:
    return [item async for item in stream]
