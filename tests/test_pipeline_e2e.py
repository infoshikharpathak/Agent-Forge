"""End-to-end through the API pipeline (orchestrator + factories stubbed):
hooks on both strategies, a ToolSource tool used by an agent, trace_id on
the persisted row, and baselines with no hooks/sources."""

import ast
import contextlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agent_forge
from agent_forge import db
from agent_forge.api import app as api
from agent_forge.core.hooks import AgentHooks, HookDecision
from agent_forge.core.orchestrator import AgentSpec
from agent_forge.tools import get_tool

from .fakes import LONG, FakeFactory, pipeline

from .test_tool_sources import FakeSource


class KeywordBlock(AgentHooks):
    def __init__(self, keyword):
        self.keyword = keyword

    async def before_agent_run(self, ctx, task):
        if self.keyword in task.lower():
            return HookDecision.deny(f"Blocked: input mentions {self.keyword!r}.")
        return HookDecision.allow()


def fake_orchestrator(plan):
    class FakeOrchestrator:
        def __init__(self, provider):
            pass

        async def clarify_goal(self, goal):
            return {"clarified_goal": goal, "was_changed": False, "reasoning": ""}

        async def _research(self, goal):
            return [], ""

        async def _generate_plan_stream(self, goal, research, feedback=None):
            yield "plan text"
            yield plan

        async def validate_plan(self, goal, plan_json):
            return {"valid": True, "feedback": ""}

        async def convergence_check(self, **kw):
            return True, "converged"

        async def synthesize_stream(self, goal, context, feedback=None):
            yield f"REPORT based on: {context[:200]}"

        async def quality_check(self, goal, report):
            return {"passes": True, "feedback": ""}

        async def grounding_check(self, goal, text, report):
            return {"grounded": True, "unsupported_claims": []}

    return FakeOrchestrator


async def _agent_uses_tool(task):
    # what a real backend does when the model calls the tool
    return f"used tool -> {await get_tool('lookup_order')(order_id='A1')}{LONG}"


@pytest.fixture
def stub_pipeline(monkeypatch):
    factories = []

    responders = {"researcher": _agent_uses_tool, "a": _agent_uses_tool}

    def make_factory(provider="openai"):
        f = FakeFactory(dict(responders))
        factories.append(f)
        return f

    monkeypatch.setattr(api, "MCPBridge", contextlib.nullcontext)
    monkeypatch.setattr(api.app_registry, "list_apps", lambda: [])
    monkeypatch.setattr(api.app_registry, "load", lambda path: None)
    monkeypatch.setattr(api, "LangGraphFactory", make_factory)
    monkeypatch.setattr(api, "AutoGenFactory", make_factory)

    def use(plan, tool_agents=True):
        if not tool_agents:
            responders.clear()
        monkeypatch.setattr(api, "Orchestrator", fake_orchestrator(plan))
        return factories

    return use


GRAPH_PLAN = pipeline("a", "b")
DEBATE_PLAN = [
    AgentSpec(name="researcher", role_description="r", system_prompt="s"),
    AgentSpec(name="critic", role_description="c", system_prompt="s"),
]


@pytest.mark.parametrize("plan", [GRAPH_PLAN, DEBATE_PLAN], ids=["graph", "conversation"])
def test_hook_blocks_keyword_and_source_tool_sees_trace(stub_pipeline, plan):
    factories = stub_pipeline(plan)
    src = FakeSource()
    app = api.create_app(hooks=[KeywordBlock("forbidden")], tool_sources=[src])
    with TestClient(app) as client:
        ok = client.post("/run", json={"goal": "check order A1", "trace_id": "e2e-1"})
        blocked = client.post("/run", json={"goal": "do the forbidden thing", "trace_id": "e2e-2"})
    assert ok.status_code == blocked.status_code == 200

    # the first agent called the source tool, inside run e2e-1
    (name, args, ctx) = src.calls[0]
    first_agent = "a" if plan is GRAPH_PLAN else "researcher"
    assert (ctx.trace_id, ctx.agent_name) == ("e2e-1", first_agent)
    assert "trace=e2e-1" in ok.json()["result"]

    row_ok = db.query_runs(trace_id="e2e-1")[0]
    assert row_ok["outcome"] == "success" and row_ok["guardrails_triggered"] == []
    assert row_ok["run_id"] != "e2e-1"  # caller trace_id is separate from run_id

    row_blocked = db.query_runs(trace_id="e2e-2")[0]
    assert f"hook:before_agent_run:blocked:{first_agent}" in row_blocked["guardrails_triggered"]
    assert row_blocked["metadata"]["hook_denials"][0]["message"] == "Blocked: input mentions 'forbidden'."
    blocked_agent = factories[-1].created[first_agent]
    assert blocked_agent.tasks == []  # never ran
    assert len(src.calls) == 1  # the blocked run never reached the tool


@pytest.mark.parametrize("plan", [GRAPH_PLAN, DEBATE_PLAN], ids=["graph", "conversation"])
def test_no_hooks_no_sources_baseline(stub_pipeline, plan):
    factories = stub_pipeline(plan, tool_agents=False)
    with TestClient(api.create_app()) as client:
        resp = client.post("/run", json={"goal": "plain goal"})
    assert resp.status_code == 200
    names = [n.name for n in plan.nodes] if plan is GRAPH_PLAN else [s.name for s in plan]
    for name in names:
        assert len(factories[-1].created[name].tasks) == 1  # every agent ran once
    row = db.query_runs()[0]
    assert row["outcome"] == "success"
    assert row["guardrails_triggered"] == [] and "hook_denials" not in row["metadata"]
    # The stub agents' model has no price: unknown cost, flagged, never $0.
    assert row["total_cost_usd"] is None and row["metadata"]["unpriced_models"] == ["fake-model"]
    assert row["trace_id"] == row["run_id"]  # no caller trace_id: defaults to run_id
    assert [a["status"] for a in row["agents_spawned"]] == ["success"] * len(names)


def test_no_hooks_conversation_tasks_unchanged():
    """Without hooks, the second speaker receives the full first message."""
    import asyncio

    from agent_forge.core.conversation import AgentConversation
    from agent_forge.core.manager import AgentManager

    class Orch:
        async def convergence_check(self, **kw):
            return True, "done"

    async def run():
        manager = AgentManager(FakeFactory())
        agents = [await manager.spawn(role="r", name=n) for n in ("alice", "bob")]
        events = [e async for e in AgentConversation(manager, agents, Orch(), max_rounds=1).run_stream("g")]
        return agents, events

    agents, events = asyncio.run(run())
    assert agents[0].tasks[0].startswith("g")
    assert "[alice]: alice says hello" in agents[1].tasks[0]
    assert events[-1].stopped_by == "orchestrator"


def test_agent_forge_never_imports_anvil():
    src = Path(agent_forge.__file__).parent
    for path in src.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n == "anvil" or n.startswith("anvil.") for n in names), f"{path} imports {names}"
