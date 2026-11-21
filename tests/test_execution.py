"""Both execution paths go through execute_agent, which sets the RunContext."""

import pytest

from agent_forge.core import execution
from agent_forge.core.context import RunContext, current_run_context, use_run_context
from agent_forge.core.conversation import ConversationMessage, StopSignal
from agent_forge.core.graph_runner import GraphRunner
from agent_forge.core.manager import AgentManager
from agent_forge.tracing import RunTracer

from .fakes import LONG, FakeAgent, FakeFactory, collect, parallel, pipeline


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = execution.execute_agent

    async def wrapper(agent, task, label, tracer, **kw):
        calls.append(label)
        return await real(agent, task, label, tracer, **kw)

    # patch where each caller looks it up
    import agent_forge.core.graph_runner as gr
    import agent_forge.core.manager as mgr

    monkeypatch.setattr(mgr, "execute_agent", wrapper)
    monkeypatch.setattr(gr, "execute_agent", wrapper)
    return calls


async def test_execute_agent_sets_context_and_traces():
    tracer = RunTracer(task="t", routing_tier="dynamic", trace_id="T1")
    agent = FakeAgent("alice", role="analyst")
    result = await execution.execute_agent(agent, "task", "alice", tracer)
    assert result.content.startswith("alice says hello")
    ctx = agent.contexts[0]
    assert (ctx.trace_id, ctx.run_id, ctx.agent_name, ctx.agent_role) == ("T1", tracer.run_id, "alice", "analyst")
    assert current_run_context() is None  # reset afterwards
    assert tracer.agents_spawned[0]["status"] == "success"


async def test_execute_agent_records_failure_and_reraises():
    tracer = RunTracer(task="t", routing_tier="dynamic")

    def boom(task):
        raise ValueError("model down")

    with pytest.raises(ValueError):
        await execution.execute_agent(FakeAgent("bob", respond=boom), "task", "bob", tracer)
    assert tracer.agents_spawned[0]["status"] == "failed"
    assert current_run_context() is None


async def test_execute_agent_without_tracer_inherits_outer_context():
    agent = FakeAgent("carol")
    with use_run_context(RunContext(trace_id="outer", run_id="r-outer")):
        await execution.execute_agent(agent, "task", "carol", None)
    assert (agent.contexts[0].trace_id, agent.contexts[0].agent_name) == ("outer", "carol")
    await execution.execute_agent(agent, "task", "carol", None)  # no outer context: fresh IDs
    assert agent.contexts[1].trace_id == agent.contexts[1].run_id != "outer"


async def test_manager_run_task_uses_execute_agent(spy):
    tracer = RunTracer(task="t", routing_tier="dynamic", trace_id="T2")
    manager = AgentManager(FakeFactory(), tracer=tracer)
    agent = await manager.spawn(role="r", name="alice")
    out = await manager.run_task(agent.agent_id, "hi")
    assert out.startswith("alice says hello")
    assert spy == ["alice"]
    assert agent.contexts[0].trace_id == "T2"
    assert tracer.agents_spawned[0]["agent_id"] == "alice"


@pytest.mark.parametrize("spec", [pipeline("a", "b"), parallel("a", "b")], ids=["pipeline", "parallel"])
async def test_graph_runner_uses_execute_agent(spy, spec):
    tracer = RunTracer(task="t", routing_tier="dynamic", trace_id="T3")
    factory = FakeFactory()
    events = await collect(GraphRunner(factory, spec, tracer=tracer).run_stream("goal"))
    assert [e.agent for e in events if isinstance(e, ConversationMessage)] == ["a", "b"]
    assert isinstance(events[-1], StopSignal)
    assert spy == ["a", "b"]
    for name in ("a", "b"):
        ctx = factory.created[name].contexts[0]
        assert (ctx.trace_id, ctx.agent_name, ctx.agent_role) == ("T3", name, f"{name}-role")


async def test_graph_retry_still_built_on_top(spy):
    attempts = []

    def flaky(task):
        attempts.append(task)
        return "short" if len(attempts) == 1 else f"good output{LONG}"

    events = await collect(GraphRunner(FakeFactory({"a": flaky}), parallel("a")).run_stream("goal"))
    assert spy == ["a", "a"]
    assert events[0].content.startswith("good output")
