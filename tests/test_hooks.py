"""AgentHooks: before/after/handoff on both execution paths."""

import pytest
from fastapi.testclient import TestClient

from agent_forge import db
from agent_forge.core.conversation import AgentConversation, ConversationMessage, StopSignal
from agent_forge.core.execution import execute_agent
from agent_forge.core.graph_runner import GraphRunner
from agent_forge.core.hooks import AgentHooks, HookDecision
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import GraphEdge, GraphSpec
from agent_forge.core.shared_thread import SharedThread
from agent_forge.tracing import RunTracer

from .fakes import LONG, FakeAgent, FakeFactory, collect, node, parallel, pipeline


class Recorder(AgentHooks):
    """Records every call; behaviour per event set via attributes."""

    def __init__(self, before=None, after=None, handoff=None):
        self.calls = []
        self._before, self._after, self._handoff = before, after, handoff

    async def before_agent_run(self, ctx, task):
        self.calls.append(("before", ctx.agent_name, task))
        return self._before(ctx, task) if self._before else HookDecision.allow()

    async def after_agent_run(self, ctx, output):
        self.calls.append(("after", ctx.agent_name, output))
        return self._after(ctx, output) if self._after else HookDecision.allow()

    async def on_handoff(self, ctx, from_agent, to_agent, payload):
        self.calls.append(("handoff", from_agent, to_agent, payload))
        return self._handoff(from_agent, to_agent, payload) if self._handoff else HookDecision.allow()

    def events(self, kind):
        return [c for c in self.calls if c[0] == kind]


class Boom(AgentHooks):
    async def before_agent_run(self, ctx, task):
        raise RuntimeError("policy service unreachable")

    async def on_handoff(self, ctx, from_agent, to_agent, payload):
        raise RuntimeError("handoff checker down")


def tracer():
    return RunTracer(task="t", routing_tier="dynamic", trace_id="T")


# ── execute_agent: before / after ──────────────────────────────────────────────

async def test_before_deny_skips_agent_and_records():
    tr = tracer()
    agent = FakeAgent("alice")
    hook = Recorder(before=lambda ctx, task: HookDecision.deny("No secrets allowed."))
    result = await execute_agent(agent, "the password is hunter2", "alice", tr, [hook])
    assert agent.tasks == []  # agent.run() never called
    assert (result.content, result.status) == ("No secrets allowed.", "blocked")
    assert tr.guardrails_triggered == ["hook:before_agent_run:blocked:alice"]
    assert tr.agents_spawned[0]["status"] == "blocked"
    assert hook.events("after") == []


async def test_before_allow_with_content_replaces_task():
    agent = FakeAgent("alice")
    hook = Recorder(before=lambda ctx, task: HookDecision.allow(task.replace("hunter2", "[redacted]")))
    await execute_agent(agent, "pw hunter2", "alice", None, [hook])
    assert agent.tasks == ["pw [redacted]"]


async def test_after_deny_withholds_output():
    tr = tracer()
    hook = Recorder(after=lambda ctx, out: HookDecision.deny("Output withheld: contains PII."))
    result = await execute_agent(FakeAgent("bob"), "t", "bob", tr, [hook])
    assert (result.content, result.status) == ("Output withheld: contains PII.", "withheld")
    assert tr.guardrails_triggered == ["hook:after_agent_run:withheld:bob"]
    assert tr.agents_spawned[0]["status"] == "withheld"
    assert tr.agents_spawned[0]["input_tokens"] == 3  # the run did happen


async def test_after_allow_with_content_replaces_output():
    hook = Recorder(after=lambda ctx, out: HookDecision.allow(out.upper()))
    result = await execute_agent(FakeAgent("bob"), "t", "bob", None, [hook])
    assert result.content.startswith("BOB SAYS HELLO") and result.status == "success"


async def test_raising_hook_fails_closed():
    tr = tracer()
    agent = FakeAgent("alice")
    result = await execute_agent(agent, "t", "alice", tr, [Boom()])
    assert agent.tasks == []
    assert result.status == "blocked"
    assert "Boom.before_agent_run failed: policy service unreachable" in result.content
    assert tr.guardrails_triggered == ["hook:before_agent_run:blocked:alice"]


async def test_wrong_return_type_fails_closed():
    class Sloppy(AgentHooks):
        async def after_agent_run(self, ctx, output):
            return True

    result = await execute_agent(FakeAgent("alice"), "t", "alice", None, [Sloppy()])
    assert result.status == "withheld" and "not HookDecision" in result.content


async def test_hooks_run_in_order_and_first_deny_wins():
    first = Recorder(before=lambda ctx, task: HookDecision.allow(task + " +first"))
    second = Recorder(before=lambda ctx, task: HookDecision.deny(f"second saw: {task}"))
    third = Recorder()
    result = await execute_agent(FakeAgent("a"), "t", "a", None, [first, second, third])
    assert result.content == "second saw: t +first"
    assert third.calls == []


async def test_hook_ctx_carries_run_ids():
    hook = Recorder()
    tr = tracer()
    await execute_agent(FakeAgent("alice", role="analyst"), "t", "alice", tr, [hook])
    # ctx is the same RunContext the agent sees
    assert hook.calls[0][1] == "alice"


async def test_denial_messages_persisted_in_run_metadata():
    tr = tracer()
    hook = Recorder(before=lambda ctx, task: HookDecision.deny("blocked: keyword"))
    await execute_agent(FakeAgent("alice"), "t", "alice", tr, [hook])
    await tr.finish(outcome="success")
    row = db.get_run(tr.run_id)
    assert row["guardrails_triggered"] == ["hook:before_agent_run:blocked:alice"]
    assert row["metadata"]["hook_denials"] == [
        {"guardrail": "hook:before_agent_run:blocked:alice", "hook": "Recorder", "message": "blocked: keyword"}
    ]


# ── Conversation path ──────────────────────────────────────────────────────────

class FakeOrchestrator:
    async def convergence_check(self, **kwargs):
        return False, ""


async def conversation(hooks, responders=None, rounds=1):
    tr = tracer()
    manager = AgentManager(FakeFactory(responders), tracer=tr, hooks=hooks)
    agents = [await manager.spawn(role="r", name=n) for n in ("alice", "bob")]
    conv = AgentConversation(manager, agents, FakeOrchestrator(), max_rounds=rounds)
    return tr, agents, await collect(conv.run_stream("goal"))


async def test_conversation_fires_before_after_and_handoff():
    hook = Recorder()
    _, agents, events = await conversation([hook])
    assert [e.agent for e in events if isinstance(e, ConversationMessage)] == ["alice", "bob"]
    assert [c[1] for c in hook.events("before")] == ["alice", "bob"]
    assert [c[1] for c in hook.events("after")] == ["alice", "bob"]
    (handoff,) = hook.events("handoff")
    assert handoff[1:3] == ("alice", "bob") and handoff[3].startswith("alice says hello")


async def test_conversation_handoff_deny_ends_conversation():
    hook = Recorder(handoff=lambda f, t, p: HookDecision.deny(f"{t} may not receive {f}'s work"))
    tr, agents, events = await conversation([hook])
    assert [e.agent for e in events if isinstance(e, ConversationMessage)] == ["alice"]
    assert events[-1] == StopSignal(reason="bob may not receive alice's work", stopped_by="hook")
    assert agents[1].tasks == []
    assert tr.guardrails_triggered == ["hook:on_handoff:denied:alice->bob"]


async def test_conversation_handoff_replacement_reaches_next_agent():
    hook = Recorder(handoff=lambda f, t, p: HookDecision.allow("SANITIZED SUMMARY"))
    _, agents, events = await conversation([hook])
    bob_task = agents[1].tasks[0]
    assert "SANITIZED SUMMARY" in bob_task and "alice says hello" not in bob_task
    # stored history keeps the real message
    assert events[0].content.startswith("alice says hello")


async def test_conversation_blocked_turn_not_retried():
    hook = Recorder(before=lambda ctx, task: HookDecision.deny("x") if ctx.agent_name == "alice" else HookDecision.allow())
    _, agents, events = await conversation([hook])
    assert events[0].content == "x"  # the hook message, not the retry fallback
    assert len(hook.events("before")) == 2  # alice once (no retry), bob once


async def test_run_task_thread_handoff():
    thread = SharedThread()
    thread.add("alice", "alice's findings")
    hook = Recorder(handoff=lambda f, t, p: HookDecision.deny("no sharing"))
    tr = tracer()
    manager = AgentManager(FakeFactory(), tracer=tr, hooks=[hook])
    bob = await manager.spawn(role="r", name="bob")
    out = await manager.run_task(bob.agent_id, "continue", thread=thread)
    assert out == "no sharing" and bob.tasks == []
    assert len(thread) == 1  # nothing appended
    assert hook.events("handoff")[0][1:3] == ("alice", "bob")
    assert tr.guardrails_triggered == ["hook:on_handoff:denied:alice->bob"]


# ── Graph path ─────────────────────────────────────────────────────────────────

async def graph(spec, hooks, responders=None):
    tr = tracer()
    factory = FakeFactory(responders)
    events = await collect(GraphRunner(factory, spec, tracer=tr, hooks=hooks).run_stream("goal"))
    ran = [e.agent for e in events if isinstance(e, ConversationMessage)]
    return tr, factory, ran, events


async def test_graph_fires_before_after_and_handoff():
    hook = Recorder()
    _, _, ran, _ = await graph(pipeline("a", "b", "c"), [hook])
    assert ran == ["a", "b", "c"]
    assert [c[1] for c in hook.events("before")] == ["a", "b", "c"]
    assert [c[1:3] for c in hook.events("handoff")] == [("a", "b"), ("b", "c")]
    assert hook.events("handoff")[0][3].startswith("a says hello")


async def test_graph_handoff_deny_routes_branch_to_end():
    hook = Recorder(handoff=lambda f, t, p: HookDecision.deny("stop at b") if t == "b" else HookDecision.allow())
    tr, factory, ran, events = await graph(pipeline("a", "b", "c"), [hook])
    assert ran == ["a"]
    assert factory.created["b"].tasks == [] and factory.created["c"].tasks == []
    assert isinstance(events[-1], StopSignal)
    assert tr.guardrails_triggered == ["hook:on_handoff:denied:a->b"]
    assert tr.hook_denials[0]["message"] == "stop at b"


async def test_graph_fan_out_deny_stops_only_that_branch():
    spec = GraphSpec(
        nodes=[node("a"), node("b"), node("c")],
        edges=[GraphEdge(from_node="a", to_node="b"), GraphEdge(from_node="a", to_node="c")],
        entry="a",
    )
    hook = Recorder(handoff=lambda f, t, p: HookDecision.deny("no c") if t == "c" else HookDecision.allow())
    _, _, ran, _ = await graph(spec, [hook])
    assert ran == ["a", "b"]


async def test_graph_conditional_route_goes_through_handoff():
    spec = GraphSpec(
        nodes=[node("triage"), node("fix"), node("escalate")],
        edges=[
            GraphEdge(from_node="triage", to_node="fix", condition="minor", condition_key="FIX"),
            GraphEdge(from_node="triage", to_node="escalate", condition="major", condition_key="ESCALATE"),
        ],
        entry="triage",
    )
    responders = {"triage": lambda task: f"looks major{LONG}\n" + '{"route": "ESCALATE"}'}
    hook = Recorder()
    _, _, ran, _ = await graph(spec, [hook], responders)
    assert ran == ["triage", "escalate"]
    assert [c[1:3] for c in hook.events("handoff")] == [("triage", "escalate")]
    assert '"route"' not in hook.events("handoff")[0][3]  # payload has routing JSON stripped

    deny = Recorder(handoff=lambda f, t, p: HookDecision.deny("no escalation"))
    _, _, ran, _ = await graph(spec, [deny], responders)
    assert ran == ["triage"]


async def test_graph_handoff_replacement_reaches_next_node():
    hook = Recorder(handoff=lambda f, t, p: HookDecision.allow("REDACTED HANDOFF"))
    _, factory, _, _ = await graph(pipeline("a", "b"), [hook])
    b_task = factory.created["b"].tasks[0]
    assert "REDACTED HANDOFF" in b_task and "a says hello" not in b_task


async def test_graph_raising_handoff_hook_fails_closed():
    tr, _, ran, events = await graph(pipeline("a", "b"), [Boom()])
    # Boom also blocks before_agent_run, so a itself is blocked; no exception escapes
    assert ran == ["a"] and isinstance(events[-1], StopSignal)
    assert "hook:before_agent_run:blocked:a" in tr.guardrails_triggered
    assert "hook:on_handoff:denied:a->b" in tr.guardrails_triggered


async def test_graph_blocked_node_not_retried():
    hook = Recorder(before=lambda ctx, task: HookDecision.deny("nope"))
    _, _, _, events = await graph(parallel("a"), [hook])
    assert events[0].content == "nope"
    assert len(hook.events("before")) == 1


async def test_no_hooks_graph_unchanged():
    _, factory, ran, _ = await graph(pipeline("a", "b"), None)
    assert ran == ["a", "b"]
    assert "a says hello" in factory.created["b"].tasks[0]


# ── API layer ──────────────────────────────────────────────────────────────────

def test_create_app_passes_hooks_to_pipeline(monkeypatch):
    from agent_forge.api import app as api

    seen = []

    async def fake_pipeline(req, hooks=None):
        seen.append(hooks)
        yield "done", '{"result": "ok"}'

    monkeypatch.setattr(api, "_pipeline", fake_pipeline)
    hook = Recorder()
    assert TestClient(api.create_app(hooks=[hook])).post("/run", json={"goal": "g"}).json() == {"result": "ok"}
    assert TestClient(api.app).post("/run", json={"goal": "g"}).status_code == 200
    assert seen == [[hook], []]
