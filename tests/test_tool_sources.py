"""ToolSource: registration, RunContext propagation, every backend, orchestrator isolation."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_forge.core.context import RunContext, use_run_context
from agent_forge.core.execution import execute_agent
from agent_forge.tools import _REGISTRY, get_openai_schemas, get_tool, list_tools
from agent_forge.tools.sources import (
    ToolSourceError,
    ToolSpec,
    register_tool_source,
    tool_source,
    unregister_tool_source,
)
from agent_forge.tracing import RunTracer

from .fakes import FakeAgent

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "description": "Order to look up"},
        "verbose": {"type": "boolean"},
    },
    "required": ["order_id"],
}


class FakeSource:
    def __init__(self, names=("lookup_order",), delay=0.0):
        self._names = names
        self._delay = delay
        self.calls: list[tuple[str, dict, RunContext]] = []

    async def list_tools(self):
        return [ToolSpec(n, f"{n} from the fake source", SCHEMA) for n in self._names]

    async def call(self, name, args, ctx):
        self.calls.append((name, args, ctx))
        await asyncio.sleep(self._delay)
        return f"{name}:{args['order_id']}:trace={ctx.trace_id}:agent={ctx.agent_name}"


@pytest.fixture
async def source():
    src = FakeSource()
    await register_tool_source(src)
    yield src
    unregister_tool_source(src)


def tracer(trace_id):
    return RunTracer(task="t", routing_tier="dynamic", trace_id=trace_id)


# ── registration ──────────────────────────────────────────────────────────────

async def test_registered_tool_has_schema_and_is_listed(source):
    assert "lookup_order" in list_tools()
    (schema,) = get_openai_schemas(["lookup_order"])
    assert schema["function"]["parameters"] == SCHEMA
    assert schema["function"]["description"] == "lookup_order from the fake source"


async def test_collision_raises_and_registers_nothing(source):
    clash = FakeSource(names=("brand_new", "lookup_order"))
    with pytest.raises(ValueError, match="lookup_order"):
        await register_tool_source(clash)
    assert "brand_new" not in _REGISTRY
    with pytest.raises(ValueError, match="web_search"):
        await register_tool_source(FakeSource(names=("web_search",)))  # built-in
    with pytest.raises(ValueError, match="listed twice"):
        await register_tool_source(FakeSource(names=("x1", "x1")))


async def test_unregister_and_context_manager():
    src = FakeSource(names=("temp_tool",))
    async with tool_source(src) as names:
        assert names == ["temp_tool"] and "temp_tool" in list_tools()
    assert "temp_tool" not in list_tools()
    assert unregister_tool_source(src) == []  # idempotent


async def test_call_outside_agent_run_fails(source):
    with pytest.raises(ToolSourceError, match="outside an agent run"):
        await get_tool("lookup_order")(order_id="A1")


async def test_wrapper_reads_current_context(source):
    with use_run_context(RunContext(trace_id="T", run_id="R", agent_name="alice")):
        out = await get_tool("lookup_order")(order_id="A1")
    assert out == "lookup_order:A1:trace=T:agent=alice"
    assert source.calls[0][1] == {"order_id": "A1"}


# ── concurrency: no context bleed ─────────────────────────────────────────────

async def test_two_concurrent_runs_keep_their_own_context():
    src = FakeSource(delay=0.01)
    async with tool_source(src):
        async def agent_logic(task):
            await asyncio.sleep(0.005)  # interleave the two runs
            return await get_tool("lookup_order")(order_id=task)

        a = FakeAgent("alice", respond=agent_logic)
        b = FakeAgent("bob", respond=agent_logic)
        await asyncio.gather(
            execute_agent(a, "A1", "alice", tracer("trace-A")),
            execute_agent(b, "B1", "bob", tracer("trace-B")),
        )
    seen = {(args["order_id"], ctx.trace_id, ctx.agent_name) for _, args, ctx in src.calls}
    assert seen == {("A1", "trace-A", "alice"), ("B1", "trace-B", "bob")}


# ── backends ──────────────────────────────────────────────────────────────────

def _assert_called(src, trace_id, agent):
    (name, args, ctx) = src.calls[-1]
    assert name == "lookup_order" and args["order_id"] == "A1"
    assert (ctx.trace_id, ctx.agent_name) == (trace_id, agent)


async def test_openai_backend_calls_source_tool(source):
    from openai.types.chat import ChatCompletion

    from agent_forge.backends.openai import OpenAIFactory

    def completion(message, finish):
        return ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    replies = [
        completion({"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "A1"})},
        }]}, "tool_calls"),
        completion({"role": "assistant", "content": "done"}, "stop"),
    ]
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        return replies.pop(0)

    factory = OpenAIFactory()
    factory._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent = await factory.create(role="support", name="openai_agent", tools=["lookup_order"])
    result = await execute_agent(agent, "find A1", "openai_agent", tracer("T-openai"))

    assert result.content == "done"
    assert sent[0]["tools"][0]["function"]["parameters"] == SCHEMA
    _assert_called(source, "T-openai", "openai_agent")
    tool_msg = sent[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "trace=T-openai" in tool_msg["content"]


async def test_anthropic_backend_calls_source_tool(source):
    from anthropic.types import Message

    from agent_forge.backends.anthropic import AnthropicFactory

    def message(content, stop):
        return Message.model_validate({
            "id": "m", "type": "message", "role": "assistant", "model": "claude",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    replies = [
        message([{"type": "tool_use", "id": "tu_1", "name": "lookup_order", "input": {"order_id": "A1"}}], "tool_use"),
        message([{"type": "text", "text": "done"}], "end_turn"),
    ]
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        return replies.pop(0)

    factory = AnthropicFactory()
    factory._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    agent = await factory.create(role="support", name="claude_agent", tools=["lookup_order"])
    result = await execute_agent(agent, "find A1", "claude_agent", tracer("T-anthropic"))

    assert result.content == "done"
    assert sent[0]["tools"][0]["input_schema"] == SCHEMA
    _assert_called(source, "T-anthropic", "claude_agent")


async def test_langgraph_backend_calls_source_tool(source, monkeypatch):
    import langchain_openai
    from langchain_core.messages import AIMessage

    from agent_forge.backends.langgraph import LangGraphFactory

    bound = []

    class FakeChat:
        def __init__(self, **kwargs):
            self._replies = [
                AIMessage(content="", tool_calls=[{"name": "lookup_order", "args": {"order_id": "A1"}, "id": "t1"}]),
                AIMessage(content="done"),
            ]

        def bind_tools(self, tools):
            bound.extend(tools)
            return self

        async def ainvoke(self, messages):
            self.last_messages = messages
            return self._replies.pop(0)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)
    agent = await LangGraphFactory().create(role="support", name="lg_agent", system_message="s", tools=["lookup_order"])
    result = await execute_agent(agent, "find A1", "lg_agent", tracer("T-langgraph"))

    assert result.content == "done"
    assert bound[0].name == "lookup_order"
    _assert_called(source, "T-langgraph", "lg_agent")


async def test_autogen_backend_calls_source_tool(source, monkeypatch):
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, ModelFamily, ModelInfo, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    from agent_forge.backends.autogen import AutoGenFactory

    replay = ReplayChatCompletionClient(
        [
            CreateResult(
                finish_reason="function_calls",
                content=[FunctionCall(id="1", name="lookup_order", arguments=json.dumps({"order_id": "A1"}))],
                usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
                cached=False,
            ),
            "done",
        ],
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False,
                             family=ModelFamily.UNKNOWN, structured_output=False),
    )
    monkeypatch.setattr(AutoGenFactory, "_build_model_client", lambda self, model_override=None: replay)
    agent = await AutoGenFactory().create(role="support", name="ag_agent", system_message="s", tools=["lookup_order"])

    # AutoGen builds its schema from the wrapper's signature
    (tool,) = agent.native._tools
    assert set(tool.schema["parameters"]["properties"]) == {"order_id", "verbose"}
    assert tool.schema["parameters"]["required"] == ["order_id"]

    result = await execute_agent(agent, "find A1", "ag_agent", tracer("T-autogen"))
    assert "trace=T-autogen" in result.content
    _assert_called(source, "T-autogen", "ag_agent")


# ── orchestrator isolation ────────────────────────────────────────────────────

async def test_orchestrator_research_never_calls_source_tools(source, monkeypatch):
    from openai.types.chat import ChatCompletion

    from agent_forge.core.orchestrator import Orchestrator

    def completion(message):
        return ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        })

    replies = [
        completion({"role": "assistant", "content": None, "tool_calls": [{
            "id": "x", "type": "function",
            "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "A1"})},
        }]}),
        completion({"role": "assistant", "content": "ok"}),
    ]
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        return replies.pop(0)

    orch = Orchestrator(provider="openai")
    monkeypatch.setattr(orch, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    calls, _ = await orch._research("goal")

    assert source.calls == []
    offered = {t["function"]["name"] for t in sent[0]["tools"]}
    assert offered == {"web_search", "get_datetime"}
    assert "not available to the orchestrator" in calls[-1].result
    # the planner still sees the tool, so it can assign it to agents
    assert "lookup_order" in orch._build_system_prompt()


def test_api_registers_sources_for_app_lifetime():
    from fastapi.testclient import TestClient

    from agent_forge.api.app import create_app

    src = FakeSource(names=("api_tool",))
    with TestClient(create_app(tool_sources=[src])):
        assert "api_tool" in list_tools()
    assert "api_tool" not in list_tools()
