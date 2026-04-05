from __future__ import annotations

"""
agent-forge FastAPI backend.

Run:
    uvicorn agent_forge.api.app:app --reload --port 8000

Endpoints:
    GET  /health
    POST /run                          — goal → final report (blocking)
    POST /run/stream                   — goal → final report chunks (SSE)
    POST /run/stream?detail=orchestration  — adds orchestrator tool calls + plan
    POST /run/stream?detail=full           — adds agent conversation too
"""

import json
from enum import Enum
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.backends.langgraph import LangGraphFactory
from agent_forge.core.conversation import AgentConversation, ConversationMessage, StopSignal
from agent_forge.core.graph_runner import GraphRunner
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import AgentSpec, GraphSpec, Orchestrator, OrchestratorToolCall
from agent_forge.tools.mcp_bridge import MCPBridge

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="agent-forge",
    version="0.1.0",
    description=(
        "Framework-agnostic AI agent orchestration API. "
        "Send a goal, get back a synthesized report produced by a dynamically planned "
        "team of AI agents that research, debate, and converge on an answer."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class DetailLevel(str, Enum):
    """Controls how much of the internal pipeline is exposed in the SSE stream.

    - result:        Only the final synthesized report (default).
    - orchestration: Adds orchestrator research tool calls and the agent plan.
    - full:          Adds the full agent-to-agent conversation as well.
    """
    result = "result"
    orchestration = "orchestration"
    full = "full"


class RunRequest(BaseModel):
    """Request body for /run and /run/stream."""
    goal: str
    """The high-level goal or question for the agent team to address."""
    max_rounds: int = 3
    """Maximum conversation rounds between agents before forcing a stop."""
    provider: str = "openai"
    """LLM provider to use for all agents and the orchestrator."""


class RunResponse(BaseModel):
    """Response body for the blocking /run endpoint."""
    result: str
    """The final synthesized report produced by the orchestrator."""

# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _pipeline(req: RunRequest) -> AsyncGenerator[tuple[str, str], None]:
    """
    Runs the full pipeline and yields (event_type, json_str) tuples.
    Callers filter by detail level.

    Wraps the entire run in an MCPBridge context so MCP tools are
    registered before the orchestrator plans and cleaned up after shutdown.
    """
    async with MCPBridge():
        async for event in _pipeline_inner(req):
            yield event


async def _pipeline_inner(req: RunRequest) -> AsyncGenerator[tuple[str, str], None]:
    """Core pipeline logic, called inside an active MCPBridge context.

    Branches on the strategy chosen by the Orchestrator:
    - ``autogen``   → multi-round debate via AgentConversation
    - ``langgraph`` → structured pipeline via GraphRunner
    """
    orchestrator = Orchestrator(provider=req.provider)

    # Phase 1: Research + plan
    plan: list[AgentSpec] | GraphSpec | None = None
    async for item in orchestrator.plan_stream(req.goal):
        if isinstance(item, OrchestratorToolCall):
            yield "orchestrator_tool_call", json.dumps({
                "tool": item.tool, "args": item.args, "result": item.result,
            })
        elif isinstance(item, GraphSpec):
            plan = item
            yield "plan_ready", json.dumps({
                "strategy": "langgraph",
                "spec": item.model_dump(by_alias=True),
            })
        elif isinstance(item, list):
            plan = item
            yield "plan_ready", json.dumps({
                "strategy": "autogen",
                "specs": [s.model_dump() for s in item],
            })
        else:
            yield "plan_chunk", json.dumps({"text": item})

    # Phase 2 + 3: Execute — AutoGen debate or LangGraph pipeline
    context_source: AgentConversation | GraphRunner

    if isinstance(plan, GraphSpec):
        # ── LangGraph path ────────────────────────────────────────────────────
        factory = LangGraphFactory(provider=req.provider)
        runner = GraphRunner(factory=factory, spec=plan)
        async for item in runner.run_stream(req.goal):
            if isinstance(item, ConversationMessage):
                yield "agent_message", json.dumps({
                    "agent": item.agent, "content": item.content, "round": item.round,
                })
            elif isinstance(item, StopSignal):
                yield "stop_signal", json.dumps({
                    "reason": item.reason, "stopped_by": item.stopped_by,
                })
        context_source = runner

    else:
        # ── AutoGen path (default) ────────────────────────────────────────────
        specs: list[AgentSpec] = plan or []
        factory = AutoGenFactory(provider=req.provider)
        manager = AgentManager(factory)
        agents = []
        for spec in specs:
            agent = await manager.spawn(
                role=spec.role_description,
                name=spec.name,
                system_message=spec.system_prompt,
                tools=spec.tools or None,
            )
            agents.append(agent)

        conversation = AgentConversation(
            manager=manager,
            agents=agents,
            orchestrator=orchestrator,
            max_rounds=req.max_rounds,
        )
        async for item in conversation.run_stream(req.goal):
            if isinstance(item, ConversationMessage):
                yield "agent_message", json.dumps({
                    "agent": item.agent, "content": item.content, "round": item.round,
                })
            elif isinstance(item, StopSignal):
                yield "stop_signal", json.dumps({
                    "reason": item.reason, "stopped_by": item.stopped_by,
                })
        await manager.shutdown()
        context_source = conversation

    # Phase 4: Synthesize
    synthesis_chunks: list[str] = []
    async for chunk in orchestrator.synthesize_stream(req.goal, context_source.to_context_text()):
        synthesis_chunks.append(chunk)
        yield "synthesis_chunk", json.dumps({"text": chunk})

    yield "done", json.dumps({"result": "".join(synthesis_chunks)})


def _sse(type: str, data: str) -> str:
    """Format a single SSE frame.

    Merges the event type into the JSON payload so clients only need to
    parse one ``data:`` field per event.

    Args:
        type: Event type string (e.g. ``"synthesis_chunk"``).
        data: JSON-encoded event payload produced by ``_pipeline()``.

    Returns:
        A complete SSE data line ending with the required double newline.
    """
    return f"data: {json.dumps({'type': type, **json.loads(data)})}\n\n"


_ORCHESTRATION_EVENTS = {"orchestrator_tool_call", "plan_chunk", "plan_ready"}
_CONVERSATION_EVENTS = {"agent_message", "stop_signal"}


async def _filtered_stream(req: RunRequest, detail: DetailLevel) -> AsyncGenerator[str, None]:
    """Wrap ``_pipeline()`` and emit only the SSE events appropriate for *detail*.

    Always emits: ``synthesis_chunk``, ``done``, ``error``.
    Also emits when detail >= orchestration: ``orchestrator_tool_call``, ``plan_chunk``, ``plan_ready``.
    Also emits when detail == full: ``agent_message``, ``stop_signal``.

    Any exception raised by the pipeline is caught and forwarded as an
    ``error`` SSE event so the client always receives a clean stream.

    Args:
        req:    The validated run request (goal, max_rounds, provider).
        detail: How much internal detail to expose to the caller.

    Yields:
        SSE-formatted strings ready to be sent as ``text/event-stream``.
    """
    try:
        async for etype, edata in _pipeline(req):
            if etype in ("synthesis_chunk", "done", "error"):
                yield _sse(etype, edata)
            elif etype in _ORCHESTRATION_EVENTS and detail in (DetailLevel.orchestration, DetailLevel.full):
                yield _sse(etype, edata)
            elif etype in _CONVERSATION_EVENTS and detail == DetailLevel.full:
                yield _sse(etype, edata)
    except Exception as exc:
        yield _sse("error", json.dumps({"message": str(exc)}))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe. Returns ``{"status": "ok"}`` when the server is up."""
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    """Run the full pipeline and return only the final report."""
    result_parts: list[str] = []
    async for etype, edata in _pipeline(req):
        if etype == "done":
            result_parts = [json.loads(edata).get("result", "")]
        elif etype == "error":
            raise RuntimeError(json.loads(edata)["message"])
    return RunResponse(result="".join(result_parts))


@app.post("/run/stream")
async def run_stream(
    req: RunRequest,
    detail: DetailLevel = Query(default=DetailLevel.result),
):
    """
    Stream the pipeline as SSE.

    detail=result         — only final synthesis chunks (default)
    detail=orchestration  — adds orchestrator research + plan events
    detail=full           — adds agent conversation events too
    """
    return StreamingResponse(
        _filtered_stream(req, detail),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
