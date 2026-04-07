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
import logging
from enum import Enum
from typing import AsyncGenerator

log = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.backends.langgraph import LangGraphFactory
from agent_forge.core.conversation import AgentConversation, ConversationMessage, StopSignal
from agent_forge.core.graph_runner import GraphRunner
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import AgentSpec, GraphSpec, Orchestrator
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

    Pipeline stages:
    1. Goal Clarity Pre-check  — rewrite vague goals before anything else
    2. Research                — tool calls run ONCE (not repeated on plan retries)
    3. Plan + Validate loop    — up to 3 attempts; validator rejects bad plans
    4. Execute                 — autogen debate or langgraph pipeline
    5. Synthesize + Quality loop — up to 1 retry with feedback
    6. Grounding Check         — verify synthesis claims are backed by agent output
    """
    log.info("Pipeline started | goal=%r | provider=%s", req.goal[:80], req.provider)
    orchestrator = Orchestrator(provider=req.provider)

    # ── Guardrail 1: Goal Clarity ─────────────────────────────────────────────
    log.info("[guardrail:goal_clarity] checking goal specificity")
    clarity = await orchestrator.clarify_goal(req.goal)
    effective_goal = clarity["clarified_goal"]
    log.info("[guardrail:goal_clarity] was_changed=%s reasoning=%r",
             clarity["was_changed"], clarity.get("reasoning", ""))
    yield "goal_clarified", json.dumps({
        "original": req.goal,
        "clarified": effective_goal,
        "reasoning": clarity.get("reasoning", ""),
        "was_changed": clarity["was_changed"],
    })

    # ── Research (runs ONCE before the plan retry loop) ───────────────────────
    log.info("[research] starting")
    tool_calls, research_text = await orchestrator._research(effective_goal)
    for tc in tool_calls:
        log.info("[research] tool_call=%s args=%s", tc.tool, tc.args)
        yield "orchestrator_tool_call", json.dumps({
            "tool": tc.tool, "args": tc.args, "result": tc.result,
        })

    # ── Guardrail 2: Plan + Validate loop (max 3 attempts) ───────────────────
    plan: list[AgentSpec] | GraphSpec | None = None
    plan_feedback: str | None = None
    MAX_PLAN_RETRIES = 3

    for plan_attempt in range(1, MAX_PLAN_RETRIES + 1):
        log.info("[plan] attempt=%d", plan_attempt)
        plan_chunks: list[str] = []

        async for item in orchestrator._generate_plan_stream(
            effective_goal, research_text, feedback=plan_feedback
        ):
            if isinstance(item, GraphSpec):
                plan = item
                node_names = [n.name for n in item.nodes]
                edges = [(e.from_node, e.to_node) for e in item.edges]
                log.info("[plan] strategy=langgraph nodes=%s edges=%s entry=%s",
                         node_names, edges, item.entry)
                yield "plan_ready", json.dumps({
                    "strategy": "langgraph",
                    "spec": item.model_dump(by_alias=True),
                })
            elif isinstance(item, list):
                plan = item
                log.info("[plan] strategy=autogen agents=%s", [s.name for s in item])
                yield "plan_ready", json.dumps({
                    "strategy": "autogen",
                    "specs": [s.model_dump() for s in item],
                })
            else:
                plan_chunks.append(item)
                yield "plan_chunk", json.dumps({"text": item})

        validation = await orchestrator.validate_plan(effective_goal, "".join(plan_chunks))
        log.info("[guardrail:plan_validator] attempt=%d valid=%s feedback=%r",
                 plan_attempt, validation["valid"], validation.get("feedback", ""))
        yield "plan_validation", json.dumps({
            "valid": validation["valid"],
            "feedback": validation.get("feedback", ""),
            "attempt": plan_attempt,
        })

        if validation["valid"]:
            break

        plan_feedback = validation["feedback"]
        if plan_attempt == MAX_PLAN_RETRIES:
            log.warning("[guardrail:plan_validator] max retries reached — proceeding with last plan")

    # ── Execute — AutoGen debate or LangGraph pipeline ────────────────────────
    context_source: AgentConversation | GraphRunner

    if isinstance(plan, GraphSpec):
        log.info("[execute] strategy=langgraph")
        factory = LangGraphFactory(provider=req.provider)
        runner = GraphRunner(factory=factory, spec=plan)
        async for item in runner.run_stream(effective_goal):
            if isinstance(item, ConversationMessage):
                log.info("[node] %s | %d chars", item.agent, len(item.content))
                yield "agent_message", json.dumps({
                    "agent": item.agent, "content": item.content, "round": item.round,
                })
            elif isinstance(item, StopSignal):
                log.info("[stop] %s (by=%s)", item.reason, item.stopped_by)
                yield "stop_signal", json.dumps({
                    "reason": item.reason, "stopped_by": item.stopped_by,
                })
        context_source = runner

    else:
        log.info("[execute] strategy=autogen")
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
            log.info("[spawn] agent=%s tools=%s", spec.name, spec.tools)
            agents.append(agent)

        conversation = AgentConversation(
            manager=manager,
            agents=agents,
            orchestrator=orchestrator,
            max_rounds=req.max_rounds,
        )
        async for item in conversation.run_stream(effective_goal):
            if isinstance(item, ConversationMessage):
                log.info("[round %d] %s | %d chars", item.round, item.agent, len(item.content))
                yield "agent_message", json.dumps({
                    "agent": item.agent, "content": item.content, "round": item.round,
                })
            elif isinstance(item, StopSignal):
                log.info("[stop] %s (by=%s)", item.reason, item.stopped_by)
                yield "stop_signal", json.dumps({
                    "reason": item.reason, "stopped_by": item.stopped_by,
                })
        await manager.shutdown()
        context_source = conversation

    # Capture conversation text once — grounding check always uses the raw version
    original_conversation_text = context_source.to_context_text()
    synthesis_context = original_conversation_text

    # ── Guardrail 3: Synthesize + Quality Check loop (max 1 retry) ───────────
    report = ""
    synthesis_feedback: str | None = None
    MAX_QUALITY_RETRIES = 2  # 1 initial + 1 retry

    for quality_attempt in range(1, MAX_QUALITY_RETRIES + 1):
        log.info("[synthesize] attempt=%d", quality_attempt)

        # Buffer chunks — only stream to client after quality check passes
        # (prevents showing a failing synthesis that gets replaced)
        synthesis_chunks: list[str] = []
        async for chunk in orchestrator.synthesize_stream(
            effective_goal, synthesis_context, feedback=synthesis_feedback
        ):
            synthesis_chunks.append(chunk)

        report = "".join(synthesis_chunks)

        qc = await orchestrator.quality_check(effective_goal, report)
        log.info("[guardrail:quality_check] attempt=%d passes=%s feedback=%r",
                 quality_attempt, qc["passes"], qc.get("feedback", ""))
        yield "quality_check", json.dumps({
            "passes": qc["passes"],
            "feedback": qc.get("feedback", ""),
            "attempt": quality_attempt,
        })

        if qc["passes"] or quality_attempt == MAX_QUALITY_RETRIES:
            # Stream the final (passing or last-attempt) synthesis to client
            for chunk in synthesis_chunks:
                yield "synthesis_chunk", json.dumps({"text": chunk})
            break

        # Prepare retry — append feedback to context so synthesizer knows what to fix
        synthesis_feedback = qc["feedback"]
        synthesis_context = original_conversation_text

    # ── Guardrail 4: Grounding Check ──────────────────────────────────────────
    log.info("[guardrail:grounding_check] verifying claims against agent conversation")
    gc = await orchestrator.grounding_check(effective_goal, original_conversation_text, report)
    log.info("[guardrail:grounding_check] grounded=%s unsupported_count=%d",
             gc["grounded"], len(gc.get("unsupported_claims", [])))
    yield "grounding_check", json.dumps({
        "grounded": gc["grounded"],
        "unsupported_claims": gc.get("unsupported_claims", []),
    })

    log.info("[done] result_len=%d", len(report))
    yield "done", json.dumps({"result": report})


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


_ORCHESTRATION_EVENTS = {
    "orchestrator_tool_call", "plan_chunk", "plan_ready",
    "goal_clarified", "plan_validation", "quality_check", "grounding_check",
}
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
