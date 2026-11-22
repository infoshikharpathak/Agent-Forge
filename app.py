from __future__ import annotations

"""
agent-forge Streamlit frontend.

Requires the backend to be running:
    uvicorn agent_forge.api.app:app --reload --port 8000

Run:
    streamlit run app.py
"""

import json

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

# ── API helpers ───────────────────────────────────────────────────────────────

def stream_run(goal: str, max_rounds: int):
    """Consume live SSE events from POST /run/stream."""
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{API_URL}/run/stream",
            params={"detail": "full"},
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            json={"goal": goal, "max_rounds": max_rounds},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])


def fetch_tools() -> list[str]:
    try:
        return httpx.get(f"{API_URL}/tools", timeout=5).json().get("tools", [])
    except Exception:
        return []


def fetch_traces(
    *, limit: int = 20, app_id: str | None = None, status: str | None = None,
    since: str | None = None, cost_above: float | None = None,
) -> list[dict]:
    params: dict = {"limit": limit}
    if app_id:
        params["app_id"] = app_id
    if status:
        params["status"] = status
    if since:
        params["since"] = since
    if cost_above is not None:
        params["cost_above"] = cost_above
    try:
        resp = httpx.get(f"{API_URL}/traces", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Failed to fetch traces: {exc}")
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

_AGENT_COLORS = ["#4F8EF7", "#F7874F", "#4FD18C", "#F7CF4F", "#C44FF7", "#F74F6E"]

def _agent_color(name: str, agent_names: list[str]) -> str:
    idx = agent_names.index(name) if name in agent_names else 0
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


def render_conversation(messages: list[dict], agent_names: list[str], stop: dict | None) -> str:
    """Render all agent messages as a single pure-HTML string.

    Avoids mixing Markdown ``---`` separators with HTML ``<span>`` tags,
    which causes the Markdown parser to silently drop intermediate sections.
    """
    if not messages:
        return ""
    parts = []
    current_round = 0
    for msg in messages:
        if msg["round"] != current_round:
            current_round = msg["round"]
            parts.append(f'<p><strong>── Round {current_round} ──</strong></p>')
        color = _agent_color(msg["agent"], agent_names)
        content_html = msg["content"].replace("\n", "<br>")
        parts.append(
            f'<div style="margin-bottom:1em">'
            f'<span style="color:{color}; font-weight:600">{msg["agent"]}</span><br>'
            f'<span>{content_html}</span>'
            f'</div>'
            f'<hr style="border:none;border-top:1px solid #333;margin:0.5em 0"/>'
        )
    if stop:
        icon = "✅" if stop["stopped_by"] == "orchestrator" else "⏹️"
        parts.append(f'<p>{icon} <em>{stop["reason"]}</em></p>')
    return "".join(parts)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="agent-forge", page_icon="🔨", layout="wide")
st.title("🔨 agent-forge")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_orch, tab_agents, tab_quality, tab_traces = st.tabs(
    ["💬 Chat", "🧠 Orchestrator", "🤖 Agent Activity", "🛡️ Quality Guardrails", "📈 Traces"]
)

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    DEFAULT_GOAL = (
        "What is the current stock price of Nvidia and Tesla, and given today's market "
        "conditions and recent AI news, which one is a better buy right now?"
    )
    goal = st.text_area("Goal", value=DEFAULT_GOAL, height=80)
    col_btn, col_rounds = st.columns([3, 1])
    with col_btn:
        run_btn = st.button("▶ Run", type="primary", disabled=not goal.strip())
    with col_rounds:
        max_rounds = st.number_input("Max rounds", min_value=1, max_value=10, value=3)
    routing_badge = st.empty()
    st.divider()
    chat_status = st.empty()
    final_report = st.empty()

# ── Tab 2: Orchestrator ───────────────────────────────────────────────────────

with tab_orch:
    routing_display = st.empty()
    orch_status = st.empty()
    orch_raw = st.empty()
    orch_details = st.container()

# ── Tab 3: Agent Activity ─────────────────────────────────────────────────────

with tab_agents:
    agents_legend = st.empty()
    st.divider()
    conv_display = st.empty()

with tab_quality:
    st.markdown("#### 🎯 Goal Clarity")
    goal_clarity_display = st.empty()
    st.divider()
    st.markdown("#### 📋 Plan Validation")
    plan_validation_display = st.empty()
    st.divider()
    st.markdown("#### ✅ Quality Check")
    quality_check_display = st.empty()
    st.divider()
    st.markdown("#### 🔍 Grounding Check")
    grounding_check_display = st.empty()

# ── Tab 5: Traces ─────────────────────────────────────────────────────────────
# Independent of the Chat tab's run — reads persisted history from GET /traces.

with tab_traces:
    st.markdown("#### 📈 Run History")
    col_a, col_b, col_c, col_d, col_e = st.columns([2, 2, 1, 2, 1])
    with col_a:
        trace_app_filter = st.text_input("App ID", value="", key="trace_app_filter")
    with col_b:
        trace_status_filter = st.selectbox(
            "Status", ["", "success", "partial", "failed"], key="trace_status_filter"
        )
    with col_c:
        trace_limit = st.number_input("Limit", min_value=1, max_value=200, value=20, key="trace_limit")
    with col_d:
        trace_cost_above_raw = st.text_input("Cost above ($)", value="", key="trace_cost_above")
    with col_e:
        st.write("")
        st.button("🔄 Refresh", key="refresh_traces")

    try:
        trace_cost_above = float(trace_cost_above_raw) if trace_cost_above_raw.strip() else None
    except ValueError:
        st.warning("Cost above must be a number — ignoring filter.")
        trace_cost_above = None

    traces = fetch_traces(
        limit=int(trace_limit),
        app_id=trace_app_filter or None,
        status=trace_status_filter or None,
        cost_above=trace_cost_above,
    )

    if not traces:
        st.info("No traces yet — run a goal from the Chat tab, or adjust filters.")
    else:
        st.dataframe(
            [
                {
                    "run_id": t["run_id"][:8],
                    "timestamp": t["timestamp"][:19],
                    "tier": t["routing_tier"],
                    "app_id": t.get("app_id") or "-",
                    "framework": t.get("framework_used") or "-",
                    "outcome": t["outcome"],
                    "latency_ms": t["total_latency_ms"],
                    # None: the model has no price in agent_forge.pricing
                    "cost_usd": "—" if t["total_cost_usd"] is None else f"{t['total_cost_usd']:.6f}",
                    "tokens_in": t["total_input_tokens"],
                    "tokens_out": t["total_output_tokens"],
                }
                for t in traces
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("##### Run detail")
        run_options = {f"{t['run_id'][:8]} — {t['timestamp'][:19]} ({t['outcome']})": t for t in traces}
        selected_label = st.selectbox("Select a run to inspect", list(run_options.keys()), key="trace_detail_select")
        if selected_label:
            detail = run_options[selected_label]
            with st.expander("Task / goal sent to agent-forge"):
                st.text(detail["task"])
            if detail.get("error"):
                st.error(detail["error"])
            if detail["guardrails_triggered"]:
                st.warning(f"Guardrails triggered: {', '.join(detail['guardrails_triggered'])}")

            if detail.get("report"):
                st.markdown("**Final report**")
                st.markdown(detail["report"])
                st.divider()

            if detail["agents_spawned"]:
                st.markdown("**Agents spawned**")
                st.dataframe(
                    [{k: v for k, v in a.items() if k != "content"} for a in detail["agents_spawned"]],
                    use_container_width=True, hide_index=True,
                )
                st.markdown("**Agent output**")
                for a in detail["agents_spawned"]:
                    with st.expander(f"`{a['agent_id']}` — {a['model']} ({a['status']})"):
                        st.markdown(a.get("content") or "_(no content recorded)_")
            else:
                st.caption("No agent-level data recorded for this run.")

if not run_btn:
    st.stop()

# ── Stream from API ───────────────────────────────────────────────────────────

chat_status.info("⏳ Connecting to backend...")
orch_status.info("Researching goal...")

# State accumulated across events
tool_calls: list[dict] = []
plan_text = ""
specs: list[dict] = []
agent_names: list[str] = []
conv_messages: list[dict] = []
stop_signal: dict | None = None
synthesis_text = ""
plan_validations: list[dict] = []
quality_checks: list[dict] = []

for event in stream_run(goal, int(max_rounds)):
    etype = event.get("type")

    if etype == "app_routed":
        tier = event["routing_tier"]
        matched_app = event.get("app_id")
        if tier == "dynamic":
            badge = "🧭 **dynamic** — no registered app matched, planning from scratch"
            detail = "No registered app matched this goal — falling through to dynamic orchestration."
            chat_status.info("⏳ Researching goal...")
        else:
            icon = "🔒" if tier == "locked" else "🔓"
            badge = f"{icon} **{tier}** — routed to app `{matched_app}`"
            detail = (
                f"Matched registered app `{matched_app}` — "
                + ("workflow and framework are fixed, dynamic planning skipped."
                   if tier == "locked" else
                   "workflow is fixed, framework chosen per run.")
            )
            chat_status.info(f"⏳ Using app `{matched_app}` ({tier})...")
        routing_badge.markdown(badge)
        with routing_display:
            (st.success if tier != "dynamic" else st.info)(detail)

    elif etype == "goal_clarified":
        was_changed = event.get("was_changed", False)
        with goal_clarity_display:
            if was_changed:
                st.warning(
                    f"**Goal was clarified.**\n\n"
                    f"**Original:** {event['original']}\n\n"
                    f"**Clarified:** {event['clarified']}\n\n"
                    f"*{event['reasoning']}*"
                )
                chat_status.info("⏳ Goal clarified — researching...")
            else:
                st.success(f"Goal is specific — no changes needed.\n\n*{event['reasoning']}*")
                chat_status.info("⏳ Researching goal...")

    elif etype == "orchestrator_tool_call":
        tool_calls.append(event)
        orch_status.info(f"🔍 Researched with {len(tool_calls)} tool call(s) — planning...")

    elif etype == "plan_validation":
        plan_validations.append(event)
        attempt = event["attempt"]
        if not event["valid"]:
            # Reset plan_text so the next attempt renders cleanly
            plan_text = ""
            chat_status.info(f"⏳ Plan attempt {attempt} rejected — replanning...")
        with plan_validation_display:
            parts = []
            for pv in plan_validations:
                if pv["valid"]:
                    parts.append(f"**Attempt {pv['attempt']}:** ✅ Plan accepted.")
                else:
                    parts.append(f"**Attempt {pv['attempt']}:** ❌ Rejected — {pv['feedback']}")
            st.markdown("\n\n".join(parts))

    elif etype == "plan_chunk":
        plan_text += event["text"]
        orch_raw.code(plan_text + "▌", language="json")

    elif etype == "plan_ready":
        orch_raw.code(plan_text, language="json")
        strategy = event.get("strategy", "autogen")

        # Normalise: both strategies expose a flat list of node/agent dicts
        if strategy == "langgraph":
            spec_obj = event["spec"]
            specs = spec_obj["nodes"]
            edges = spec_obj.get("edges", [])
            orch_status.success(f"Planned LangGraph pipeline — {len(specs)} node(s)")
        else:
            specs = event["specs"]
            edges = []
            orch_status.success(f"Planned AutoGen team — {len(specs)} agent(s)")

        agent_names = [s["name"] for s in specs]

        with orch_details:
            if tool_calls:
                st.markdown("#### 🔍 Research")
                for tc in tool_calls:
                    label = f"`{tc['tool']}({', '.join(f'{k}={v!r}' for k, v in tc['args'].items())})`"
                    with st.expander(label):
                        st.markdown(tc["result"])

            if strategy == "langgraph" and edges:
                st.markdown("#### 🔗 Graph edges")
                for e in edges:
                    label = f"- `{e['from']}` → `{e['to']}`"
                    if e.get("condition"):
                        label += f" *(if {e['condition']})*"
                    st.markdown(label)

            label = "Graph nodes" if strategy == "langgraph" else "Agent specs"
            st.markdown(f"#### {label}")
            for spec in specs:
                with st.expander(f"`{spec['name']}` — {spec['role_description']}"):
                    st.markdown(f"**System prompt**\n\n{spec['system_prompt']}")
                    if spec.get("tools"):
                        st.markdown(f"**Tools:** {', '.join(spec['tools'])}")

        # Agent/node legend in Tab 3
        with agents_legend:
            cols = st.columns(len(specs))
            for i, spec in enumerate(specs):
                color = _agent_color(spec["name"], agent_names)
                cols[i].markdown(
                    f'<span style="color:{color}; font-weight:700">● {spec["name"]}</span><br>'
                    f'<span style="font-size:0.85em">{spec["role_description"]}</span>',
                    unsafe_allow_html=True,
                )

        label = "pipeline" if strategy == "langgraph" else "conversation"
        chat_status.info(f"⏳ Agents in {label}...")

    elif etype == "agent_message":
        conv_messages.append(event)
        conv_display.markdown(
            render_conversation(conv_messages, agent_names, None),
            unsafe_allow_html=True,
        )

    elif etype == "stop_signal":
        stop_signal = event
        conv_display.markdown(
            render_conversation(conv_messages, agent_names, stop_signal),
            unsafe_allow_html=True,
        )
        chat_status.info("⏳ Synthesizing final report...")

    elif etype == "quality_check":
        quality_checks.append(event)
        if not event["passes"] and event["attempt"] < 2:
            chat_status.info("⏳ Quality check failed — re-synthesizing...")
        with quality_check_display:
            parts = []
            for qc in quality_checks:
                if qc["passes"]:
                    parts.append(f"**Attempt {qc['attempt']}:** ✅ Report passed quality check.")
                else:
                    parts.append(f"**Attempt {qc['attempt']}:** ❌ Failed — {qc['feedback']}")
            st.markdown("\n\n".join(parts))

    elif etype == "grounding_check":
        unsupported = event.get("unsupported_claims", [])
        with grounding_check_display:
            if event.get("grounded", True):
                st.success("All claims are grounded in the agent conversation.")
            else:
                st.warning(f"{len(unsupported)} unsupported claim(s) found:")
                for claim in unsupported:
                    st.markdown(f"- {claim}")

    elif etype == "synthesis_chunk":
        synthesis_text += event["text"]
        final_report.markdown(synthesis_text + "▌")

    elif etype == "done":
        final_report.markdown(synthesis_text)
        chat_status.success("✅ Done")

    elif etype == "error":
        chat_status.error(f"Backend error: {event['message']}")
        st.stop()
