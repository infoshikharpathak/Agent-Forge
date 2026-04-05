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
    """Consume SSE events from POST /run/stream."""
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{API_URL}/run/stream",
            json={"goal": goal, "max_rounds": max_rounds},
        ) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])


def fetch_tools() -> list[str]:
    try:
        return httpx.get(f"{API_URL}/tools", timeout=5).json().get("tools", [])
    except Exception:
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

_AGENT_COLORS = ["#4F8EF7", "#F7874F", "#4FD18C", "#F7CF4F", "#C44FF7", "#F74F6E"]

def _agent_color(name: str, agent_names: list[str]) -> str:
    idx = agent_names.index(name) if name in agent_names else 0
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


def render_conversation(messages: list[dict], agent_names: list[str], stop: dict | None) -> str:
    if not messages:
        return ""
    md_parts = []
    current_round = 0
    for msg in messages:
        if msg["round"] != current_round:
            current_round = msg["round"]
            md_parts.append(f"**── Round {current_round} ──**")
        color = _agent_color(msg["agent"], agent_names)
        md_parts.append(
            f'<span style="color:{color}; font-weight:600">{msg["agent"]}</span>\n\n'
            f'{msg["content"]}'
        )
    if stop:
        icon = "✅" if stop["stopped_by"] == "orchestrator" else "⏹️"
        md_parts.append(f"---\n{icon} *{stop['reason']}*")
    return "\n\n---\n\n".join(md_parts)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="agent-forge", page_icon="🔨", layout="wide")
st.title("🔨 agent-forge")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_orch, tab_agents = st.tabs(["💬 Chat", "🧠 Orchestrator", "🤖 Agent Activity"])

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
    st.divider()
    chat_status = st.empty()
    final_report = st.empty()

# ── Tab 2: Orchestrator ───────────────────────────────────────────────────────

with tab_orch:
    orch_status = st.empty()
    orch_raw = st.empty()
    orch_details = st.container()

# ── Tab 3: Agent Activity ─────────────────────────────────────────────────────

with tab_agents:
    agents_legend = st.empty()
    st.divider()
    conv_display = st.empty()

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

for event in stream_run(goal, int(max_rounds)):
    etype = event.get("type")

    if etype == "orchestrator_tool_call":
        tool_calls.append(event)
        orch_status.info(f"🔍 Researched with {len(tool_calls)} tool call(s) — planning...")

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
                    st.markdown(f"- `{e['from']}` → `{e['to']}`")

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

    elif etype == "synthesis_chunk":
        synthesis_text += event["text"]
        final_report.markdown(synthesis_text + "▌")

    elif etype == "done":
        final_report.markdown(synthesis_text)
        chat_status.success("✅ Done")

    elif etype == "error":
        chat_status.error(f"Backend error: {event['message']}")
        st.stop()
