from __future__ import annotations

"""
Streamlit UI for agent-forge.

Run:
    streamlit run app.py
"""

import asyncio
import queue
import threading

import streamlit as st

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.conversation import AgentConversation, ConversationMessage, StopSignal
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import AgentSpec, Orchestrator

# ── Async helpers ─────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine synchronously in a dedicated thread."""
    result_box: list = [None]
    error_box: list = [None]

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except Exception as exc:
            error_box[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    if error_box[0]:
        raise error_box[0]
    return result_box[0]


def iter_async_stream(async_gen_factory):
    """
    Sync generator that bridges any async generator to Streamlit.
    Pass a zero-argument lambda that returns the async generator.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()

    async def drain():
        async for item in async_gen_factory():
            q.put(item)
        q.put(sentinel)

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(drain())
        finally:
            loop.close()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
    t.join()


# ── Helpers ───────────────────────────────────────────────────────────────────

# One color per agent (cycles if more than 6)
_AGENT_COLORS = ["#4F8EF7", "#F7874F", "#4FD18C", "#F7CF4F", "#C44FF7", "#F74F6E"]

def _agent_color(name: str, specs: list[AgentSpec]) -> str:
    names = [s.name for s in specs]
    idx = names.index(name) if name in names else 0
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


def render_conversation(
    messages: list[ConversationMessage],
    specs: list[AgentSpec],
    stop: StopSignal | None = None,
) -> str:
    """Render the conversation as markdown for st.markdown()."""
    if not messages:
        return ""

    md_parts = []
    current_round = 0

    for msg in messages:
        if msg.round != current_round:
            current_round = msg.round
            md_parts.append(f"**── Round {current_round} ──**")

        color = _agent_color(msg.agent, specs)
        md_parts.append(
            f'<span style="color:{color}; font-weight:600">{msg.agent}</span>\n\n'
            f"{msg.content}"
        )

    if stop:
        icon = "✅" if stop.stopped_by == "orchestrator" else "⏹️"
        md_parts.append(f"---\n{icon} *{stop.reason}*")

    return "\n\n---\n\n".join(md_parts)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="agent-forge", page_icon="🔨", layout="wide")
st.title("🔨 agent-forge")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_orch, tab_agents = st.tabs(["💬 Chat", "🧠 Orchestrator", "🤖 Agent Activity"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    DEFAULT_GOAL = (
        "Analyse the impact of rising US interest rates on emerging market equities "
        "and produce a concise investment brief."
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
    orch_specs_area = st.container()

# ── Tab 3: Agent Activity ─────────────────────────────────────────────────────

with tab_agents:
    agents_legend = st.empty()
    st.divider()
    conv_display = st.empty()

if not run_btn:
    st.stop()

# ── Step 1: Stream orchestrator planning → Tab 2 ─────────────────────────────

chat_status.info("⏳ Planning agents...")
orch_status.info("Planning agents for your goal...")

full_text = ""
specs: list[AgentSpec] = []
orchestrator = Orchestrator(provider="openai")

for item in iter_async_stream(lambda: orchestrator.plan_stream(goal)):
    if isinstance(item, list):
        specs = item
    else:
        full_text += item
        orch_raw.code(full_text + "▌", language="json")

orch_raw.code(full_text, language="json")
orch_status.success(f"Planned {len(specs)} agent(s)")

with orch_specs_area:
    st.markdown("#### Agent specs")
    for spec in specs:
        with st.expander(f"`{spec.name}` — {spec.role_description}"):
            st.markdown(f"**System prompt**\n\n{spec.system_prompt}")
            if spec.tools:
                st.markdown(f"**Tools:** {', '.join(spec.tools)}")

# ── Step 2: Show agent legend → Tab 3 ────────────────────────────────────────

with agents_legend:
    cols = st.columns(len(specs))
    for i, spec in enumerate(specs):
        color = _agent_color(spec.name, specs)
        cols[i].markdown(
            f'<span style="color:{color}; font-weight:700">● {spec.name}</span><br>'
            f'<span style="font-size:0.85em">{spec.role_description}</span>',
            unsafe_allow_html=True,
        )

# ── Step 3: Spawn agents ──────────────────────────────────────────────────────

chat_status.info("⏳ Spawning agents...")

factory = AutoGenFactory(provider="openai")
manager = AgentManager(factory)

agents = []
for spec in specs:
    agent = run_async(manager.spawn(
        role=spec.role_description,
        name=spec.name,
        system_message=spec.system_prompt,
    ))
    agents.append(agent)

# ── Step 4: Run conversation → Tab 3 ─────────────────────────────────────────

chat_status.info("⏳ Agents in conversation...")

conversation = AgentConversation(
    manager=manager,
    agents=agents,
    orchestrator=orchestrator,
    max_rounds=int(max_rounds),
)

conv_messages: list[ConversationMessage] = []
stop_signal: StopSignal | None = None

for item in iter_async_stream(lambda: conversation.run_stream(goal)):
    if isinstance(item, ConversationMessage):
        conv_messages.append(item)
        conv_display.markdown(
            render_conversation(conv_messages, specs),
            unsafe_allow_html=True,
        )
    elif isinstance(item, StopSignal):
        stop_signal = item
        conv_display.markdown(
            render_conversation(conv_messages, specs, stop_signal),
            unsafe_allow_html=True,
        )

# ── Step 5: Synthesize → Tab 1 ────────────────────────────────────────────────

chat_status.info("⏳ Synthesizing final report...")

synthesis_text = ""
context_text = conversation.to_context_text()

for chunk in iter_async_stream(lambda: orchestrator.synthesize_stream(goal, context_text)):
    synthesis_text += chunk
    final_report.markdown(synthesis_text + "▌")

final_report.markdown(synthesis_text)
chat_status.success("✅ Done")

# ── Step 6: Shutdown ──────────────────────────────────────────────────────────

run_async(manager.shutdown())
