from __future__ import annotations

"""
LangGraph-based execution runner for agent-forge.

Used when the Orchestrator chooses the ``langgraph`` strategy — i.e. when the
goal is best solved by a structured pipeline of sequential or parallel stages
rather than a multi-agent debate.

GraphRunner builds a LangGraph ``StateGraph`` from a ``GraphSpec``, runs it,
and yields the same ``ConversationMessage`` / ``StopSignal`` events as
``AgentConversation`` so the API and Streamlit layers need no changes.
"""

import operator
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, TypedDict

from agent_forge.core.conversation import ConversationMessage, StopSignal

if TYPE_CHECKING:
    from agent_forge.core.factory import AgentFactory
    from agent_forge.core.orchestrator import GraphSpec


class _AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    goal: str
    messages: Annotated[list[dict], operator.add]


class GraphRunner:
    """
    Builds and runs a LangGraph ``StateGraph`` from a ``GraphSpec``.

    Each node in the spec becomes a graph node whose function calls the
    corresponding agent with the goal plus all work accumulated so far.
    Edges are added as specified; nodes with no outgoing edges are wired
    to ``END`` automatically.

    Args:
        factory: Agent factory used to create node agents.
        spec:    Graph specification produced by the Orchestrator.
    """

    def __init__(self, factory: AgentFactory, spec: GraphSpec) -> None:
        self._factory = factory
        self._spec = spec
        self._messages: list[dict] = []

    async def run_stream(
        self, goal: str
    ) -> AsyncGenerator[ConversationMessage | StopSignal, None]:
        """
        Execute the graph and stream events.

        Yields:
            ``ConversationMessage`` for each node's output.
            ``StopSignal`` once all nodes have completed.
        """
        from langgraph.graph import StateGraph, END

        workflow: StateGraph = StateGraph(_AgentState)

        # ── Spawn agents for every node ───────────────────────────────────────
        agents: dict[str, Any] = {}
        for node in self._spec.nodes:
            agent = await self._factory.create(
                role=node.role_description,
                name=node.name,
                system_message=node.system_prompt,
                tools=node.tools or None,
            )
            agents[node.name] = agent

        # ── Build node functions ──────────────────────────────────────────────
        node_task_map = {n.name: n.task_prompt for n in self._spec.nodes}

        def _make_node_fn(agent: Any, node_name: str):
            async def node_fn(state: _AgentState) -> dict:
                task = node_task_map[node_name]
                context = "\n\n".join(
                    f"{m['agent']}:\n{m['content']}" for m in state["messages"]
                )
                if context:
                    task += f"\n\nWork completed so far:\n{context}"
                content = await agent.run(task)
                return {"messages": [{"agent": node_name, "content": content}]}
            return node_fn

        for node in self._spec.nodes:
            workflow.add_node(node.name, _make_node_fn(agents[node.name], node.name))

        if not self._spec.edges:
            # ── Parallel independent nodes ────────────────────────────────────
            # No edges means agents are fully independent — each receives its
            # own task_prompt with no shared context from other nodes.
            node_map = {n.name: n for n in self._spec.nodes}
            for node in self._spec.nodes:
                task = node_map[node.name].task_prompt
                content = await agents[node.name].run(task)
                msg = {"agent": node.name, "content": content}
                self._messages.append(msg)
                yield ConversationMessage(agent=node.name, content=content, round=1)

        else:
            # ── Sequential pipeline via LangGraph ─────────────────────────────
            # Edges define the execution order; each node receives accumulated
            # context from upstream nodes.
            for edge in self._spec.edges:
                workflow.add_edge(edge.from_node, edge.to_node)

            # Wire terminal nodes (no outgoing edge) → END
            nodes_with_outgoing = {e.from_node for e in self._spec.edges}
            for node in self._spec.nodes:
                if node.name not in nodes_with_outgoing:
                    workflow.add_edge(node.name, END)

            workflow.set_entry_point(self._spec.entry)
            graph = workflow.compile()

            # stream_mode="updates" yields {node_name: node_output} per step,
            # not the full state snapshot — required for correct node extraction.
            async for event in graph.astream(
                {"goal": goal, "messages": []}, stream_mode="updates"
            ):
                for node_name, state_update in event.items():
                    if node_name == "__end__":
                        continue
                    for msg in state_update.get("messages", []):
                        self._messages.append(msg)
                        yield ConversationMessage(
                            agent=msg["agent"],
                            content=msg["content"],
                            round=1,
                        )

        # Clean up agents
        for agent in agents.values():
            await agent.close()

        yield StopSignal(reason="Graph execution complete.", stopped_by="graph")

    def to_context_text(self) -> str:
        """Return accumulated node outputs as a single context string for synthesis."""
        return "\n\n---\n\n".join(
            f"{m['agent']}:\n{m['content']}" for m in self._messages
        )
