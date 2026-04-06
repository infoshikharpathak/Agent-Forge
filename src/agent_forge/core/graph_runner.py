from __future__ import annotations

"""
LangGraph-based execution runner for agent-forge.

Used when the Orchestrator chooses the ``langgraph`` strategy — i.e. when the
goal is best solved by a structured pipeline of sequential or parallel stages,
or a graph with conditional routing based on node output.

GraphRunner builds a LangGraph ``StateGraph`` from a ``GraphSpec``, runs it,
and yields the same ``ConversationMessage`` / ``StopSignal`` events as
``AgentConversation`` so the API and Streamlit layers need no changes.

Conditional routing:
    Nodes signal their route by including ``[ROUTE: KEY]`` at the end of their
    response.  GraphRunner extracts the key, matches it to the outgoing
    conditional edges, and routes accordingly.  The tag is stripped from the
    content before it is displayed or passed to the synthesis phase.
"""

import logging
import operator
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, TypedDict

log = logging.getLogger("uvicorn.error")

from agent_forge.core.conversation import ConversationMessage, StopSignal

if TYPE_CHECKING:
    from agent_forge.core.factory import AgentFactory
    from agent_forge.core.orchestrator import GraphSpec

# Matches [ROUTE: SOME_KEY] anywhere in a string (case-insensitive)
_ROUTE_TAG = re.compile(r"\[ROUTE:\s*(\w+)\]", re.IGNORECASE)
_ROUTE_STRIP = re.compile(r"\[ROUTE:\s*\w+\]", re.IGNORECASE)


def _extract_route(content: str) -> str | None:
    """Return the uppercase route key from a [ROUTE: KEY] tag, or None."""
    m = _ROUTE_TAG.search(content)
    return m.group(1).upper() if m else None


def _strip_route(content: str) -> str:
    """Remove [ROUTE: KEY] tags from content before display / synthesis."""
    return _ROUTE_STRIP.sub("", content).strip()


class _AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    goal: str
    messages: Annotated[list[dict], operator.add]


class GraphRunner:
    """
    Builds and runs a LangGraph ``StateGraph`` from a ``GraphSpec``.

    Execution modes
    ---------------
    **Parallel independent** (``spec.edges`` is empty):
        Each node receives only its own ``task_prompt`` — no shared context.
        Nodes run sequentially but in isolation (true fan-out requires the
        LangGraph ``Send`` API; direct calls keep outputs cleanly separated).

    **Sequential / conditional pipeline** (``spec.edges`` non-empty):
        Edges define execution order and optional conditional routing.
        Nodes with conditional outgoing edges embed a ``[ROUTE: KEY]`` tag
        in their output; the runner extracts it, routes to the matching node,
        and strips the tag before display.

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
            ``ConversationMessage`` for each node's output (route tags stripped).
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
                    f"{m['agent']}:\n{_strip_route(m['content'])}"
                    for m in state["messages"]
                )
                if context:
                    task += f"\n\nWork completed so far:\n{context}"
                # Keep raw content in state so the router can read [ROUTE:] tags
                content = await agent.run(task)
                return {"messages": [{"agent": node_name, "content": content}]}
            return node_fn

        for node in self._spec.nodes:
            workflow.add_node(node.name, _make_node_fn(agents[node.name], node.name))

        # ── No edges → parallel independent execution ─────────────────────────
        if not self._spec.edges:
            for node in self._spec.nodes:
                content = await agents[node.name].run(node.task_prompt)
                clean = _strip_route(content)
                self._messages.append({"agent": node.name, "content": clean})
                yield ConversationMessage(agent=node.name, content=clean, round=1)

        # ── Edges defined → LangGraph pipeline (sequential / conditional) ──────
        else:
            # Group outgoing edges by source node
            edges_by_source: dict[str, list] = defaultdict(list)
            for edge in self._spec.edges:
                edges_by_source[edge.from_node].append(edge)

            for from_node, outgoing in edges_by_source.items():
                conditional = [e for e in outgoing if e.condition_key]
                unconditional = [e for e in outgoing if not e.condition_key]

                if conditional:
                    # Build condition_key → to_node map
                    cond_map = {
                        e.condition_key.upper(): e.to_node
                        for e in conditional
                    }
                    # Fallback: first unconditional edge, or END
                    fallback = unconditional[0].to_node if unconditional else END

                    def _make_router(fn: str, cm: dict, fb: Any):
                        def router(state: _AgentState) -> Any:
                            node_msgs = [
                                m for m in state["messages"] if m["agent"] == fn
                            ]
                            if node_msgs:
                                key = _extract_route(node_msgs[-1]["content"])
                                log.info(
                                    "[router] node=%s extracted_key=%s map=%s",
                                    fn, key, list(cm.keys()),
                                )
                                if key and key in cm:
                                    log.info("[router] routing %s → %s", fn, cm[key])
                                    return cm[key]
                            log.warning(
                                "[router] no valid route tag from %s — falling back to %s",
                                fn, fb,
                            )
                            return fb
                        return router

                    workflow.add_conditional_edges(
                        from_node, _make_router(from_node, cond_map, fallback)
                    )
                else:
                    for edge in unconditional:
                        workflow.add_edge(edge.from_node, edge.to_node)

            # Wire terminal nodes (no outgoing edge) → END
            nodes_with_outgoing = set(edges_by_source.keys())
            for node in self._spec.nodes:
                if node.name not in nodes_with_outgoing:
                    workflow.add_edge(node.name, END)

            workflow.set_entry_point(self._spec.entry)
            graph = workflow.compile()

            # stream_mode="updates" yields {node_name: node_output} per step,
            # not the full accumulated state snapshot.
            async for event in graph.astream(
                {"goal": goal, "messages": []}, stream_mode="updates"
            ):
                for node_name, state_update in event.items():
                    if node_name == "__end__":
                        continue
                    for msg in state_update.get("messages", []):
                        clean = _strip_route(msg["content"])
                        self._messages.append({"agent": msg["agent"], "content": clean})
                        yield ConversationMessage(
                            agent=msg["agent"],
                            content=clean,
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
