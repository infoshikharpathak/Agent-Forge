from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_forge.core.agent import AgentRunResult, BaseAgent
from agent_forge.core.execution import base_run_context, execute_agent
from agent_forge.core.factory import AgentFactory
from agent_forge.core.hooks import Hooks, run_hooks

if TYPE_CHECKING:
    from agent_forge.core.shared_thread import SharedThread
    from agent_forge.tracing import RunTracer

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Orchestrator that owns one factory and coordinates agent lifecycles.

    Usage:
        factory = AutoGenFactory(config)
        manager = AgentManager(factory)

        agent = await manager.spawn(role="research_analyst", name="alice")
        result = await manager.run_task(agent.agent_id, "Summarise Q1 earnings.")
        await manager.release(agent.agent_id)
        await manager.shutdown()
    """

    def __init__(
        self, factory: AgentFactory, tracer: RunTracer | None = None, hooks: Hooks | None = None
    ) -> None:
        self._factory = factory
        self._tracer = tracer
        self.hooks: list = list(hooks or [])

    @property
    def tracer(self) -> RunTracer | None:
        return self._tracer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def spawn(self, role: str, name: str, **kwargs: Any) -> BaseAgent:
        """Create a new agent via the factory and return it."""
        agent = await self._factory.create(role=role, name=name, **kwargs)
        logger.info("Spawned agent id=%s role=%s name=%s", agent.agent_id, role, name)
        return agent

    async def release(self, agent_id: str) -> None:
        """Kill a single agent by ID."""
        logger.info("Releasing agent id=%s", agent_id)
        await self._factory.kill(agent_id)

    async def release_all(self) -> None:
        """Kill every active agent."""
        logger.info("Releasing all agents (%d active)", len(self._factory))
        await self._factory.kill_all()

    async def shutdown(self) -> None:
        """Graceful shutdown: release all agents."""
        await self.release_all()
        logger.info("AgentManager shut down cleanly.")

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def run_task(
        self,
        agent_id: str,
        task: str,
        *,
        thread: SharedThread | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Route a task to a specific agent and return its response.

        If a SharedThread is provided, prior agent findings are injected as
        context before the task runs, and the result is appended to the thread.
        """
        return (await self.run_task_result(agent_id, task, thread=thread, **kwargs)).content

    async def run_task_result(
        self,
        agent_id: str,
        task: str,
        *,
        thread: SharedThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Like run_task, but returns the full AgentRunResult (including a
        hook "blocked"/"withheld" status).

        Handing the SharedThread's findings to this agent is a handoff from
        the thread's latest author: on_handoff hooks may replace that
        context, or deny it, in which case the agent is not run.
        """
        agent = self._factory.get(agent_id)
        logger.info("Running task on agent id=%s", agent_id)

        full_task = task
        if thread and not thread.is_empty():
            context = thread.to_context()
            if self.hooks:
                source = thread.messages()[-1].agent
                ctx = base_run_context(self._tracer).for_agent(source, None)
                outcome = await run_hooks(
                    self.hooks, "on_handoff", ctx, context,
                    subject=f"{source}->{agent.name}", tracer=self._tracer,
                    handoff=(source, agent.name),
                )
                if not outcome.allowed:
                    result = AgentRunResult(content=outcome.value, model="", status="blocked")
                    if self._tracer is not None:
                        self._tracer.record_agent(agent.name, result, 0.0, status="blocked")
                    return result
                context = outcome.value
            full_task = f"{task}\n\n{context}"

        result = await execute_agent(agent, full_task, agent.name, self._tracer, self.hooks, **kwargs)

        if thread is not None and result.status != "blocked":
            thread.add(agent.name, result.content)

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> BaseAgent:
        return self._factory.get(agent_id)

    def list_agents(self) -> list[BaseAgent]:
        return self._factory.list()

    def __repr__(self) -> str:
        agents = self._factory.list()
        return f"<AgentManager backend={self._factory.__class__.__name__} active={len(agents)}>"
