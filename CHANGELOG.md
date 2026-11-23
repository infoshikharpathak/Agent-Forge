# Changelog

## 0.1.0 (unreleased)

First release on PyPI, as `garrison-forge` (import `agent_forge`, CLI
`agent-forge`), one of the Garrison family of packages.

- **License:** Apache-2.0 (was MIT).

- **Orchestration:** a planner (Orchestrator) that builds a team for a goal,
  an AgentManager over pluggable factories, and a graph runner with
  conditional routing, failure handling and quality guardrails (goal
  clarity, plan validation, quality and grounding checks).
- **Backends:** OpenAI (core), Anthropic, AutoGen and LangGraph. The last
  three are extras: `pip install "garrison-forge[anthropic]"`, `[autogen]`,
  `[langgraph]`, or `[all]`.
- **Tools:** a process-wide tool registry, tool sources, the built-in tools
  (`[tools]`) and MCP servers (`[mcp]`).
- **Hooks and RunContext:** hooks around agent runs and tool calls, and a
  per-run context with `trace_id` and `app`.
- **App Registry:** fixed agent workflows for recurring tasks, loaded from
  `AGENT_FORGE_APPS_DIR` (new in this release) or the source tree's apps/.
- **Traceability:** a local SQLite run store with tokens and cost per agent
  (`agent_forge.pricing`), and `agent-forge traces` to query it.
- **HTTP API** (`[api]`): `/run`, `/run/stream` and `/health`.

Packaging for this release: Streamlit, AutoGen, LangGraph, the built-in
tools' libraries and MCP moved from required dependencies to extras, and the
`agent-forge run` demo loads AutoGen only when it runs.

Agent Forge runs the agents of
[Garrison](https://github.com/infoshikharpathak/garrison) (renamed from
Bastion before its first release).
