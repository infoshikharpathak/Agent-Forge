from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict

from agent_forge.config.settings import ProviderConfig, Settings

if TYPE_CHECKING:
    from agent_forge.core.shared_thread import SharedThread


@dataclass
class OrchestratorToolCall:
    """Emitted during plan_stream when the orchestrator calls a tool."""
    tool: str
    args: dict
    result: str


class AgentSpec(BaseModel):
    """Specification for a single agent in an AutoGen debate team."""

    name: str
    role_description: str
    system_prompt: str
    tools: list[str] = []
    model: str | None = None


class GraphNode(BaseModel):
    """A single node in a LangGraph execution graph."""

    name: str
    role_description: str
    system_prompt: str
    task_prompt: str
    tools: list[str] = []
    model: str | None = None


class GraphEdge(BaseModel):
    """A directed edge between two nodes in a LangGraph execution graph.

    Set ``condition`` and ``condition_key`` to make the edge conditional.
    The source node's system_prompt must then instruct it to end its response
    with ``[ROUTE: <condition_key>]`` so the runner can extract the route.
    """

    model_config = ConfigDict(populate_by_name=True)
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    condition: str | None = None
    condition_key: str | None = None


class GraphSpec(BaseModel):
    """Full specification for a LangGraph execution graph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    entry: str


class Orchestrator:
    """
    Uses an LLM to plan what agents are needed for a given goal and
    writes detailed system prompts for each one.

    Usage:
        orchestrator = Orchestrator(provider="openai")
        specs = await orchestrator.plan("Analyse AAPL earnings and write a short report.")

        for spec in specs:
            agent = await manager.spawn(
                name=spec.name,
                role=spec.role_description,
                system_message=spec.system_prompt,
            )
    """

    _SYSTEM_BASE = """\
You are an agent orchestration planner. Given a goal, choose the best execution \
strategy and plan a team of AI agents to accomplish it.

STRATEGY SCORING — before choosing, score the goal on two dimensions (0-10):
- reasoning_score: how much benefit agents get from debating, challenging each other, \
and iterating on a shared answer. High for open-ended analysis, opinions, comparisons. \
Low for extraction, parsing, or tasks with a single correct answer.
- workflow_score: how clearly the goal maps to distinct, ordered stages or parallel \
independent sub-tasks. High when there are obvious steps or branches. Low for \
exploratory or conversational goals.

Choose the strategy with the higher score. If scores are close (within 2 points), \
prefer autogen. Also set multi_agent_needed (true if >1 agent adds value) and \
confidence (0-1, your certainty in the strategy choice).

STRATEGY CHOICE:
- autogen: Use when reasoning_score > workflow_score — agents debating and challenging \
each other improves the result. Open-ended analysis, opinions, comparisons, recommendations.
- langgraph: Use when workflow_score > reasoning_score. Two valid cases: \
(1) PARALLEL INDEPENDENT — multiple agents each handle a completely separate sub-task \
with no need to see each other's output; \
(2) CONDITIONAL PIPELINE — a node's output determines which path to take next. \
Do NOT use langgraph just because a task sounds sequential — if agents benefit from \
debating each other's findings, use autogen.

RETURN a JSON object — choose exactly one of these formats:

For autogen:
{{
  "strategy": "autogen",
  "reasoning_score": 8,
  "workflow_score": 3,
  "multi_agent_needed": true,
  "confidence": 0.85,
  "agents": [
    {{
      "name": "snake_case_name",
      "role_description": "one sentence",
      "system_prompt": "detailed, specific system prompt",
      "tools": ["tool_name"],
      "model": null
    }}
  ]
}}

For langgraph:
{{
  "strategy": "langgraph",
  "reasoning_score": 3,
  "workflow_score": 9,
  "multi_agent_needed": true,
  "confidence": 0.90,
  "nodes": [
    {{
      "name": "snake_case_name",
      "role_description": "one sentence",
      "system_prompt": "detailed persona, constraints, and domain knowledge for this node",
      "task_prompt": "the specific task this node must perform — tailored to its exact job, not the generic goal",
      "tools": ["tool_name"],
      "model": null
    }}
  ],
  "edges": [
    {{"from": "node_a", "to": "node_b"}},
    {{"from": "node_a", "to": "node_c", "condition": "if risk is HIGH", "condition_key": "HIGH_RISK"}},
    {{"from": "node_a", "to": "node_d", "condition": "if risk is LOW", "condition_key": "LOW_RISK"}}
  ],
  "entry": "first_node_name"
}}

Rules:
- For langgraph: if nodes are fully independent, leave "edges" as an empty array []. Each
  node will receive only its own task_prompt — no shared context. If stages must build
  on each other, define edges explicitly.
- Conditional edges: add "condition" (human-readable) and "condition_key" (short
  uppercase token e.g. "HIGH_RISK") to an edge to make it conditional. When a node
  has conditional outgoing edges, its system_prompt MUST include routing instructions:
  "End your response with a routing decision as a JSON object on its own line:
  {{\\"route\\": \\"HIGH_RISK\\"}} if ... or {{\\"route\\": \\"LOW_RISK\\"}} if ..."
  The route value must exactly match the condition_key of the chosen edge (uppercase).
  Include one unconditional edge from the same node as the fallback.
- CRITICAL: Agents are NOT interactive chatbots. Each agent receives all context in a \
single message and must produce its complete output immediately — it cannot ask follow-up \
questions or wait for more input. Write system_prompts and task_prompts that assume the \
full goal and any data (transcript, documents, etc.) are already present in the message. \
Never write prompts that say "please provide", "once you share", or "when given".
- tools: choose ONLY from the available tools listed below. Empty list if none needed.
- Only create agents/nodes that are genuinely necessary.
- IMPORTANT: use the current date from the research context when writing system_prompt
  and task_prompt — never hardcode a specific year. Reference "latest", "current", or
  the actual date from context instead.
- model: optionally assign a specific model to each agent/node based on the complexity \
of its role. Use null to inherit the provider default. Only use model names from the \
list below — do not use models from other providers.
{models_guidance}

Available tools:
{tools_list}"""

    _MODELS_GUIDANCE: dict[str, str] = {
        "openai": (
            '  - null (default "gpt-5.4-mini"): use for most agents — capable and well-rounded.\n'
            '  - "gpt-5.4-nano": use for simple, well-scoped tasks: extraction, formatting, '
            'summarization, classification where deep reasoning is not needed.'
        ),
        "anthropic": (
            '  - "claude-opus-4-6": most capable — reserve for the most demanding reasoning.\n'
            '  - null (default "claude-sonnet-4-6"): good balance — use for most agents.\n'
            '  - "claude-haiku-4-5-20251001": cheapest — use for simple tasks.'
        ),
        "gemini": (
            '  - null (default "gemini-2.5-flash"): use for all agents — only one tier available.'
        ),
    }

    def _build_system_prompt(self) -> str:
        from agent_forge.tools import list_tools
        tools_str = "\n".join(f"  - {t}" for t in list_tools())
        models_str = self._MODELS_GUIDANCE.get(self._config.provider, "  - null (use provider default)")
        return self._SYSTEM_BASE.format(tools_list=tools_str, models_guidance=models_str)

    @staticmethod
    def _apply_strategy_overrides(data: dict, goal: str) -> dict:
        """
        Rule-based strategy override applied when LLM confidence is low (< 0.7).
        Keyword signals in the goal are used to correct uncertain decisions.
        Does nothing when the LLM is confident.
        """
        if data.get("confidence", 1.0) >= 0.7:
            return data

        goal_lower = goal.lower()
        LG_KEYWORDS = frozenset({
            "extract", "parse", "pipeline", "sequential", "stages",
            "workflow", "structured", "step by step", "scrape", "generate report",
        })
        AG_KEYWORDS = frozenset({
            "debate", "compare", "opinion", "recommend", "should i",
            "better", "best", "pros and cons", "vs", "versus", "analyse",
        })
        lg_hits = sum(1 for kw in LG_KEYWORDS if kw in goal_lower)
        ag_hits = sum(1 for kw in AG_KEYWORDS if kw in goal_lower)

        if lg_hits > ag_hits and data.get("strategy") != "langgraph":
            data["strategy"] = "langgraph"
            data["override_reason"] = f"keyword override: langgraph signals={lg_hits}, autogen signals={ag_hits}"
        elif ag_hits > lg_hits and data.get("strategy") != "autogen":
            data["strategy"] = "autogen"
            data["override_reason"] = f"keyword override: autogen signals={ag_hits}, langgraph signals={lg_hits}"

        return data

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        self._config: ProviderConfig = Settings.for_orchestrator(provider, model=model)

    def _client(self) -> AsyncOpenAI:
        """Create a fresh client per call so it binds to the current event loop."""
        kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        return AsyncOpenAI(**kwargs)

    async def plan(self, goal: str) -> list[AgentSpec]:
        """
        Given a goal, return a list of AgentSpecs needed to accomplish it.
        The LLM decides how many agents are needed and writes their system prompts.
        """
        response = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return [AgentSpec(**spec) for spec in data["agents"]]

    _RESEARCH_SYSTEM = (
        "You are a research assistant helping an AI orchestration planner stay current. "
        "Use the available tools to gather a small amount of targeted, relevant context "
        "for the goal — current date, recent news, or key facts. "
        "Make 1-3 focused tool calls, then stop."
    )

    # Tools the orchestrator is allowed to use. Built-in tools only: tools
    # from a ToolSource (or MCP) are for agents, never the research loop.
    _ORCHESTRATOR_TOOLS = ["web_search", "get_datetime"]

    @classmethod
    def _research_tool(cls, name: str):
        """The built-in tool *name* if the research loop may call it, else None."""
        from agent_forge.tools import _REGISTRY, _ensure_loaded
        from agent_forge.tools.sources import is_source_tool

        if name not in cls._ORCHESTRATOR_TOOLS:
            return None
        _ensure_loaded()
        fn = _REGISTRY.get(name)
        if fn is None or is_source_tool(fn):
            return None
        return fn

    async def _research(self, goal: str) -> tuple[list[OrchestratorToolCall], str]:
        """
        Run a tool-calling loop to gather current context for the goal.
        Always calls get_datetime first so the planner always knows the exact
        current date — prevents agents from being written with hardcoded years.
        Returns (tool_calls_made, research_summary_text).
        """
        from agent_forge.tools import get_tool, get_openai_schemas

        # Always fetch the current date first — unconditionally
        get_datetime = get_tool("get_datetime")
        current_datetime = get_datetime()
        tool_calls_made: list[OrchestratorToolCall] = [
            OrchestratorToolCall(tool="get_datetime", args={}, result=current_datetime)
        ]

        tool_schemas = get_openai_schemas(
            [n for n in self._ORCHESTRATOR_TOOLS if self._research_tool(n) is not None]
        )
        messages: list[dict] = [
            {"role": "system", "content": self._RESEARCH_SYSTEM},
            {"role": "user", "content": f"Gather current context for: {goal}"},
        ]

        for _ in range(4):  # safety cap on rounds
            response = await self._client().chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                fn = self._research_tool(tc.function.name)
                args = json.loads(tc.function.arguments)
                import inspect as _inspect
                if fn is None:
                    result = f"Tool {tc.function.name!r} is not available to the orchestrator."
                elif _inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = fn(**args)
                tool_calls_made.append(
                    OrchestratorToolCall(tool=tc.function.name, args=args, result=result)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        research_text = "\n\n".join(
            f"[{tc.tool}({', '.join(f'{k}={v}' for k, v in tc.args.items())})]\n{tc.result}"
            for tc in tool_calls_made
        )
        return tool_calls_made, research_text

    async def _generate_plan_stream(
        self,
        goal: str,
        research_text: str,
        feedback: str | None = None,
    ):
        """
        Async generator — streams plan JSON chunks then yields the parsed spec.
        Accepts pre-computed research_text so research is not repeated on retries.
        Optional feedback from a prior validator run is appended to the prompt.

        Phase 1 — yields str chunks as the plan JSON streams.
        Phase 2 — yields list[AgentSpec] | GraphSpec as the final item.
        """
        user_content = f"Goal: {goal}"
        if research_text:
            user_content += f"\n\nCurrent context from research:\n{research_text}"
        if feedback:
            user_content += (
                f"\n\nPREVIOUS PLAN WAS REJECTED. Validator feedback:\n{feedback}\n\n"
                "Please fix the issues described above and generate a revised plan."
            )

        stream = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            stream=True,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_content},
            ],
        )
        chunks: list[str] = []
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                chunks.append(text)
                yield text

        data = self._apply_strategy_overrides(json.loads("".join(chunks)), goal)
        if data.get("strategy") == "langgraph":
            yield GraphSpec(
                nodes=[GraphNode(**n) for n in data["nodes"]],
                edges=[GraphEdge.model_validate(e) for e in data["edges"]],
                entry=data["entry"],
            )
        else:
            yield [AgentSpec(**spec) for spec in data.get("agents", [])]

    async def plan_stream(self, goal: str):
        """
        Async generator for live UI display.
        Phase 1 — yields OrchestratorToolCall as the orchestrator researches.
        Phase 2 — yields str chunks as the plan JSON streams.
        Phase 3 — yields list[AgentSpec] | GraphSpec as the final item.
        """
        tool_calls, research_text = await self._research(goal)
        for tc in tool_calls:
            yield tc
        async for item in self._generate_plan_stream(goal, research_text):
            yield item

    _GOAL_CLARITY_SYSTEM = (
        "You are a goal clarification assistant. Evaluate whether the user's goal is "
        "specific and actionable enough for a team of AI research agents to tackle.\n\n"
        "A goal is VAGUE if it is a broad topic, a one-word subject, or lacks a clear "
        "question or deliverable (e.g. 'Tesla', 'AI news', 'crypto').\n"
        "A goal is SPECIFIC if it asks a clear question or requests a defined output "
        "(e.g. 'Compare Tesla Q1 2025 earnings vs Q1 2024 and assess whether the stock "
        "is a buy today').\n\n"
        "Return JSON: {\"clarified_goal\": \"the goal to use — rewritten if vague, "
        "unchanged if already specific\", \"was_changed\": true|false, "
        "\"reasoning\": \"one sentence explaining what you changed and why, or why no "
        "change was needed\"}\n\n"
        "If the goal is already specific, set was_changed=false and return the original "
        "text verbatim. Do NOT over-engineer a precise goal — only rewrite when it is "
        "genuinely too vague to plan agents for."
    )

    _PLAN_VALIDATOR_SYSTEM = (
        "You are a plan quality reviewer for an AI agent orchestration system. "
        "You are given the user's goal and the orchestrator's proposed agent plan as JSON.\n\n"
        "Evaluate the plan on four criteria:\n"
        "1. COVERAGE — do the agents/nodes collectively address all major aspects of the goal?\n"
        "2. ROLE DISTINCTNESS — are agent roles meaningfully different, or is there overlap?\n"
        "3. PROMPT SPECIFICITY — are system_prompts and task_prompts detailed and precise, "
        "or vague placeholders?\n"
        "4. STRATEGY FIT — is the chosen strategy (autogen vs langgraph) appropriate?\n\n"
        "Return JSON: {\"valid\": true|false, \"feedback\": \"if valid=false: concise "
        "actionable feedback listing exactly what must be fixed. If valid=true: empty string.\"}\n\n"
        "Be strict but fair. Return valid=true if the plan is good enough to proceed, "
        "even if minor improvements exist. Only return valid=false when there is a clear, "
        "correctable problem that would materially harm output quality."
    )

    _QUALITY_CHECK_SYSTEM = (
        "You are a report quality reviewer. Given the original goal and a synthesized "
        "report, decide whether the report adequately answers the goal.\n\n"
        "A report PASSES if: it directly addresses the core question, reaches a concrete "
        "conclusion or recommendation, and has no large obvious gaps.\n"
        "A report FAILS if: it ignores a major part of the goal, is mostly generic "
        "background without addressing the specific question, or ends without a conclusion "
        "when the goal asks for one.\n\n"
        "Return JSON: {\"passes\": true|false, \"feedback\": \"if passes=false: specific "
        "description of what is missing or wrong. If passes=true: empty string.\"}"
    )

    _GROUNDING_CHECK_SYSTEM = (
        "You are a factual grounding verifier. Given the original goal, the full agent "
        "conversation (the evidence base), and the final synthesized report, identify any "
        "claims in the report that are NOT supported by the agent conversation.\n\n"
        "A claim is UNSUPPORTED if it states a specific fact, number, date, or conclusion "
        "not found anywhere in the agent conversation, or contradicts what the agents said. "
        "Minor paraphrasing and reasonable inferences from the conversation are fine.\n\n"
        "Return JSON: {\"grounded\": true|false, \"unsupported_claims\": "
        "[\"verbatim or near-verbatim claim from the report with no backing in the conversation\"]}\n\n"
        "Return grounded=true and unsupported_claims=[] if all claims are supported."
    )

    _SYNTHESIZE_SYSTEM = (
        "You are a senior analyst synthesizing the findings from multiple AI agents. "
        "Given the original goal and the full agent conversation, produce a single, clean, "
        "well-structured final response for the user. Eliminate redundancy, resolve any "
        "contradictions, and present the most important insights clearly and concisely.\n\n"
        "CRITICAL: preserve concrete, verifiable specifics from the conversation — exact "
        "identifiers (commit hashes, IDs, file paths, line numbers, tool names), numbers, "
        "names, and dates. These are load-bearing evidence, not filler — condensing prose "
        "is fine, but never generalize away a specific fact an agent already found in favor "
        "of vaguer phrasing."
    )

    _CONVERGENCE_SYSTEM = (
        "You are a convergence judge for a multi-agent debate. After each round you decide "
        "whether agents have genuinely converged or should keep debating.\n\n"
        "You receive the goal, structured positions from the CURRENT round, structured "
        "positions from the PREVIOUS round (empty if first check), and the full conversation.\n\n"
        "Evaluate THREE things:\n"
        "1. ALIGNMENT — are all agents' positions semantically the same or complementary? "
        "Flag if any agent holds a materially different stance.\n"
        "2. NOVELTY — did any agent introduce a meaningfully new fact, argument, or concern "
        "compared to the previous round? Repetition and paraphrasing do NOT count as new.\n"
        "3. CONFIDENCE — is the average confidence across agents >= 0.7?\n\n"
        "Converge ONLY IF: positions are aligned AND no major new information emerged "
        "AND average confidence >= 0.7.\n"
        "Continue IF: any agent disagrees, new substantive information emerged, or "
        "confidence is low.\n\n"
        "Return JSON: {\"converged\": true|false, \"reason\": \"one sentence explanation\", "
        "\"checks\": {\"aligned\": true|false, \"no_new_info\": true|false, "
        "\"confidence_sufficient\": true|false}}"
    )

    async def synthesize_stream(self, goal: str, context: SharedThread | str, feedback: str | None = None):
        """
        Async generator. Reads the full context (SharedThread or pre-formatted
        conversation text) and streams a clean final synthesis back to the user.
        """
        context_text = context if isinstance(context, str) else context.to_context()
        user_content = f"Goal: {goal}\n\n{context_text}"
        if feedback:
            user_content += (
                f"\n\nQUALITY CHECK FEEDBACK (previous synthesis was inadequate):\n{feedback}\n\n"
                "Please address the gaps described above in your revised synthesis."
            )
        stream = await self._client().chat.completions.create(
            model=self._config.model,
            stream=True,
            messages=[
                {"role": "system", "content": self._SYNTHESIZE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    async def _should_stop_simple(self, goal: str, history_text: str) -> tuple[bool, str]:
        """Fallback convergence check used when agents produce no position blocks."""
        response = await self._client().chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation moderator. Given the goal and the conversation "
                        "history between AI agents, decide if the agents have converged enough to stop.\n\n"
                        "Return JSON: {\"converged\": true/false, \"reason\": \"brief explanation\"}\n\n"
                        "Stop if: agents have reached agreement or complementary conclusions, the goal "
                        "is adequately addressed, or further rounds would add little value.\n"
                        "Continue if: there are unresolved disagreements worth exploring, important "
                        "aspects of the goal are uncovered, or agents have raised new questions."
                    ),
                },
                {"role": "user", "content": f"Goal: {goal}\n\n{history_text}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return data["converged"], data["reason"]

    async def convergence_check(
        self,
        goal: str,
        round_num: int,
        current_positions: list[dict],
        previous_positions: list[dict],
        history_text: str,
        min_rounds: int = 2,
    ) -> tuple[bool, str]:
        """
        Hybrid convergence check. Returns (should_stop, reason).

        Enforces a minimum round count, then evaluates three criteria:
        - Position alignment across agents
        - Novelty vs the previous round (repetition ≠ convergence)
        - Average confidence >= 0.7

        Falls back to simple prompt-based check if no position blocks were extracted.
        """
        if round_num < min_rounds:
            return False, f"Minimum {min_rounds} rounds not yet reached."

        if not current_positions:
            return await self._should_stop_simple(goal, history_text)

        user_content = (
            f"Goal: {goal}\n\n"
            f"Round: {round_num}\n\n"
            f"Current round positions:\n{json.dumps(current_positions, indent=2)}\n\n"
            f"Previous round positions:\n"
            f"{json.dumps(previous_positions, indent=2) if previous_positions else 'N/A (first check)'}\n\n"
            f"Full conversation:\n{history_text}"
        )
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._CONVERGENCE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            return data["converged"], data["reason"]
        except Exception:
            return await self._should_stop_simple(goal, history_text)

    # ── App registry ──────────────────────────────────────────────────────────

    _APP_MATCH_SYSTEM = (
        "You are an app-routing classifier for an AI agent orchestration system. "
        "Given a user's goal and a list of registered apps (each with triggers and a "
        "description), decide whether the goal semantically matches ONE of the apps "
        "closely enough that the app's pre-built workflow should handle it directly, "
        "instead of dynamically planning a new agent team.\n\n"
        "Match on MEANING, not just shared keywords — e.g. a goal about 'why is my "
        "service throwing 500s' should match an app whose trigger is 'error "
        "investigation' even with no literal word overlap. Only match when you are "
        "genuinely confident the app's workflow is what the goal calls for; when in "
        "doubt, return no match so the goal falls through to dynamic planning.\n\n"
        "Return JSON: {\"app_id\": \"<id of best match, or null if none fit>\", "
        "\"confidence\": <0.0-1.0>, \"reasoning\": \"one sentence\"}"
    )

    async def match_app(self, goal: str, apps: list[Any]) -> dict:
        """
        Semantically match a goal against registered apps (AppConfig instances).
        Returns {"app_id": str|None, "confidence": float, "reasoning": str}.
        Falls back to no match on any error — a failed match always falls
        through to dynamic planning, never blocks the pipeline.
        """
        if not apps:
            return {"app_id": None, "confidence": 0.0, "reasoning": "No apps registered."}

        apps_desc = "\n\n".join(
            f"- app_id: {a.app_id}\n  name: {a.name}\n  triggers: {a.triggers}\n  description: {a.description}"
            for a in apps
        )
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._APP_MATCH_SYSTEM},
                    {"role": "user", "content": f"Goal: {goal}\n\nRegistered apps:\n{apps_desc}"},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"app_id": None, "confidence": 0.0, "reasoning": "App match failed — falling back to dynamic."}

    _APP_CONSTRAINED_PLAN_SYSTEM = """\
You are an agent orchestration planner. A specific app has already defined which \
agents/nodes to use and their high-level roles — you must NOT invent, rename, remove, \
or add agents. Your only job is to:
1. Pick the best execution strategy: "autogen" (agents debate/discuss) or "langgraph" \
(structured pipeline). The app's suggested strategy is given below — override it only \
if clearly wrong for this specific goal.
2. Write a detailed system_prompt for each named agent/node.
3. For langgraph only: write a task_prompt per node, and edges connecting them \
(sequential unless a clear conditional need is described in a role).

The agent/node list must contain EXACTLY the agents given below, in the same order, \
using their exact given names — nothing added, removed, or renamed.

Fixed agents for this app:
{fixed_agents}

Suggested strategy (from the app config): {suggested_strategy}

RETURN a JSON object — choose exactly one of these formats (every field shown is \
required, including "role_description" on every agent/node):

For autogen:
{{
  "strategy": "autogen",
  "agents": [
    {{
      "name": "<exact fixed agent name>",
      "role_description": "one sentence",
      "system_prompt": "detailed, specific system prompt",
      "tools": [],
      "model": null
    }}
  ]
}}

For langgraph:
{{
  "strategy": "langgraph",
  "nodes": [
    {{
      "name": "<exact fixed agent name>",
      "role_description": "one sentence",
      "system_prompt": "detailed persona, constraints, and domain knowledge for this node",
      "task_prompt": "the specific task this node must perform",
      "tools": [],
      "model": null
    }}
  ],
  "edges": [
    {{"from": "<node_a>", "to": "<node_b>"}}
  ],
  "entry": "<first node name>"
}}

Rules:
- tools: always an empty list — this app does not use tools.
- model: use null to inherit the provider default.
- For langgraph: chain the fixed agents sequentially via edges (node 1 → node 2 → ...) \
unless a role clearly calls for conditional routing.
"""

    async def plan_from_app(
        self, app: Any, goal: str, research_text: str = ""
    ) -> tuple[GraphSpec | list[AgentSpec], str]:
        """
        Build an execution plan for a matched app (AppConfig).

        - framework_locked: deterministic, template-based, no LLM call at all.
        - not locked: one constrained LLM call — agent names/roles are fixed by
          the app; the LLM only picks the framework and writes prompts.

        Returns (plan, framework_used).
        """
        from agent_forge.registry import build_plan_from_app

        if app.framework_locked:
            return build_plan_from_app(app)

        fixed_agents = "\n".join(f"- {s.agent}: {s.role}" for s in app.workflow)
        user_content = f"Goal: {goal}"
        if research_text:
            user_content += f"\n\nCurrent context from research:\n{research_text}"

        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": self._APP_CONSTRAINED_PLAN_SYSTEM.format(
                            fixed_agents=fixed_agents, suggested_strategy=app.framework,
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
            )
            data = json.loads(response.choices[0].message.content)
        except Exception:
            # Safe fallback: behave like a locked app if the constrained call fails.
            return build_plan_from_app(app)

        if data.get("strategy") == "langgraph":
            plan: GraphSpec | list[AgentSpec] = GraphSpec(
                nodes=[GraphNode(**n) for n in data["nodes"]],
                edges=[GraphEdge.model_validate(e) for e in data["edges"]],
                entry=data["entry"],
            )
            return plan, "langgraph"
        plan = [AgentSpec(**s) for s in data.get("agents", [])]
        return plan, "autogen"

    # ── Quality guardrails ────────────────────────────────────────────────────

    async def clarify_goal(self, goal: str) -> dict:
        """
        Pre-flight check: is the goal specific enough to plan agents for?
        Returns {"clarified_goal": str, "was_changed": bool, "reasoning": str}.
        Falls back to the original goal if the LLM response cannot be parsed.
        """
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._GOAL_CLARITY_SYSTEM},
                    {"role": "user", "content": goal},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"clarified_goal": goal, "was_changed": False, "reasoning": "Clarity check failed — using original goal."}

    async def validate_plan(self, goal: str, plan_json: str) -> dict:
        """
        Review a generated plan JSON against the goal.
        Returns {"valid": bool, "feedback": str}.
        Falls back to valid=True on parse error to avoid blocking the pipeline.
        """
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._PLAN_VALIDATOR_SYSTEM},
                    {"role": "user", "content": f"Goal: {goal}\n\nProposed plan:\n{plan_json}"},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"valid": True, "feedback": ""}

    async def quality_check(self, goal: str, report: str) -> dict:
        """
        Check whether the synthesized report adequately answers the goal.
        Returns {"passes": bool, "feedback": str}.
        Falls back to passes=True on parse error.
        """
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._QUALITY_CHECK_SYSTEM},
                    {"role": "user", "content": f"Goal: {goal}\n\nReport:\n{report}"},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"passes": True, "feedback": ""}

    async def grounding_check(self, goal: str, conversation_text: str, report: str) -> dict:
        """
        Verify that claims in the final report are backed by the agent conversation.
        Returns {"grounded": bool, "unsupported_claims": list[str]}.
        Falls back to grounded=True on parse error.
        """
        try:
            response = await self._client().chat.completions.create(
                model=self._config.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._GROUNDING_CHECK_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Goal: {goal}\n\n"
                            f"Agent conversation (evidence base):\n{conversation_text}\n\n"
                            f"Final report to verify:\n{report}"
                        ),
                    },
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"grounded": True, "unsupported_claims": []}
