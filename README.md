# agent-forge

[![CI](https://github.com/infoshikharpathak/Agent-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/infoshikharpathak/Agent-Forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

**Framework-agnostic AI agent orchestration.**

Send a goal, get back a synthesized report — produced by a dynamically planned team of AI agents that research, debate, and converge on an answer. No agents are predefined by default; the orchestrator writes every system prompt and task prompt from scratch for each goal.

Two additive capabilities sit on top of that dynamic core:

- **App Registry** — apps (e.g. [logscribe](../logscribe), [meeting-scribe](../meeting-scribe)) can register a fixed agent workflow in a config file. When a goal matches, agent-forge skips dynamic planning entirely (or partially) instead of inventing a team from scratch every time — cheaper, faster, and repeatable for the same kind of task.
- **Traceability** — every run, whichever path it took, is persisted to a local SQLite database: which agents ran, what each one produced, tokens/cost/latency, which guardrails fired. Queryable via CLI, API, or the Streamlit UI's Traces tab.

Zero registered apps and no queries against `/traces` is exactly today's behavior — both are fully additive.

---

## How it works

```
User goal
   │
   ▼
[App Registry routing]        ← does the goal semantically match a registered app?
   │                              locked  → skip straight to Execute, no LLM planning call
   │                              unlocked→ fixed agents, framework/prompts still LLM-chosen
   │                              none    → dynamic (current behavior, unchanged)
   │                              emits app_routed (routing_tier, app_id)
   │
   ▼
[Guardrail 1: Goal Clarity]   ← rewrites vague goals into specific, actionable ones
   │                              (skippable per app config)
   │                              emits goal_clarified (was_changed + reasoning)
   │
   ▼
Orchestrator scores goal      ← reasoning_score vs workflow_score (0-10 each)
   │                              rule-based override if confidence < 0.7
   │                              emits plan_ready with scores + confidence
   │                              emits goal_clarified (was_changed + reasoning)
   │
   ▼
Orchestrator research         ← get_datetime (always) + web_search as needed
   │                              runs ONCE — not repeated on plan retries
   │
   ▼
[Guardrail 2: Plan Validator] ← up to 3 attempts
   │  checks: coverage, role distinctness, prompt specificity, strategy fit
   │  rejects bad plans with feedback → orchestrator replans
   │  emits plan_validation per attempt
   │
   ├─── strategy: autogen ────────────────────────────────────────────────────┐
   │                                                                           │
   │    AgentManager + AutoGenFactory                                          │
   │      └── spawn N debate agents (each with a tailored system prompt)      │
   │                                                                           │
   │    AgentConversation   ← multi-round debate loop                          │
   │      agents see full conversation history                                 │
   │      orchestrator judges convergence after each round                     │
   │      max_rounds safety cap prevents runaway loops                         │
   │                                                                           │
   └─── strategy: langgraph ──────────────────────────────────────────────────┤
                                                                               │
        LangGraphFactory                                                        │
          └── spawn one agent per graph node                                   │
                                                                               │
        GraphRunner   ← builds a LangGraph StateGraph from the spec            │
                                                                               │
          mode A — parallel independent (edges: []):                           │
            each node receives only its own task_prompt                        │
            nodes run in isolation, no shared context                          │
                                                                               │
          mode B — sequential / conditional pipeline (edges defined):          │
            nodes execute in order defined by edges                            │
            each node receives the goal + all upstream output as context       │
            conditional edges: node emits {"route": "KEY"} JSON               │
            GraphRunner validates key against known edges, routes accordingly  │
            routing JSON stripped from content before display / synthesis      │
            retry on invalid output — fallback placeholder after 1 failed retry│
                                                                               │
   ┌───────────────────────────────────────────────────────────────────────────┘
   │
   ▼
Orchestrator.synthesize       ← reads full execution output, writes final report
   │
   ▼
[Guardrail 3: Quality Check]  ← up to 1 retry
   │  checks: does the report directly answer the goal?
   │  synthesis buffered internally — only streamed after it passes
   │  re-synthesizes with feedback if it fails
   │  emits quality_check per attempt
   │
   ▼
[Guardrail 4: Grounding Check] ← verifies every claim is backed by agent conversation
   │  flags unsupported claims / hallucinations
   │  emits grounding_check (grounded + unsupported_claims list)
   │
   ▼
FastAPI (SSE stream)  ← every event streamed in real time
   │                    detail=result | orchestration | full
   ▼
Streamlit UI          ← 5 tabs: Chat · Orchestrator · Agent Activity · Quality Guardrails · Traces
   │
   ▼
RunTracer → SQLite     ← every run persisted regardless of outcome (Traceability, below)
```

---

## Architecture

```
src/agent_forge/
├── core/
│   ├── agent.py          # BaseAgent ABC + AgentStatus enum + AgentRunResult
│   ├── factory.py        # AgentFactory ABC (create / close)
│   ├── manager.py        # AgentManager — lifecycle + task routing + trace recording
│   ├── orchestrator.py   # Orchestrator — plan, judge convergence, synthesize, match/plan_from_app
│   ├── conversation.py   # AgentConversation — multi-round AutoGen debate loop
│   ├── graph_runner.py   # GraphRunner — LangGraph structured pipeline runner
│   └── shared_thread.py  # SharedThread — sequential context passing
├── backends/
│   ├── autogen/          # ✅ AutoGen agentchat implementation
│   ├── langgraph/        # ✅ LangGraph + LangChain implementation
│   └── anthropic/        # 🔧 stub
├── tools/
│   ├── __init__.py       # Registry — @register, get_tools(), list_tools()
│   ├── web.py            # web_search (DuckDuckGo), fetch_url
│   ├── finance.py        # stock_price, company_financials (yfinance)
│   ├── utility.py        # get_datetime, calculator, wikipedia_search
│   └── mcp_bridge.py     # MCPBridge — connects MCP servers, registers tools (with per-server tool allowlist)
├── apps/                 # App Registry configs — one JSON/YAML file per registered app
├── api/
│   └── app.py            # FastAPI backend — /health, /run, /run/stream, /apps, /traces
├── config/
│   └── settings.py       # Provider config + env resolution
├── registry.py            # AppRegistry, AppConfig — loads apps/, builds locked-app plans (no LLM)
├── tracing.py              # RunTracer — per-run trace accumulation, cost estimation
├── db.py                   # SQLite persistence for traces (append-only)
└── main.py                # CLI — `agent-forge run` (demo) / `agent-forge traces`

app.py                    # Streamlit frontend — 5 tabs: Chat · Orchestrator · Agent Activity · Quality Guardrails · Traces
mcp_servers.json          # MCP server config (gitignored, create from example)
mcp_servers.example.json  # Example MCP server config
agent_forge_traces.db     # SQLite trace store (gitignored, zero-config default path)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# add your OPENAI_API_KEY to .env
```

---

## Running

**Backend** (required first):

```bash
uvicorn agent_forge.api.app:app --reload --port 8000
```

**Frontend** (in a second terminal):

```bash
streamlit run app.py
```

**CLI** (`pip install -e .` registers the `agent-forge` console script):

```bash
agent-forge run                          # hardcoded demo goal, autogen only
agent-forge traces --last 10             # query run history
agent-forge traces --app logscribe --status failed
```

---

## API

Interactive docs available at `http://localhost:8000/docs` once the backend is running.

### `GET /health`

Liveness probe.

```json
{"status": "ok"}
```

### `POST /run`

Blocking. Returns only the final synthesized report.

```json
// Request
{"goal": "...", "max_rounds": 3, "provider": "openai"}

// Response
{"result": "..."}
```

### `POST /run/stream`

SSE stream. The `detail` query parameter controls how much of the internal pipeline is exposed:

| `detail` | Events included |
|---|---|
| `result` (default) | `synthesis_chunk`, `done`, `error` |
| `orchestration` | + `orchestrator_tool_call`, `plan_chunk`, `plan_ready`, `goal_clarified`, `plan_validation`, `quality_check`, `grounding_check` |
| `full` | + `agent_message`, `stop_signal` |

Every SSE frame is a `data:` line containing a JSON object with a `type` field:

```
data: {"type": "goal_clarified", "original": "...", "clarified": "...", "reasoning": "...", "was_changed": true}
data: {"type": "orchestrator_tool_call", "tool": "web_search", "args": {...}, "result": "..."}
data: {"type": "plan_chunk", "text": "..."}
data: {"type": "plan_ready", "strategy": "autogen", "specs": [...]}
data: {"type": "plan_ready", "strategy": "langgraph", "spec": {"nodes": [...], "edges": [...], "entry": "..."}}
data: {"type": "plan_validation", "valid": true, "feedback": "", "attempt": 1}
data: {"type": "agent_message", "agent": "analyst", "content": "...", "round": 1}
data: {"type": "stop_signal", "reason": "...", "stopped_by": "orchestrator"}
data: {"type": "synthesis_chunk", "text": "..."}
data: {"type": "quality_check", "passes": true, "feedback": "", "attempt": 1}
data: {"type": "grounding_check", "grounded": true, "unsupported_claims": []}
data: {"type": "done", "result": "..."}
data: {"type": "error", "message": "..."}
```

---

## App Registry

Apps can register a fixed agent workflow instead of relying on dynamic planning every time — useful when the same kind of task recurs (log root-cause analysis, meeting minutes, etc.) and inventing a new team per request is unnecessary cost/latency.

Drop a config file in `src/agent_forge/apps/`:

```json
{
  "app_id": "logscribe",
  "name": "LogScribe — Production Error Root-Cause Analysis",
  "triggers": ["log error analysis", "root cause analysis of a production error"],
  "description": "...",
  "workflow": [
    {
      "agent": "root_cause_analyst",
      "role": "You are a senior SRE performing root-cause analysis...",
      "tools": ["mcp_repo_git_log", "mcp_repo_git_show"]
    }
  ],
  "framework": "langgraph",
  "framework_locked": true,
  "models": {"root_cause_analyst": "gpt-5.4-mini"},
  "guardrails": {"skip_goal_clarity": true, "skip_plan_validation": true, "hallucination_check": true}
}
```

No code changes needed — restart the backend (or call the registry's `register()` programmatically) and it's live.

### Routing tiers

| Tier | Condition | Behavior |
|---|---|---|
| **Locked** | matched app, `framework_locked: true` | Plan built deterministically from the config — zero LLM planning calls. Agent system prompts come from a fixed template, not orchestrator-written prose. |
| **Unlocked** | matched app, `framework_locked: false` | Agent names/roles are fixed by the config; one LLM call still picks the framework (autogen/langgraph) and writes prompts. |
| **Dynamic** | no app matches | Today's behavior, byte-for-byte unchanged. |

Matching is semantic (an LLM call comparing the goal against each registered app's `triggers`/`description`), not keyword-based — and is skipped entirely when zero apps are registered, so an unmodified install pays no extra cost or latency.

Per-app `guardrails` flags (`skip_goal_clarity`, `skip_plan_validation`, `hallucination_check`) let each app opt in/out of the standard guardrail pipeline — e.g. skip goal clarity for an app whose goal is always already fully-specified, but keep grounding checks on for anything where a fabricated claim matters (root-cause analysis, meeting minutes).

---

## Traceability

Every run — locked, unlocked, or dynamic, successful or failed — is persisted to a local SQLite database (`agent_forge_traces.db`, zero config, path overridable via `AGENT_FORGE_DB_PATH`). Traces are append-only.

Each trace records: routing tier + matched app, framework used, every spawned agent (model, tokens, latency, status, **and its full output**), which guardrails actually fired, the final synthesized report, total cost/latency/tokens, and outcome/error.

Query via CLI, API, or the Streamlit UI's **Traces** tab (filter by app/status/cost, drill into any run to see the goal, the final report, and each agent's full output):

```bash
agent-forge traces --last 10
agent-forge traces --app logscribe --status failed
```

```
GET /traces?limit=10&app_id=logscribe&status=failed&cost_above=0.01
GET /traces/{run_id}
```

**Scope note:** v1 tokenizes spawned-agent calls only — the orchestrator's own planning/guardrail LLM calls aren't yet counted, so `total_cost_usd`/`total_tokens` understate true spend for dynamic (non-locked) runs. A known, disclosed gap, not silently rounded into the totals.

---

## MCP servers

agent-forge supports [Model Context Protocol](https://modelcontextprotocol.io) servers as a first-class tool source. Any MCP server's tools are automatically discovered at startup, registered in the tool registry, and made available for the orchestrator to assign to agents — no code changes needed.

### Setup

Copy the example config and edit it:

```bash
cp mcp_servers.example.json mcp_servers.json
```

`mcp_servers.json` (project root):

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
  },
  {
    "name": "repo",
    "command": "uvx",
    "args": ["mcp-server-git", "--repository", "/path/to/monitored/repo"],
    "tools": ["git_log", "git_show", "git_diff", "git_status"]
  },
  {
    "name": "myserver",
    "url": "http://localhost:3000/sse"
  }
]
```

Each server's tools are registered as `mcp_{name}_{tool_name}` (e.g. `mcp_filesystem_read_file`, `mcp_repo_git_log`). The orchestrator sees them in its available tools list and assigns them to agents that need them; a registered app's config can also name them explicitly in a workflow step's `tools` list (see App Registry, above).

**`tools` allowlist (optional, recommended for anything with write capability):** without it, *every* tool a server advertises gets registered — including mutating ones like a git server's `git_commit`/`git_checkout` or a filesystem server's `write_file`/`edit_file`. Set `tools` to the exact subset you want available; anything else the server offers is never even registered, for any agent, locked or dynamic. The `repo` example above only exposes read-only git operations.

### Adding a new server

1. Add an entry to `mcp_servers.json`
2. Restart the backend

That's it — no Python code to write.

### Transports

| Field | Transport | Example |
|---|---|---|
| `command` + `args` | stdio (subprocess) | `"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]` |
| `url` | HTTP / SSE | `"url": "http://localhost:3000/sse"` |

---

## Tool library

Tools are auto-discovered via a `@register` decorator. The orchestrator always calls `get_datetime` before planning so agent prompts never hardcode a year. Agents receive whichever tools the orchestrator assigns per goal.

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo search |
| `fetch_url` | HTTP page fetch |
| `stock_price` | Live price via yfinance |
| `company_financials` | Key financial metrics via yfinance |
| `calculator` | Safe AST-based expression evaluator |
| `get_datetime` | Current date/time |
| `wikipedia_search` | Wikipedia article summary |

Adding a tool:

```python
# src/agent_forge/tools/my_tools.py
from agent_forge.tools import register

@register("my_tool", description="Does something useful.")
def my_tool(query: str) -> str:
    ...
```

Import the module once in `tools/__init__.py` and it is available to all agents.

---

## Execution strategies

The orchestrator automatically picks the right strategy per goal — no configuration needed.

### AutoGen — multi-round debate

Used for open-ended, opinion, or analytical goals where multiple perspectives improve the answer (e.g. *"Should I buy NVDA or TSLA?"*).

- Orchestrator plans a team of agents with complementary roles
- Each agent gets a tailored system prompt written for the specific goal
- Agents debate in rounds, each seeing the full conversation history
- Orchestrator judges convergence after each round; stops when agents agree or `max_rounds` is hit
- Orchestrator synthesizes the debate into a final report

### LangGraph — structured pipeline

Used when the goal has a clear deterministic structure. Two valid cases:

**Parallel independent** (no edges): Multiple agents each handle a completely separate sub-task with no need to see each other's output (e.g. analyse 5 companies simultaneously). Each node receives only its own `task_prompt`.

**Sequential / conditional pipeline** (edges defined): Stages build on each other, or a node's output determines which path to take next.

- Orchestrator defines a graph: nodes (agents with system + task prompts) and directed edges
- Each node receives the goal plus all work completed by upstream nodes as context
- **Conditional edges**: a node signals its route by ending its response with `[ROUTE: KEY]`. GraphRunner extracts the key, routes to the matching next node, and strips the tag before display or synthesis. If no valid tag is found, falls back to the unconditional edge (or END).

The SSE event stream, Streamlit UI, and synthesis phase are identical for both strategies.

---

## Strategy selection

The orchestrator scores every goal on two dimensions before choosing a strategy:

| Score | What it measures |
|---|---|
| `reasoning_score` (0-10) | How much agents benefit from debating and challenging each other |
| `workflow_score` (0-10) | How clearly the goal maps to distinct, ordered stages or parallel sub-tasks |

The strategy with the higher score wins. If scores are within 2 points, autogen is preferred. When LLM confidence is below 0.7, a keyword-based override kicks in — goals containing words like *extract*, *pipeline*, *sequential* bias toward langgraph; words like *debate*, *compare*, *recommend* bias toward autogen.

The `plan_ready` SSE event includes the scores and confidence:

```json
{
  "strategy": "autogen",
  "reasoning_score": 8,
  "workflow_score": 3,
  "multi_agent_needed": true,
  "confidence": 0.85
}
```

---

## Conditional routing

LangGraph nodes with conditional outgoing edges signal their route via a JSON object on its own line at the end of their response:

```json
{"route": "HIGH_RISK"}
```

GraphRunner extracts the key, validates it against the node's known outgoing edges, and routes accordingly. If the key is missing or invalid, the unconditional edge (or END) is used as fallback. The routing JSON is stripped from content before display or synthesis.

This replaces the previous `[ROUTE: KEY]` tag format.

---

## Failure handling

Every agent and graph node is wrapped with a single retry on invalid output:

1. Agent runs and produces output
2. If output is empty or below the minimum length threshold → retry once
3. If retry also fails → a safe fallback placeholder is used so the pipeline never silently propagates bad output

Thresholds: 50 characters for graph nodes, 20 characters for debate agents.

---

## Quality guardrails

Four LLM-based guardrails run automatically on every request — no configuration needed.

### 1. Goal Clarity Pre-check

Runs before research. Rewrites vague goals (e.g. *"Tesla"*) into specific, actionable ones before any agents are planned. If the goal is already specific, it passes through unchanged.

- `was_changed: true` → clarified goal is used for all downstream steps
- `was_changed: false` → original goal used as-is
- Emits: `goal_clarified`

### 2. Plan Validator

After the orchestrator produces a plan, the validator reviews it on four criteria: coverage (do agents address all aspects of the goal?), role distinctness, prompt specificity, and strategy fit (autogen vs langgraph).

- If the plan fails, feedback is injected and the orchestrator replans — up to **3 attempts**
- Research runs only once; only plan generation repeats on retry
- Emits: `plan_validation` per attempt

### 3. Quality Check

After synthesis, checks whether the final report actually answers the goal. Synthesis is buffered internally — it is only streamed to the client once it passes (or retries are exhausted), preventing the UI from showing a failing report that gets replaced.

- If the report fails, feedback is injected and synthesis is retried — **1 retry**
- Emits: `quality_check` per attempt

### 4. Grounding Check

After the final synthesis, verifies that every significant claim in the report is backed by something in the agent conversation. Flags hallucinations — facts the synthesizer added that the agents never mentioned.

- Does not trigger a retry — surfaced as a transparency signal
- Emits: `grounding_check` with `grounded` bool and `unsupported_claims` list

All four guardrails have defensive JSON parse fallbacks — a single LLM hiccup cannot abort the pipeline.

---

## Per-agent model selection

The orchestrator can assign a different model to each agent or graph node based on the complexity of its role. This is automatic — no configuration needed.

**Model tiers (OpenAI):**

| Model | When used |
|---|---|
| `gpt-5.4-mini` | Default for all agents and the orchestrator |
| `gpt-5.4-nano` | Simple, well-scoped tasks — extraction, formatting, summarization, classification |

The orchestrator specifies the model in its plan JSON:

```json
{
  "name": "action_extractor",
  "role_description": "Extracts action items from the transcript",
  "system_prompt": "...",
  "task_prompt": "...",
  "tools": [],
  "model": "gpt-5.4-nano"
}
```

`null` (or omitting the field) uses the provider default. Model guidance is provider-specific — the orchestrator only sees model names valid for the active provider.

**Orchestrator model tier:**

The orchestrator itself always uses a separate model config (`Settings.for_orchestrator()`), keeping it decoupled from agent model defaults. This ensures the orchestrator — which handles planning, convergence, synthesis, and all 4 guardrails — is never inadvertently downgraded when cheap agent models are assigned.

---

## Data-bearing goals

When a goal contains raw data (e.g. a transcript, document, or dataset), agent-forge preserves the original payload throughout execution:

- The **goal clarity guardrail** may rewrite a vague goal description for better planning — but the **original goal** (with full data) is always passed to agents during execution.
- Both AutoGen debate agents and LangGraph nodes receive the original goal prepended to their task, ensuring they have access to the complete payload without needing to ask for it.

---

## Adding a backend

1. Copy the structure of `src/agent_forge/backends/autogen/`
2. Implement `BaseAgent` (`run`, `close`) and `AgentFactory` (`create`)
3. Add the dependency to `pyproject.toml` and re-install

The manager, orchestrator, conversation loop, graph runner, and API layer are all backend-agnostic — only the factory changes.

---

## Planned

- Anthropic backend implementation
- Additional tools (code executor, news API, etc.)
- Context compression for long agent conversations
- Authentication / multi-user support
- Tokenize the orchestrator's own planning/guardrail calls into trace totals (currently only spawned agents are counted)
- Dashboard/UI for aggregate trace analytics (currently per-run only)
- Auto-promotion between App Registry tiers (locked ↔ unlocked) based on trace data

---

## Roadmap

### Near-term

**Model performance registry**
Automated evaluation pipeline that scores models per task type on quality, accuracy, hallucination rate, cost, and latency — using a frontier model as judge. Results are stored in a SQL registry and surfaced to the orchestrator for intelligent model routing decisions at plan time. *Traceability (shipped) provides the raw data layer this would build on — per-run cost/latency/tokens/guardrail outcomes are already captured; this item is the analysis/scoring layer on top.*

**Persistent orchestrator memory**
Captures guardrail outcomes (plan validation scores, quality check results, grounding flags) and user feedback per run. Used to improve plan quality and agent system prompts over time — the orchestrator learns which planning patterns work well for which goal types. *`guardrails_triggered` per run is already in the trace store; this item is about feeding it back into planning, not just recording it.*

---

### Medium-term

**RAG-based prompt storage and categorisation**
Stores prompts, agent outputs, and quality scores in a retrieval pipeline. The orchestrator can reference high-performing prompts from similar past goals when planning new runs, reducing trial-and-error in prompt generation.

**Usage pattern analysis**
Identifies high-frequency task categories from accumulated run data. Surfaces optimisation priorities — which task types run most often, cost the most, or produce the lowest quality scores — to guide infrastructure and fine-tuning decisions.

**Multi-model routing**
Assigns models to agent roles based on registry performance data rather than static defaults. Example: frontier model for synthesis and convergence judgment, mini for research and debate, nano for extraction and formatting — dynamically adjusted as registry data matures.

---

### Long-term vision

**Fine-tuned SLMs for high-frequency task types**
Uses accumulated usage data and categorised prompt/output pairs to train small, hyper-specific models. A fine-tuned SLM for a repeated task type (e.g. meeting MOM extraction, earnings report summarisation) can outperform general-purpose mini/nano models at a fraction of the cost.

**Three-tier execution strategy**

| Tier | Models | Latency | Cost | Best for |
|---|---|---|---|---|
| Real-time | Frontier | Instant | High | Complex reasoning, synthesis, judgment |
| Near-time | Mini / Nano | Seconds | Low | Research, extraction, formatting |
| Lazy | Fine-tuned SLMs | Minutes | Lowest | High-frequency, repeated task patterns |

**Self-optimising orchestration**
The system passively improves over time: usage data feeds back into prompt quality scoring, model selection refines via the performance registry, and accumulated outputs eventually seed custom model training. No manual intervention required — the orchestrator gets better the more it is used.
