# agent-forge

**Framework-agnostic AI agent orchestration.**

Send a goal, get back a synthesized report — produced by a dynamically planned team of AI agents that research, debate, and converge on an answer. No agents are predefined; the orchestrator writes every system prompt and task prompt from scratch for each goal.

---

## How it works

```
User goal
   │
   ▼
Orchestrator          ← researches the goal (get_datetime always, web_search as needed)
   │                    chooses execution strategy, writes agent system + task prompts
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
            conditional edges: node embeds [ROUTE: KEY] in response            │
            GraphRunner extracts key, routes to matching next node             │
            tag stripped from content before display / synthesis               │
                                                                               │
   ┌───────────────────────────────────────────────────────────────────────────┘
   │
   ▼
Orchestrator.synthesize  ← reads full execution output, writes final report
   │
   ▼
FastAPI (SSE stream)  ← every event streamed in real time
   │                    detail=result | orchestration | full
   ▼
Streamlit UI          ← 3 tabs: Chat · Orchestrator · Agent Activity
                         Orchestrator tab shows graph edges and conditions
```

---

## Architecture

```
src/agent_forge/
├── core/
│   ├── agent.py          # BaseAgent ABC + AgentStatus enum
│   ├── factory.py        # AgentFactory ABC (create / close)
│   ├── manager.py        # AgentManager — lifecycle + task routing
│   ├── orchestrator.py   # Orchestrator — plan, judge convergence, synthesize
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
│   └── mcp_bridge.py     # MCPBridge — connects MCP servers, registers tools
├── api/
│   └── app.py            # FastAPI backend — /health, /run, /run/stream
├── config/
│   └── settings.py       # Provider config + env resolution
└── main.py               # Minimal smoke-test entrypoint

app.py                    # Streamlit frontend (pure SSE consumer)
mcp_servers.json          # MCP server config (gitignored, create from example)
mcp_servers.example.json  # Example MCP server config
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
| `orchestration` | + `orchestrator_tool_call`, `plan_chunk`, `plan_ready` |
| `full` | + `agent_message`, `stop_signal` |

Every SSE frame is a `data:` line containing a JSON object with a `type` field:

```
data: {"type": "synthesis_chunk", "text": "..."}
data: {"type": "done", "result": "..."}
data: {"type": "orchestrator_tool_call", "tool": "web_search", "args": {...}, "result": "..."}
data: {"type": "plan_chunk", "text": "..."}
data: {"type": "plan_ready", "strategy": "autogen", "specs": [...]}
data: {"type": "plan_ready", "strategy": "langgraph", "spec": {"nodes": [...], "edges": [...], "entry": "..."}}
data: {"type": "agent_message", "agent": "analyst", "content": "...", "round": 1}
data: {"type": "stop_signal", "reason": "...", "stopped_by": "orchestrator"}
data: {"type": "error", "message": "..."}
```

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
    "name": "myserver",
    "url": "http://localhost:3000/sse"
  }
]
```

Each server's tools are registered as `mcp_{name}_{tool_name}` (e.g. `mcp_filesystem_read_file`). The orchestrator sees them in its available tools list and assigns them to agents that need them.

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

## Adding a backend

1. Copy the structure of `src/agent_forge/backends/autogen/`
2. Implement `BaseAgent` (`run`, `close`) and `AgentFactory` (`create`)
3. Add the dependency to `pyproject.toml` and re-install

The manager, orchestrator, conversation loop, graph runner, and API layer are all backend-agnostic — only the factory changes.

---

## Planned

- Anthropic backend implementation
- Additional tools (code executor, news API, etc.)
- Authentication / multi-user support
