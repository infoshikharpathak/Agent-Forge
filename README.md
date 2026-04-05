# agent-forge

**Framework-agnostic AI agent orchestration.**

Send a goal, get back a synthesized report — produced by a dynamically planned team of AI agents that research, debate, and converge on an answer. No agents are predefined; the orchestrator writes every system prompt from scratch for each goal.

---

## How it works

```
User goal
   │
   ▼
Orchestrator          ← researches the goal (web_search, get_datetime)
   │                    plans a bespoke team, writes each agent's system prompt
   ▼
AgentManager          ← spawns agents via the factory, tears them down after
   └── AgentFactory   ← abstract: create / run / close agents
         └── AutoGenFactory  ✅ (LangGraph + Anthropic stubs pending)
   │
   ▼
AgentConversation     ← multi-round debate loop
   │                    agents see the full conversation history
   │                    orchestrator judges convergence after each round
   │                    max_rounds safety cap prevents runaway loops
   ▼
Orchestrator.synthesize  ← reads the full debate, writes the final report
   │
   ▼
FastAPI (SSE stream)  ← every event delivered in real time
   │
   ▼
Streamlit UI          ← 3 tabs: Chat · Orchestrator · Agent Activity
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
│   ├── conversation.py   # AgentConversation — multi-round debate loop
│   └── shared_thread.py  # SharedThread — sequential context passing
├── backends/
│   ├── autogen/          # ✅ AutoGen agentchat implementation
│   ├── langgraph/        # 🔧 stub
│   └── anthropic/        # 🔧 stub
├── tools/
│   ├── __init__.py       # Registry — @register, get_tools(), list_tools()
│   ├── web.py            # web_search (DuckDuckGo), fetch_url
│   ├── finance.py        # stock_price, company_financials (yfinance)
│   └── utility.py        # get_datetime, calculator, wikipedia_search
├── api/
│   └── app.py            # FastAPI backend — /health, /run, /run/stream
├── config/
│   └── settings.py       # Provider config + env resolution
└── main.py               # Minimal smoke-test entrypoint

app.py                    # Streamlit frontend (pure SSE consumer)
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
data: {"type": "plan_ready", "specs": [...]}
data: {"type": "agent_message", "agent": "Analyst", "content": "...", "round": 1}
data: {"type": "stop_signal", "reason": "...", "stopped_by": "orchestrator"}
data: {"type": "error", "message": "..."}
```

---

## Tool library

Tools are auto-discovered via a `@register` decorator. The orchestrator gets `web_search` and `get_datetime` to stay current before planning. Agents receive whichever tools the orchestrator assigns per goal.

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

## Adding a backend

1. Copy the structure of `src/agent_forge/backends/autogen/`
2. Implement `BaseAgent` (`run`, `close`) and `AgentFactory` (`create`)
3. Uncomment the relevant dependency in `pyproject.toml` and re-install

The manager, orchestrator, conversation loop, and API layer are all backend-agnostic — only the factory changes.

---

## Planned

- LangGraph and Anthropic backend implementations
- Parallel agent execution (currently sequential within a round)
- Additional tools (code executor, news API, etc.)
- Authentication / multi-user support
