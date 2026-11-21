"""RunContext, caller-supplied trace_id on RunTracer, and the API pass-through."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_forge import db
from agent_forge.core.context import RunContext, current_run_context, use_run_context
from agent_forge.tracing import RunTracer


def test_trace_id_defaults_to_run_id():
    tracer = RunTracer(task="t", routing_tier="dynamic")
    assert tracer.trace_id == tracer.run_id
    ctx = tracer.run_context()
    assert ctx.trace_id == ctx.run_id == tracer.run_id
    assert ctx.agent_name is None and ctx.app is None


def test_caller_supplied_trace_id_kept():
    tracer = RunTracer(task="t", routing_tier="dynamic", trace_id="req-123")
    assert tracer.trace_id == "req-123" and tracer.run_id != "req-123"
    assert tracer.run_context().trace_id == "req-123"


async def test_trace_id_persisted_and_queryable():
    tracer = RunTracer(task="t", routing_tier="dynamic", trace_id="req-abc")
    await tracer.finish(outcome="success")
    row = db.get_run(tracer.run_id)
    assert row["trace_id"] == "req-abc"
    assert [r["run_id"] for r in db.query_runs(trace_id="req-abc")] == [tracer.run_id]
    assert db.query_runs(trace_id="nope") == []


async def test_default_trace_id_persisted_as_run_id():
    tracer = RunTracer(task="t", routing_tier="dynamic")
    await tracer.finish(outcome="success")
    assert db.get_run(tracer.run_id)["trace_id"] == tracer.run_id


def test_old_db_without_trace_id_column_migrates(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(db._SCHEMA.replace("    trace_id TEXT,\n", ""))
    monkeypatch.setenv("AGENT_FORGE_DB_PATH", str(path))
    with db._connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert "trace_id" in cols


async def test_context_is_task_local_and_reset():
    assert current_run_context() is None
    seen = {}

    async def run(name):
        with use_run_context(RunContext(trace_id=name, run_id=name, agent_name=name)):
            await asyncio.sleep(0)
            seen[name] = current_run_context().trace_id

    await asyncio.gather(run("a"), run("b"))
    assert seen == {"a": "a", "b": "b"}
    assert current_run_context() is None


def test_api_run_passes_trace_id_to_persisted_row(monkeypatch):
    """The pipeline fails right after the tracer is created; the failed run is
    still persisted, so its row shows the caller's trace_id."""
    import contextlib

    from agent_forge.api import app as api

    class FakeOrchestrator:
        def __init__(self, provider):
            pass

        async def clarify_goal(self, goal):
            raise RuntimeError("stop here")

    monkeypatch.setattr(api, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(api, "MCPBridge", contextlib.nullcontext)
    monkeypatch.setattr(api.app_registry, "list_apps", lambda: [])

    client = TestClient(api.app)
    with pytest.raises(RuntimeError, match="stop here"):
        client.post("/run", json={"goal": "g", "trace_id": "caller-trace-7"})

    rows = db.query_runs(trace_id="caller-trace-7")
    assert len(rows) == 1 and rows[0]["outcome"] == "failed"
    assert client.get("/traces", params={"trace_id": "caller-trace-7"}).json()[0]["run_id"] == rows[0]["run_id"]
