"""Unknown costs read "unknown" in the Streamlit Traces tab and the CLI,
never None, a dash or $0."""

from pathlib import Path

from agent_forge.main import _format_traces_table
from agent_forge.trace_view import agent_rows, format_cost, run_rows

RUN = {"run_id": "e6fa30d1-0000", "timestamp": "2026-11-23T18:27:55.000", "routing_tier": "locked",
       "app_id": "release_notes", "framework_used": "langgraph", "outcome": "success",
       "total_latency_ms": 5784, "total_input_tokens": 387, "total_output_tokens": 108}


def run(cost, unpriced=()):
    return {**RUN, "total_cost_usd": cost, "metadata": {"unpriced_models": list(unpriced)} if unpriced else {}}


def test_format_cost():
    assert format_cost(None) == "unknown"
    assert format_cost(0.000016) == "0.000016"
    assert format_cost(0.0) == "0.000000"  # a real zero stays a number
    assert format_cost(0.000016, partial=True) == "0.000016 + unknown"


def test_run_rows_mark_unknown_and_partial_totals():
    rows = run_rows([run(None, ["gpt-4.1-mini"]), run(0.000016, ["gpt-4.1-mini"]), run(0.000444)])
    assert [r["cost_usd"] for r in rows] == ["unknown", "0.000016 + unknown", "0.000444"]


def test_agent_rows_show_unknown_not_none():
    agents = [{"agent_id": "change_classifier", "model": "gpt-5.4-nano", "cost_usd": 0.00002, "content": "x"},
              {"agent_id": "notes_writer", "model": "gpt-4.1-mini", "cost_usd": None, "content": "y"}]
    rows = agent_rows(agents)
    assert [r["cost_usd"] for r in rows] == ["0.000020", "unknown"]
    assert all("content" not in r for r in rows)
    assert None not in [v for r in rows for v in r.values()]


def test_cli_traces_table_says_unknown():
    table = _format_traces_table([run(None, ["gpt-4.1-mini"]), run(0.000016, ["gpt-4.1-mini"])])
    assert "unknown" in table and "0.000016 + unknown" in table
    assert "None" not in table and "—" not in table


def test_streamlit_app_uses_the_shared_rows():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "from agent_forge.trace_view import agent_rows, run_rows" in app
    assert "run_rows(traces)" in app and 'agent_rows(detail["agents_spawned"])' in app
