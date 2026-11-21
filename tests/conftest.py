import os

import pytest

# Backends build SDK clients at construction time; no network calls are made.
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")


@pytest.fixture(autouse=True)
def _isolated_trace_db(tmp_path, monkeypatch):
    """Every test writes run traces to its own throwaway SQLite file."""
    monkeypatch.setenv("AGENT_FORGE_DB_PATH", str(tmp_path / "traces.db"))
