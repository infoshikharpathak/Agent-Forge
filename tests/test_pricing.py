"""One pricing table; unknown models cost None (never $0) and are flagged."""

import logging
import sqlite3

import pytest

from agent_forge import db, pricing
from agent_forge.core.agent import AgentRunResult
from agent_forge.tracing import RunTracer


@pytest.fixture(autouse=True)
def _fresh_warnings():
    pricing._warned.clear()
    yield
    pricing._warned.clear()


def test_known_model_cost_from_token_counts():
    # gpt-4o-mini: $0.15 in / $0.60 out per 1M tokens
    assert pricing.estimate_cost("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert pricing.estimate_cost("gpt-4o-mini", 0, 1_000_000) == 0.60
    assert pricing.estimate_cost("gpt-4o-mini", 1716, 285) == round((1716 * 0.15 + 285 * 0.60) / 1e6, 6)
    # gpt-4o: $2.50 / $10.00
    assert pricing.estimate_cost("gpt-4o", 2000, 500) == 0.01


def test_dated_snapshot_uses_its_alias_price():
    assert pricing.estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15
    assert pricing.price_for("gpt-4o-2024-08-06") == pricing.PRICING_PER_1M["gpt-4o"]


def test_unknown_model_is_none_and_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_forge.pricing"):
        assert pricing.estimate_cost("mystery-model", 100, 100) is None
        assert pricing.estimate_cost("mystery-model", 5, 5) is None
        assert pricing.estimate_cost(None, 5, 5) is None
    warnings = [r.getMessage() for r in caplog.records]
    assert len([w for w in warnings if "'mystery-model'" in w]) == 1
    assert "not $0" in warnings[0]


def test_sum_costs_skips_unknown():
    assert pricing.sum_costs([0.1, None, 0.2]) == 0.3
    assert pricing.sum_costs([None, None]) is None
    assert pricing.sum_costs([]) is None
    assert pricing.sum_costs([0.0]) == 0.0  # a priced, genuinely free call stays 0


async def _run(models):
    tracer = RunTracer(task="t", routing_tier="tier")
    for i, model in enumerate(models):
        tracer.record_agent(f"a{i}", AgentRunResult(content="x", model=model, input_tokens=1_000_000,
                                                    output_tokens=0), 1.0, "success")
    await tracer.finish("success")
    return db.get_run(tracer.run_id)


async def test_run_total_leaves_unpriced_agents_out_and_flags_them():
    row = await _run(["gpt-4o-mini", "mystery-model", "gpt-4o"])
    assert row["total_cost_usd"] == 2.65  # 0.15 + 2.50; mystery-model left out
    assert [a["cost_usd"] for a in row["agents_spawned"]] == [0.15, None, 2.5]
    assert row["metadata"]["unpriced_models"] == ["mystery-model"]


async def test_run_with_no_priced_agent_has_null_total():
    row = await _run(["mystery-model"])
    assert row["total_cost_usd"] is None
    assert row["metadata"]["unpriced_models"] == ["mystery-model"]


async def test_fully_priced_run_has_no_unpriced_flag():
    row = await _run(["gpt-4o-mini"])
    assert row["total_cost_usd"] == 0.15 and "unpriced_models" not in row["metadata"]


def test_old_db_with_not_null_cost_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old_schema = db._SCHEMA.replace("total_cost_usd REAL,", "total_cost_usd REAL NOT NULL,")
    conn = sqlite3.connect(path)
    conn.execute(old_schema)
    conn.execute(
        "INSERT INTO runs VALUES ('r1','t1','2026-01-01','task','tier',NULL,NULL,'[]','[]',1.0,0.5,1,1,"
        "'success',NULL,'','{}')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("AGENT_FORGE_DB_PATH", str(path))

    assert db.get_run("r1")["total_cost_usd"] == 0.5  # old rows kept
    with sqlite3.connect(path) as c:
        notnull = {r[1]: r[3] for r in c.execute("PRAGMA table_info(runs)")}
    assert notnull["total_cost_usd"] == 0
    db.insert_run({**db.get_run("r1"), "run_id": "r2", "total_cost_usd": None})
    assert db.get_run("r2")["total_cost_usd"] is None
