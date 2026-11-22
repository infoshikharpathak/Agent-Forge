"""Per-model token pricing: the one table every cost estimate reads.

Agent Forge's run traces use it, and so does bastion (agent and injection-
classifier costs), so a price is added here once and applies everywhere.

A model that isn't in the table has *no* cost estimate: ``estimate_cost``
returns None (and logs a warning once per model), never 0.0, so an unpriced
run can't pass for a free one. Totals add up only the priced parts and
report which models were left out.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# USD per 1M tokens, as (input, output): standard tier, uncached input.
# Estimates for dashboards, not billing truth. Update a row only from the
# provider's own pricing page, and say when it was checked.
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # OpenAI, from https://developers.openai.com/api/docs/pricing
    # (Standard, short context), checked 2026-11-22.
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Earlier entries; source and date not recorded.
    "gpt-5.4-mini": (0.25, 1.00),
    "gpt-5.4-nano": (0.05, 0.20),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-flash": (0.075, 0.30),
}

# Providers report dated snapshots ("gpt-4o-mini-2024-07-18"); they cost the
# same as the alias they snapshot, so fall back to it.
_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")

_warned: set[str] = set()


def price_for(model: str | None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for *model*, or None if unpriced."""
    if not model:
        return None
    if model in PRICING_PER_1M:
        return PRICING_PER_1M[model]
    return PRICING_PER_1M.get(_SNAPSHOT.sub("", model))


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost of one call, or None when *model* has no price."""
    rates = price_for(model)
    if rates is None:
        key = model or "<none>"
        if key not in _warned:
            _warned.add(key)
            log.warning("No price for model %r: its cost is recorded as unknown (null), not $0. "
                        "Add it to agent_forge.pricing.PRICING_PER_1M.", key)
        return None
    in_rate, out_rate = rates
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


def sum_costs(costs: list[float | None]) -> float | None:
    """Total of the known costs; None if none of them is known."""
    known = [c for c in costs if c is not None]
    return round(sum(known), 6) if known else None
