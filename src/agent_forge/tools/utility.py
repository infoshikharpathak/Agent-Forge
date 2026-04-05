from __future__ import annotations

"""Utility tools: datetime, calculator, wikipedia."""

import ast
import math
from datetime import datetime, timezone

from agent_forge.tools import register


@register("get_datetime")
def get_datetime() -> str:
    """Return the current UTC date and time as a formatted string."""
    return datetime.now(timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S")


# Safe names exposed to the calculator
_MATH_NAMES: dict = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


@register("calculator")
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports standard arithmetic and math functions (sqrt, log, sin, cos, etc.).
    Example: calculator("sqrt(144) + log(10)")
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = eval(  # noqa: S307
            compile(tree, "<calculator>", "eval"),
            {"__builtins__": {}},
            _MATH_NAMES,
        )
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@register("wikipedia_search")
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia and return a summary of the most relevant article.
    Use for factual background information on people, places, concepts, or events.
    """
    import wikipedia  # type: ignore

    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(query, results=3)
        if not results:
            return f"No Wikipedia results found for: {query}"
        return wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
    except wikipedia.DisambiguationError as exc:
        try:
            return wikipedia.summary(exc.options[0], sentences=sentences, auto_suggest=False)
        except Exception:
            return f"Ambiguous query '{query}'. Try being more specific. Options: {exc.options[:5]}"
    except Exception as exc:
        return f"Wikipedia search failed for '{query}': {exc}"
