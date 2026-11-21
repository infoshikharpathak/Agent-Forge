from __future__ import annotations

"""
ToolSource: plug an external tool provider into the tool registry.

A ToolSource lists tools (name, description, JSON schema) and executes
them. ``register_tool_source`` loads its tools into the registry as wrapper
callables, so every backend (AutoGen, LangGraph, OpenAI, Anthropic) can use
them unchanged, exactly like MCP tools:

- the JSON schema is attached as ``_mcp_input_schema`` (what the OpenAI
  schema builder and LangGraph already read), and
- a real ``__signature__`` is built from it, for backends that introspect
  function signatures (AutoGen).

When an agent calls a source tool, the wrapper reads the current
RunContext (set by ``execute_agent``) and passes it to ``source.call``, so
the source knows which trace and which agent the call belongs to.

    class MyTools:
        async def list_tools(self):
            return [ToolSpec("lookup_order", "Find an order", {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            })]

        async def call(self, name, args, ctx):
            return await my_backend.call(name, args, trace_id=ctx.trace_id)

    names = await register_tool_source(MyTools())
    ...
    unregister_tool_source(source)

Source tools are for agents only: the Orchestrator's research loop never
calls them (the planner may still assign them to agents).
"""

import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from agent_forge.core.context import RunContext, current_run_context


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters_json_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@runtime_checkable
class ToolSource(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, args: dict[str, Any], ctx: RunContext) -> Any: ...


class ToolSourceError(RuntimeError):
    pass


# id(source) -> tool names it registered
_SOURCE_TOOLS: dict[int, list[str]] = {}

_JSON_TO_PY: dict[str, Any] = {
    "string": str, "integer": int, "number": float, "boolean": bool,
    "array": list, "object": dict,
}


def _signature_from_schema(schema: dict) -> inspect.Signature:
    props: dict = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    # required first, so the signature is valid Python
    for name in sorted(props, key=lambda n: n not in required):
        if not name.isidentifier():
            continue
        annotation = _JSON_TO_PY.get((props[name] or {}).get("type"), Any)
        default = inspect.Parameter.empty if name in required else None
        if default is None and annotation is not Any:
            annotation = annotation | None
        params.append(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )
    return inspect.Signature(params, return_annotation=str)


def _make_wrapper(source: ToolSource, spec: ToolSpec):
    async def wrapper(**kwargs: Any) -> Any:
        ctx = current_run_context()
        if ctx is None:
            raise ToolSourceError(
                f"Tool {spec.name!r} was called outside an agent run (no RunContext)."
            )
        # Optional params the model left out arrive as None via the signature defaults.
        args = {k: v for k, v in kwargs.items() if v is not None}
        return await source.call(spec.name, args, ctx)

    wrapper.__name__ = spec.name
    wrapper.__doc__ = spec.description
    wrapper.__signature__ = _signature_from_schema(spec.parameters_json_schema)  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        **{p.name: p.annotation for p in wrapper.__signature__.parameters.values()},  # type: ignore[attr-defined]
        "return": str,
    }
    wrapper._mcp_input_schema = spec.parameters_json_schema  # type: ignore[attr-defined]
    wrapper._tool_source = source  # type: ignore[attr-defined]
    return wrapper


async def register_tool_source(source: ToolSource) -> list[str]:
    """Register every tool *source* lists. Raises ValueError (and registers
    nothing) if any name collides with an existing tool."""
    from agent_forge.tools import _REGISTRY, _ensure_loaded, register_callable

    if id(source) in _SOURCE_TOOLS:
        raise ValueError("This tool source is already registered.")
    _ensure_loaded()
    specs = list(await source.list_tools())

    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    taken = sorted(n for n in set(names) if n in _REGISTRY)
    if dupes or taken:
        raise ValueError(
            "Tool source names collide with existing tools: "
            + ", ".join(taken + [f"{d} (listed twice)" for d in dupes])
        )

    for spec in specs:
        register_callable(spec.name, _make_wrapper(source, spec), spec.description)
    _SOURCE_TOOLS[id(source)] = names
    return names


def unregister_tool_source(source: ToolSource) -> list[str]:
    """Remove every tool *source* registered. Returns their names."""
    from agent_forge.tools import unregister

    names = _SOURCE_TOOLS.pop(id(source), [])
    unregister(names)
    return names


@asynccontextmanager
async def tool_source(source: ToolSource) -> AsyncIterator[list[str]]:
    """``async with tool_source(src) as names:`` registers for the block,
    like MCPBridge."""
    names = await register_tool_source(source)
    try:
        yield names
    finally:
        unregister_tool_source(source)


def is_source_tool(fn: Any) -> bool:
    return getattr(fn, "_tool_source", None) is not None
