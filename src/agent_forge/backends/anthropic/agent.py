from __future__ import annotations

import json
import inspect
from typing import Any

from anthropic import AsyncAnthropic

from agent_forge.core.agent import AgentRunResult, AgentStatus, BaseAgent


class AnthropicAgent(BaseAgent):
    """BaseAgent wrapper around the Anthropic Messages API with tool use."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        client: AsyncAnthropic,
        system_message: str,
        model: str = "claude-sonnet-4-5-20241022",
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        self._client = client
        self._system_message = system_message
        self._model = model
        self._tools = tools or []
        self._tool_handlers = tool_handlers or {}
        self._messages: list[dict] = []

    async def run(self, task: str, **kwargs: Any) -> AgentRunResult:
        self._messages.append({"role": "user", "content": task})

        self.status = AgentStatus.RUNNING
        input_tokens = 0
        output_tokens = 0

        try:
            while True:
                create_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": 4096,
                    "system": self._system_message,
                    "messages": self._messages,
                }
                if self._tools:
                    create_kwargs["tools"] = self._tools

                response = await self._client.messages.create(**create_kwargs)

                input_tokens += response.usage.input_tokens
                output_tokens += response.usage.output_tokens

                self._messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    text_parts = [
                        block.text for block in response.content if block.type == "text"
                    ]
                    self.status = AgentStatus.IDLE
                    return AgentRunResult(
                        content="\n".join(text_parts),
                        model=self._model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    handler = self._tool_handlers.get(block.name)
                    if handler is None:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": f"Unknown tool: {block.name}"}),
                            "is_error": True,
                        })
                        continue
                    try:
                        if inspect.iscoroutinefunction(handler):
                            result = await handler(**block.input)
                        else:
                            result = handler(**block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                        })
                    except Exception as exc:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": str(exc)}),
                            "is_error": True,
                        })

                self._messages.append({"role": "user", "content": tool_results})

        except Exception:
            self.status = AgentStatus.IDLE
            raise

    def reset_conversation(self) -> None:
        """Clear the conversation history for a fresh chat."""
        self._messages.clear()

    async def close(self) -> None:
        self.status = AgentStatus.CLOSED
