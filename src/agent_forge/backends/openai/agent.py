from __future__ import annotations

import inspect
import json
from typing import Any

from openai import AsyncOpenAI

from agent_forge.core.agent import AgentRunResult, AgentStatus, BaseAgent


class OpenAIAgent(BaseAgent):
    """BaseAgent wrapper around the OpenAI Chat Completions API with tool use."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        client: AsyncOpenAI,
        system_message: str,
        model: str = "gpt-4o-mini",
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
                    "messages": [
                        {"role": "system", "content": self._system_message},
                        *self._messages,
                    ],
                }
                if self._tools:
                    create_kwargs["tools"] = self._tools
                    create_kwargs["tool_choice"] = "auto"

                response = await self._client.chat.completions.create(**create_kwargs)

                usage = response.usage
                if usage:
                    input_tokens += usage.prompt_tokens
                    output_tokens += usage.completion_tokens

                msg = response.choices[0].message

                if not msg.tool_calls:
                    content = msg.content or ""
                    self._messages.append({"role": "assistant", "content": content})
                    self.status = AgentStatus.IDLE
                    return AgentRunResult(
                        content=content,
                        model=self._model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                self._messages.append(msg.model_dump(exclude_none=True))

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    handler = self._tool_handlers.get(fn_name)
                    if handler is None:
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": f"Unknown tool: {fn_name}"}),
                        })
                        continue

                    try:
                        if inspect.iscoroutinefunction(handler):
                            result = await handler(**args)
                        else:
                            result = handler(**args)
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                        })
                    except Exception as exc:
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": str(exc)}),
                        })

        except Exception:
            self.status = AgentStatus.IDLE
            raise

    def reset_conversation(self) -> None:
        self._messages.clear()

    async def close(self) -> None:
        self.status = AgentStatus.CLOSED
