from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from json import JSONDecodeError

from coding_agent.context.context import UsageStats
from coding_agent.telemetry.logger import TelemetryLogger


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        stream: bool = False,
        on_reasoning_delta: Any = None,
        debug_dir: Path | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.stream = stream
        self.on_reasoning_delta = on_reasoning_delta
        self.debug_dir = debug_dir
        self.telemetry = telemetry

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": self.stream}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        self._event(
            "llm.chat.start",
            "开始请求大模型",
            "llm",
            {
                "base_url": self.base_url,
                "model": self.model,
                "stream": self.stream,
                "message_count": len(messages),
                "tool_count": len(tools),
            },
        )
        if self.stream:
            return self._stream_chat(payload)

        try:
            with self._span("HTTP 非流式请求", "http", {"url": f"{self.base_url}/chat/completions"}):
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            if detail:
                raise LLMError(
                    f"{exc}. Response body: {detail}"
                ) from exc
            raise LLMError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc

        try:
            data = response.json()
        except JSONDecodeError as exc:
            body = response.text.strip()
            detail = f" Response body: {body[:500]}" if body else " Response body was empty."
            raise LLMError(f"Failed to parse non-streaming model response as JSON.{detail}") from exc
        result = {
            "message": data["choices"][0]["message"],
            "usage": UsageStats.from_response_usage(data.get("usage")),
        }
        self._event(
            "llm.chat.end",
            "大模型非流式响应解析完成",
            "llm",
            {
                "content_length": len(result["message"].get("content") or ""),
                "has_tool_calls": bool(result["message"].get("tool_calls")),
                "usage": self._debug_usage(result.get("usage")),
            },
        )
        self._write_debug_log(
            payload=payload,
            response={"json": data, "parsed": self._debug_result(result)},
        )
        return result

    def _stream_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._event(
            "llm.stream.start",
            "开始处理大模型流式响应",
            "llm",
            {"model": self.model},
        )
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        usage: UsageStats | None = None
        try:
            with self._span("HTTP 流式请求", "http", {"url": f"{self.base_url}/chat/completions"}):
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                        else:
                            data = line
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        chunks.append(chunk)
                        usage = UsageStats.from_response_usage(chunk.get("usage")) or usage
                        reasoning_delta = self._merge_stream_chunk(message, chunk, tool_calls)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                            if self.on_reasoning_delta is not None:
                                self.on_reasoning_delta(reasoning_delta)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            if detail:
                raise LLMError(f"{exc}. Response body: {detail}") from exc
            raise LLMError(str(exc)) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(str(exc)) from exc
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        result = {"message": message, "usage": usage}
        self._event(
            "llm.stream.end",
            "大模型流式响应解析完成",
            "llm",
            {
                "chunk_count": len(chunks),
                "content_length": len(message.get("content") or ""),
                "reasoning_length": len(message.get("reasoning_content") or ""),
                "tool_call_count": len(tool_calls),
                "usage": self._debug_usage(usage),
            },
        )
        self._write_debug_log(
            payload=payload,
            response={"stream_chunks": chunks, "parsed": self._debug_result(result)},
        )
        return result

    def _merge_stream_chunk(
        self,
        message: dict[str, Any],
        chunk: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> str | None:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return None
        role = delta.get("role")
        if isinstance(role, str):
            message["role"] = role
        content = delta.get("content")
        if isinstance(content, str):
            message["content"] = message.get("content", "") + content
            return None
        reasoning_content = delta.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
        streamed_tool_calls = delta.get("tool_calls")
        if isinstance(streamed_tool_calls, list):
            self._merge_tool_calls(tool_calls, streamed_tool_calls)
        return None

    def _merge_tool_calls(
        self,
        current_tool_calls: list[dict[str, Any]],
        delta_tool_calls: list[dict[str, Any]],
    ) -> None:
        self._event(
            "llm.tool_calls.merge",
            "合并流式 tool_calls 增量",
            "stream",
            {"delta_tool_call_count": len(delta_tool_calls), "current_tool_call_count": len(current_tool_calls)},
        )
        for delta_tool_call in delta_tool_calls:
            if not isinstance(delta_tool_call, dict):
                continue
            index = delta_tool_call.get("index")
            if not isinstance(index, int):
                continue
            while len(current_tool_calls) <= index:
                current_tool_calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            merged = current_tool_calls[index]
            tool_call_id = delta_tool_call.get("id")
            if isinstance(tool_call_id, str):
                merged["id"] = tool_call_id
            tool_call_type = delta_tool_call.get("type")
            if isinstance(tool_call_type, str):
                merged["type"] = tool_call_type
            function_delta = delta_tool_call.get("function")
            if isinstance(function_delta, dict):
                function = merged.setdefault("function", {"name": "", "arguments": ""})
                name = function_delta.get("name")
                if isinstance(name, str):
                    function["name"] += name
                arguments = function_delta.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] += arguments

    def _write_debug_log(self, *, payload: dict[str, Any], response: dict[str, Any]) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.debug_dir / f"{timestamp}.json"
        record = {
            "request": {
                "url": f"{self.base_url}/chat/completions",
                "payload": payload,
            },
            "response": response,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        self._event(
            "llm.debug_log.write",
            "大模型请求和响应 debug JSON 已落盘",
            "debug",
            {"path": str(path)},
        )

    def _debug_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": result.get("message"),
            "usage": self._debug_usage(result.get("usage")),
        }

    def _debug_usage(self, usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "cache_hit_ratio": usage.cache_hit_ratio,
        }

    def _event(self, event: str, message_zh: str, phase: str, metadata: dict[str, Any] | None = None) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            event,
            message_zh,
            function="OpenAICompatibleClient",
            phase=phase,
            metadata=metadata,
        )

    def _span(self, name: str, phase: str, metadata: dict[str, Any] | None = None) -> Any:
        if self.telemetry is None:
            return _NullSpan()
        return self.telemetry.span(name, function="OpenAICompatibleClient", phase=phase, metadata=metadata)


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
