#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DeepSeek V4 tool-call streaming parser compatibility patch.
#

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

ESCAPED_ARGUMENTS_PARAM_NAME = "__vllm_param_arguments__"

_DSML_DEBUG_ENABLED = os.getenv("VLLM_ASCEND_DSML_DEBUG", "1").lower() not in {
    "0",
    "false",
    "off",
}


def _dsml_debug(event: str, **fields: Any) -> None:
    if not _DSML_DEBUG_ENABLED:
        return
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[vllm-ascend DSML debug] {event} {details}".rstrip(), flush=True)


def _parser_type(parser: Any) -> str | None:
    if parser is None:
        return None
    parser_type = type(parser)
    return f"{parser_type.__module__}.{parser_type.__qualname__}"


def _dsml_tag_preview(text: str, limit: int = 240) -> str:
    """Keep tag spelling visible without logging a complete parameter body."""
    preview = text[:limit]
    parameter_idx = preview.find("<｜DSML｜parameter")
    if parameter_idx != -1:
        tag_end = preview.find(">", parameter_idx)
        if tag_end != -1:
            preview = preview[: tag_end + 1] + "<parameter-body-redacted>"
    return preview.replace("\r", "\\r").replace("\n", "\\n")


def _debug_delegating_output(
    parser: DelegatingParser,
    delta_message: DeltaMessage | None,
    tool_parser: Any,
) -> None:
    content = delta_message.content if delta_message is not None else None
    if not content:
        return

    probe = getattr(parser, "_deepseek_v4_debug_returned_content_probe", "") + content
    parser._deepseek_v4_debug_returned_content_probe = probe[-512:]
    if ("DSML" in probe or "<｜" in probe) and not getattr(
        parser, "_deepseek_v4_debug_content_leak_announced", False
    ):
        state = parser._stream_state
        _dsml_debug(
            "raw-dsml-returned-as-content",
            delegating_parser_id=id(parser),
            content_preview=_dsml_tag_preview(probe),
            tool_parser_type=_parser_type(tool_parser),
            reasoning_parser_type=_parser_type(getattr(parser, "_reasoning_parser", None)),
            reasoning_ended=state.reasoning_ended,
            tool_call_text_started=state.tool_call_text_started,
        )
        parser._deepseek_v4_debug_content_leak_announced = True


_dsml_debug("patch-loaded", file=__file__)


def _ensure_parser_regexes(self: DeepSeekV4ToolParser) -> None:
    self.tool_call_complete_regex = re.compile(
        re.escape(self.tool_call_start_token) + r"(.*?)" + re.escape(self.tool_call_end_token),
        re.DOTALL,
    )
    self.invoke_complete_regex = re.compile(
        r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>',
        re.DOTALL,
    )
    self.parameter_complete_regex = re.compile(
        r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
        re.DOTALL,
    )
    self.parameter_start_regex = re.compile(r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>')
    self.invoke_start_regex = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>')


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_overlap = min(len(text), len(tag) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(tag[:overlap]):
            return overlap
    return 0


def _ensure_streaming_attrs(self: DeepSeekV4ToolParser) -> None:
    if not hasattr(self, "_buffer"):
        self._buffer = ""
    if not hasattr(self, "_in_tool_calls"):
        self._in_tool_calls = False
    if not hasattr(self, "_active_tool_index"):
        self._active_tool_index = None
    if not hasattr(self, "_active_tool_name"):
        self._active_tool_name = None
    if not hasattr(self, "_streaming_param_mode"):
        self._streaming_param_mode = None
    if not hasattr(self, "_streaming_param_key"):
        self._streaming_param_key = None
    if not hasattr(self, "_streaming_param_raw_parts"):
        self._streaming_param_raw_parts = []
    if not hasattr(self, "_args_started"):
        self._args_started = []
    if not hasattr(self, "_pending_delta_messages"):
        self._pending_delta_messages = deque()

    _ensure_parser_regexes(self)

    if not hasattr(self, "current_tool_index"):
        self.current_tool_index = 0
    if not hasattr(self, "prev_tool_call_arr"):
        self.prev_tool_call_arr = []
    if not hasattr(self, "streamed_args_for_tool"):
        self.streamed_args_for_tool = []


def _function_name(tool) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    return getattr(getattr(tool, "function", None), "name", None)


def _function_parameters(tool):
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("parameters")
        return getattr(function, "parameters", None)
    return getattr(getattr(tool, "function", None), "parameters", None)


def _extract_types_from_schema(schema: Any) -> list[str]:
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()
    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.add(type_value)
    elif isinstance(type_value, list):
        types.update(t for t in type_value if isinstance(t, str))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        for value in enum_values:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(choice_field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema(choice))

    return list(types) if types else ["string"]


_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}


def _coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
    if isinstance(schema_type, str):
        schema_type = [schema_type]

    normalized_types = {_TYPE_ALIASES.get(key, key) for t in schema_type for key in [t.strip().lower()]}

    for candidate_type in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if candidate_type not in normalized_types:
            continue

        if candidate_type == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate_type == "string":
            return value
        if candidate_type == "integer":
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        if candidate_type == "number":
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (TypeError, ValueError):
                continue
        if candidate_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _convert_param_value_checked(value: str, param_type: str) -> Any:
    if value.lower() == "null":
        return None

    param_type = param_type.lower()
    if param_type in ["string", "str", "text"]:
        return value
    if param_type in ["integer", "int"]:
        return int(value)
    if param_type in ["number", "float"]:
        val = float(value)
        return val if val != int(val) else int(val)
    if param_type in ["boolean", "bool"]:
        value = value.strip()
        if value.lower() not in ["false", "0", "true", "1"]:
            raise ValueError("Invalid boolean value")
        return value.lower() in ["true", "1"]
    if param_type in ["object", "array"]:
        return json.loads(value)
    return json.loads(value)


def _convert_param_value(self: DeepSeekV4ToolParser, value: str, param_type) -> Any:
    if not isinstance(param_type, list):
        param_type = [param_type]
    for current_type in param_type:
        try:
            return _convert_param_value_checked(value, current_type)
        except Exception:
            continue
    return value


def _extract_param_name(param_name: str) -> str:
    if param_name == ESCAPED_ARGUMENTS_PARAM_NAME:
        return "arguments"
    return param_name


def _get_param_config(self: DeepSeekV4ToolParser, request, function_name):
    if not request or not request.tools or not function_name:
        return {}
    for tool in request.tools:
        if _function_name(tool) != function_name:
            continue
        params = _function_parameters(tool)
        if isinstance(params, dict):
            properties = params.get("properties")
            if isinstance(properties, dict):
                return properties
        return {}
    return {}


def _coerce_param_value(
    self: DeepSeekV4ToolParser,
    value: str,
    *,
    string_attr: str,
    param_type,
):
    if string_attr == "true":
        return value
    if param_type:
        return _coerce_to_schema_type(value, param_type)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _repair_param_dict(
    param_dict: dict,
    param_config: dict[str, dict],
) -> dict:
    allowed = set(param_config.keys())
    for wrapper in ("arguments", "input"):
        if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = param_dict[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return param_dict
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return inner
    return param_dict


def _parse_invoke_params(
    self: DeepSeekV4ToolParser,
    invoke_str: str,
    request: ChatCompletionRequest | None = None,
    function_name: str | None = None,
) -> dict:
    _ensure_parser_regexes(self)
    param_config = _get_param_config(self, request, function_name)
    param_dict = {}
    for param_name, string_attr, param_val in self.parameter_complete_regex.findall(invoke_str):
        original_param_name = param_name
        param_name = _extract_param_name(param_name)
        param_type = None
        if original_param_name == ESCAPED_ARGUMENTS_PARAM_NAME and "arguments" in param_config:
            param_type = _extract_types_from_schema(param_config["arguments"])
        elif param_name in param_config and isinstance(param_config[param_name], dict):
            param_type = _extract_types_from_schema(param_config[param_name])

        param_dict[param_name] = _coerce_param_value(
            self,
            param_val,
            string_attr=string_attr,
            param_type=param_type,
        )

    return _repair_param_dict(param_dict, param_config)


def _patched_extract_tool_calls(
    self: DeepSeekV4ToolParser,
    model_output: str,
    request: ChatCompletionRequest,
) -> ExtractedToolCallInformation:
    if self.tool_call_start_token not in model_output:
        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    first_tool_idx = model_output.find(self.tool_call_start_token)
    content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

    try:
        _ensure_parser_regexes(self)
        tool_calls = []
        block_start = first_tool_idx
        while block_start != -1:
            payload_start = block_start + len(self.tool_call_start_token)
            block_end = model_output.find(self.tool_call_end_token, payload_start)
            if block_end == -1:
                tool_call_payload = model_output[payload_start:]
            else:
                tool_call_payload = model_output[payload_start:block_end]

            for invoke_name, invoke_content in self.invoke_complete_regex.findall(tool_call_payload):
                params = _parse_invoke_params(self, invoke_content, request, invoke_name)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=invoke_name,
                            arguments=json.dumps(params, ensure_ascii=False),
                        ),
                    )
                )

            if block_end == -1:
                break
            block_start = model_output.find(
                self.tool_call_start_token,
                block_end + len(self.tool_call_end_token),
            )

        if not tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )
    except Exception:
        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)


def _reset_streaming_state(self: DeepSeekV4ToolParser) -> None:
    _ensure_streaming_attrs(self)
    self.current_tool_index = 0
    self._buffer = ""
    self._in_tool_calls = False
    self._active_tool_index = None
    self._active_tool_name = None
    self._streaming_param_mode = None
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()
    self.prev_tool_call_arr.clear()
    self.streamed_args_for_tool.clear()
    self._pending_delta_messages.clear()
    self._args_started.clear()


def _json_escape_string_content(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _drain_pending_tool_call_deltas(self: DeepSeekV4ToolParser):
    while self._pending_delta_messages:
        yield self._pending_delta_messages.popleft()


def _pop_pending_delta_message(self: DeepSeekV4ToolParser) -> DeltaMessage | None:
    if not self._pending_delta_messages:
        return None

    content_parts = []
    merged_tool_calls: dict[int, DeltaToolCall] = {}
    while self._pending_delta_messages:
        message = self._pending_delta_messages.popleft()
        if message.content:
            content_parts.append(message.content)
        for tool_call in message.tool_calls or []:
            index = tool_call.index
            function = tool_call.function
            if index not in merged_tool_calls:
                merged_tool_calls[index] = DeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=function.name if function else None,
                        arguments=function.arguments if function else None,
                    ),
                )
                continue

            merged = merged_tool_calls[index]
            if tool_call.id is not None:
                merged.id = tool_call.id
            if tool_call.type is not None:
                merged.type = tool_call.type
            if function is None:
                continue
            if merged.function is None:
                merged.function = DeltaFunctionCall()
            if function.name is not None:
                merged.function.name = function.name
            if function.arguments is not None:
                merged.function.arguments = (merged.function.arguments or "") + function.arguments

    content = "".join(content_parts) or None
    return DeltaMessage(content=content, tool_calls=list(merged_tool_calls.values()))


def _merge_delta_messages(
    first: DeltaMessage | None,
    second: DeltaMessage | None,
) -> DeltaMessage | None:
    if first is None:
        return second
    if second is None:
        return first

    content = None
    if first.content is not None or second.content is not None:
        content = (first.content or "") + (second.content or "")

    reasoning = None
    if first.reasoning is not None or second.reasoning is not None:
        reasoning = (first.reasoning or "") + (second.reasoning or "")

    merged_tool_calls: dict[int, DeltaToolCall] = {}
    for message in (first, second):
        for tool_call in message.tool_calls or []:
            index = tool_call.index
            function = tool_call.function
            if index not in merged_tool_calls:
                merged_tool_calls[index] = DeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=function.name if function else None,
                        arguments=function.arguments if function else None,
                    ),
                )
                continue

            merged = merged_tool_calls[index]
            if tool_call.id is not None:
                merged.id = tool_call.id
            if tool_call.type is not None:
                merged.type = tool_call.type
            if function is None:
                continue
            if merged.function is None:
                merged.function = DeltaFunctionCall()
            if function.name is not None:
                merged.function.name = function.name
            if function.arguments is not None:
                merged.function.arguments = (merged.function.arguments or "") + function.arguments

    return DeltaMessage(
        role=second.role or first.role,
        content=content,
        reasoning=reasoning,
        tool_calls=list(merged_tool_calls.values()),
    )


def _queue_delta_message(self: DeepSeekV4ToolParser, message: DeltaMessage | None) -> None:
    if message is not None:
        self._pending_delta_messages.append(message)


def _emit_tool_name_delta(self: DeepSeekV4ToolParser, index: int, name: str) -> DeltaMessage:
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                id=self._generate_tool_call_id(),
                function=DeltaFunctionCall(name=name, arguments=""),
                type="function",
            )
        ]
    )


def _emit_tool_args_delta(self: DeepSeekV4ToolParser, index: int, arguments: str) -> DeltaMessage | None:
    if not arguments:
        return None
    self.streamed_args_for_tool[index] += arguments
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=arguments),
            )
        ]
    )


def _begin_streaming_tool_call(self: DeepSeekV4ToolParser, name: str) -> None:
    self._active_tool_index = self.current_tool_index
    self._active_tool_name = name
    self.current_tool_index += 1
    self.prev_tool_call_arr.append({"name": name, "arguments": {}})
    self.streamed_args_for_tool.append("")
    self._args_started.append(False)
    self._queue_delta_message(self._emit_tool_name_delta(self._active_tool_index, name))


def _append_param_prefix(self: DeepSeekV4ToolParser, index: int, key: str, *, is_string: bool) -> None:
    key_json = json.dumps(key, ensure_ascii=False)
    prefix = "{" if not self._args_started[index] else ","
    frag = prefix + key_json + ":"
    if is_string:
        frag += '"'
    self._args_started[index] = True
    self._queue_delta_message(self._emit_tool_args_delta(index, frag))


def _append_json_param_value(self: DeepSeekV4ToolParser, index: int, key: str, value: Any) -> None:
    key_json = json.dumps(key, ensure_ascii=False)
    value_json = json.dumps(value, ensure_ascii=False)
    prefix = "{" if not self._args_started[index] else ","
    self._args_started[index] = True
    self._queue_delta_message(self._emit_tool_args_delta(index, prefix + key_json + ":" + value_json))


def _append_raw_param_value(
    self: DeepSeekV4ToolParser,
    index: int,
    key: str,
    raw_value: str,
    *,
    is_string: bool,
) -> None:
    _append_param_prefix(self, index, key, is_string=is_string)
    if is_string:
        frag = _json_escape_string_content(raw_value) + '"'
    else:
        frag = raw_value
    self._queue_delta_message(self._emit_tool_args_delta(index, frag))


def _param_types_for_name(
    self: DeepSeekV4ToolParser,
    name: str,
    request: ChatCompletionRequest | None,
) -> list[str]:
    param_config = _get_param_config(self, request, self._active_tool_name)
    if name in param_config and isinstance(param_config[name], dict):
        return _extract_types_from_schema(param_config[name])
    return ["string"]


def _can_stream_raw_param(param_types: list[str]) -> bool:
    return set(param_types).issubset({"object", "array"})


def _finish_buffered_param(
    self: DeepSeekV4ToolParser,
    index: int,
    request: ChatCompletionRequest | None,
) -> None:
    key = self._streaming_param_key
    if key is None:
        return

    raw_value = "".join(self._streaming_param_raw_parts)
    param_types = _param_types_for_name(self, key, request)
    value = _coerce_to_schema_type(raw_value, param_types)
    _append_json_param_value(self, index, key, value)
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()


def _should_buffer_wrapper_param(self: DeepSeekV4ToolParser, key: str, request: ChatCompletionRequest | None) -> bool:
    if self._args_started[self._active_tool_index]:
        return False
    param_config = _get_param_config(self, request, self._active_tool_name)
    return bool(param_config and key in ("arguments", "input") and key not in param_config)


def _finish_buffered_wrapper_param(
    self: DeepSeekV4ToolParser,
    index: int,
    request: ChatCompletionRequest | None,
) -> None:
    key = self._streaming_param_key
    if key is None:
        return

    raw_value = "".join(self._streaming_param_raw_parts)
    is_string = self._streaming_param_mode == "wrapper_string"
    value: Any = raw_value
    if not is_string:
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value

    param_dict = {key: value}
    param_config = _get_param_config(self, request, self._active_tool_name)
    repaired = _repair_param_dict(param_dict, param_config)
    if isinstance(repaired, dict) and repaired is not param_dict:
        for repaired_key, repaired_value in repaired.items():
            _append_json_param_value(self, index, repaired_key, repaired_value)
    else:
        _append_raw_param_value(self, index, key, raw_value, is_string=is_string)

    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()


def _close_streaming_tool_call(self: DeepSeekV4ToolParser) -> None:
    index = self._active_tool_index
    if index is None:
        return

    suffix = "}" if self._args_started[index] else "{}"
    self._queue_delta_message(self._emit_tool_args_delta(index, suffix))
    with suppress(json.JSONDecodeError, IndexError):
        self.prev_tool_call_arr[index] = {
            "name": self._active_tool_name,
            "arguments": json.loads(self.streamed_args_for_tool[index]),
        }

    self._active_tool_index = None
    self._active_tool_name = None
    self._streaming_param_mode = None
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()


def _safe_content_len_before_tag_end(self: DeepSeekV4ToolParser) -> int:
    safe_len = len(self._buffer)
    parameter_end_token = "</｜DSML｜parameter>"
    for overlap in range(1, len(parameter_end_token)):
        if self._buffer.endswith(parameter_end_token[:overlap]):
            safe_len = len(self._buffer) - overlap
            break
    return safe_len


def _process_streaming_buffer(self: DeepSeekV4ToolParser, request: ChatCompletionRequest | None) -> None:
    parameter_end_token = "</｜DSML｜parameter>"
    invoke_end_token = "</｜DSML｜invoke>"

    while True:
        if not self._in_tool_calls:
            start_idx = self._buffer.find(self.tool_call_start_token)
            if start_idx == -1:
                overlap = _partial_tag_overlap(self._buffer, self.tool_call_start_token)
                sendable_idx = len(self._buffer) - overlap
                if sendable_idx > 0:
                    content = self._buffer[:sendable_idx]
                    self._buffer = self._buffer[sendable_idx:]
                    self._queue_delta_message(DeltaMessage(content=content))
                return

            if start_idx > 0:
                content = self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx:]
                self._queue_delta_message(DeltaMessage(content=content))
                continue

            self._buffer = self._buffer[len(self.tool_call_start_token) :]
            self._in_tool_calls = True
            _dsml_debug(
                "tool-start-consumed",
                tool_parser_id=id(self),
                remaining_buffer_len=len(self._buffer),
            )
            continue

        if self._active_tool_index is None:
            stripped_len = len(self._buffer) - len(self._buffer.lstrip())
            if stripped_len:
                self._buffer = self._buffer[stripped_len:]
                continue

            if self._buffer.startswith(self.tool_call_end_token):
                self._buffer = self._buffer[len(self.tool_call_end_token) :]
                self._in_tool_calls = False
                continue

            match = self.invoke_start_regex.match(self._buffer)
            if match is None:
                return

            self._buffer = self._buffer[match.end() :]
            _dsml_debug(
                "invoke-start-consumed",
                tool_parser_id=id(self),
                function_name=match.group(1),
                remaining_buffer_len=len(self._buffer),
            )
            self._begin_streaming_tool_call(match.group(1))
            continue

        index = self._active_tool_index

        if self._streaming_param_mode is not None:
            end_pos = self._buffer.find(parameter_end_token)
            if end_pos != -1:
                raw_content = self._buffer[:end_pos]
                self._buffer = self._buffer[end_pos + len(parameter_end_token) :]
                if self._streaming_param_mode.startswith("wrapper_"):
                    self._streaming_param_raw_parts.append(raw_content)
                    _finish_buffered_wrapper_param(self, index, request)
                elif self._streaming_param_mode == "buffered_json":
                    self._streaming_param_raw_parts.append(raw_content)
                    _finish_buffered_param(self, index, request)
                elif self._streaming_param_mode == "string":
                    frag = _json_escape_string_content(raw_content) + '"'
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
                else:
                    frag = raw_content
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))

                self._streaming_param_mode = None
                continue

            safe_len = _safe_content_len_before_tag_end(self)
            if safe_len > 0:
                raw_content = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                if self._streaming_param_mode.startswith("wrapper_") or self._streaming_param_mode == "buffered_json":
                    self._streaming_param_raw_parts.append(raw_content)
                elif self._streaming_param_mode == "string":
                    frag = _json_escape_string_content(raw_content)
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
                else:
                    frag = raw_content
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
            return

        stripped_len = len(self._buffer) - len(self._buffer.lstrip())
        if stripped_len:
            self._buffer = self._buffer[stripped_len:]
            continue

        if self._buffer.startswith(invoke_end_token):
            self._buffer = self._buffer[len(invoke_end_token) :]
            _close_streaming_tool_call(self)
            continue

        match = self.parameter_start_regex.match(self._buffer)
        if match is None:
            return

        self._buffer = self._buffer[match.end() :]
        key = _extract_param_name(match.group(1))
        string_attr = match.group(2)
        is_string = string_attr == "true"
        _dsml_debug(
            "parameter-start-consumed",
            tool_parser_id=id(self),
            parameter_name=key,
            string_attr=string_attr,
            remaining_buffer_len=len(self._buffer),
        )
        if _should_buffer_wrapper_param(self, key, request):
            self._streaming_param_key = key
            self._streaming_param_raw_parts.clear()
            self._streaming_param_mode = "wrapper_string" if is_string else "wrapper_json"
            continue

        if not is_string:
            param_types = _param_types_for_name(self, key, request)
            if not _can_stream_raw_param(param_types):
                self._streaming_param_key = key
                self._streaming_param_raw_parts.clear()
                self._streaming_param_mode = "buffered_json"
                continue

        _append_param_prefix(self, index, key, is_string=is_string)
        self._streaming_param_mode = "string" if is_string else "json"


def _finish_streaming(
    self: DeepSeekV4ToolParser,
    request: ChatCompletionRequest | None = None,
) -> DeltaMessage | None:
    _ensure_streaming_attrs(self)
    _dsml_debug(
        "tool-parser-finish-enter",
        tool_parser_id=id(self),
        in_tool_calls=self._in_tool_calls,
        active_tool_index=self._active_tool_index,
        buffer_len=len(self._buffer),
        buffer_preview=(
            "<inside-tool-call-residue>"
            if self._in_tool_calls
            else _dsml_tag_preview(self._buffer)
        ),
    )
    _process_streaming_buffer(self, request)
    pending_delta = _pop_pending_delta_message(self)

    if not self._in_tool_calls:
        if self._buffer:
            pending_delta = _merge_delta_messages(
                pending_delta,
                DeltaMessage(content=self._buffer),
            )
            self._buffer = ""
        return pending_delta

    # Anything left in the buffer belongs to an unfinished DSML structure.
    # It must never be exposed as ordinary response content.
    self._buffer = ""

    active_index = self._active_tool_index
    if active_index is not None:
        # Do not force-close an incomplete invoke or parameter. Remove its
        # bookkeeping so DelegatingParser cannot append a fabricated `{}`.
        for values in (
            self.prev_tool_call_arr,
            self.streamed_args_for_tool,
            self._args_started,
        ):
            if active_index < len(values):
                del values[active_index:]
        self.current_tool_index = active_index

    self._streaming_param_mode = None
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()
    self._active_tool_index = None
    self._active_tool_name = None
    self._in_tool_calls = False

    _dsml_debug(
        "tool-parser-finish-discarded-incomplete-dsml",
        tool_parser_id=id(self),
        active_tool_index=active_index,
    )

    return pending_delta


def _patched_extract_tool_calls_streaming(
    self: DeepSeekV4ToolParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest,
) -> DeltaMessage | None:
    _ensure_streaming_attrs(self)
    if not previous_text:
        self._reset_streaming_state()

    debug_relevant = (
        self._in_tool_calls
        or "DSML" in delta_text
        or "<｜" in delta_text
        or bool(_partial_tag_overlap(delta_text, self.tool_call_start_token))
    )
    if debug_relevant:
        _dsml_debug(
            "tool-parser-input",
            tool_parser_id=id(self),
            previous_text_len=len(previous_text),
            delta_len=len(delta_text),
            delta_preview=(
                _dsml_tag_preview(delta_text)
                if not self._in_tool_calls
                else "<inside-tool-call>"
            ),
            buffer_len_before=len(self._buffer),
            in_tool_calls=self._in_tool_calls,
        )

    self._buffer += delta_text
    _process_streaming_buffer(self, request)

    pending_delta = _pop_pending_delta_message(self)
    if pending_delta is not None:
        content = pending_delta.content or ""
        first_tool_call = (pending_delta.tool_calls or [None])[0]
        function = first_tool_call.function if first_tool_call is not None else None
        if content or (function is not None and function.name):
            _dsml_debug(
                "tool-parser-output",
                tool_parser_id=id(self),
                content_has_dsml="<｜DSML｜" in content,
                content_preview=_dsml_tag_preview(content),
                tool_call_count=len(pending_delta.tool_calls or []),
                function_name=function.name if function is not None else None,
            )
        return pending_delta

    if not delta_text and delta_token_ids and self.prev_tool_call_arr:
        return DeltaMessage(content="")

    return None


_original_delegating_parse_delta = DelegatingParser.parse_delta


def _patched_delegating_parse_delta(
    self: DelegatingParser,
    delta_text: str,
    delta_token_ids: list[int],
    request: Any,
    prompt_token_ids: list[int] | None = None,
    *,
    finished: bool,
) -> DeltaMessage | None:
    tool_parser = getattr(self, "_tool_parser", None)
    prefix_delta = None
    state = self._stream_state

    if isinstance(tool_parser, DeepSeekV4ToolParser) and not getattr(
        self, "_deepseek_v4_debug_stream_announced", False
    ):
        _dsml_debug(
            "stream-start",
            delegating_parser_id=id(self),
            delegating_parser_type=_parser_type(self),
            tool_parser_type=_parser_type(tool_parser),
            reasoning_parser_type=_parser_type(getattr(self, "_reasoning_parser", None)),
            tool_choice=getattr(request, "tool_choice", None),
            has_tools=bool(getattr(request, "tools", None)),
            prompt_token_ids_is_none=prompt_token_ids is None,
            reasoning_ended=state.reasoning_ended,
            engine_based=getattr(state, "engine_based", None),
        )
        self._deepseek_v4_debug_stream_announced = True

    raw_delta_has_dsml = "DSML" in delta_text or "<｜" in delta_text
    if not isinstance(tool_parser, DeepSeekV4ToolParser) and raw_delta_has_dsml:
        _dsml_debug(
            "deepseek-v4-parser-type-check-failed",
            delegating_parser_id=id(self),
            actual_tool_parser_type=_parser_type(tool_parser),
            delta_preview=_dsml_tag_preview(delta_text),
        )

    if isinstance(tool_parser, DeepSeekV4ToolParser):
        reasoning_parser = getattr(self, "_reasoning_parser", None)

        # With no reasoning parser there is no reasoning phase to wait for.
        if reasoning_parser is None:
            state.reasoning_ended = True

        # DeepSeek V4 servers normally configure both the reasoning parser and
        # the tool parser.  A response may nevertheless start directly with a
        # DSML tool call.  Hold back only a possible split start marker; once a
        # complete marker is present, route it to the tool parser before the
        # reasoning parser can expose it through ``delta.content``.
        is_auto_tool_request = bool(getattr(request, "tools", None)) and getattr(
            request, "tool_choice", None
        ) in (None, "auto")
        if reasoning_parser is not None and not state.reasoning_ended and is_auto_tool_request:
            probe_text = getattr(self, "_deepseek_v4_dsml_probe_text", "") + delta_text
            probe_token_ids = getattr(self, "_deepseek_v4_dsml_probe_token_ids", []) + list(
                delta_token_ids
            )
            start_token = tool_parser.tool_call_start_token
            start_idx = probe_text.find(start_token)

            if start_idx == -1:
                overlap = _partial_tag_overlap(probe_text, start_token)
                if overlap and not finished:
                    if not getattr(self, "_deepseek_v4_debug_probe_announced", False):
                        _dsml_debug(
                            "holding-possible-split-start",
                            delegating_parser_id=id(self),
                            overlap=overlap,
                            probe_preview=_dsml_tag_preview(probe_text),
                            reasoning_ended=state.reasoning_ended,
                        )
                        self._deepseek_v4_debug_probe_announced = True
                    self._deepseek_v4_dsml_probe_text = probe_text
                    self._deepseek_v4_dsml_probe_token_ids = probe_token_ids
                    return None

                # No DSML marker was found.  Flush ordinary text normally.  At
                # EOS, discard only the suffix that is still an exact partial
                # DSML marker, because it cannot be retracted after emission.
                if overlap:
                    probe_text = probe_text[:-overlap]
                delta_text = probe_text
                delta_token_ids = probe_token_ids
            else:
                _dsml_debug(
                    "complete-start-detected-before-reasoning",
                    delegating_parser_id=id(self),
                    start_idx=start_idx,
                    probe_preview=_dsml_tag_preview(probe_text),
                    reasoning_parser_type=_parser_type(reasoning_parser),
                )
                text_before_dsml = probe_text[:start_idx]
                if text_before_dsml:
                    prefix_delta = _original_delegating_parse_delta(
                        self,
                        text_before_dsml,
                        [],
                        request,
                        prompt_token_ids,
                        finished=False,
                    )

                # The prefix has already gone through reasoning extraction.
                # Start the tool phase with a clean text history so it receives
                # the DSML marker exactly once.
                state.previous_text = ""
                state.previous_token_ids = []
                state.reasoning_ended = True
                state.prompt_reasoning_checked = True
                state.tool_call_text_started = False
                delta_text = probe_text[start_idx:]
                delta_token_ids = probe_token_ids

            self._deepseek_v4_dsml_probe_text = ""
            self._deepseek_v4_dsml_probe_token_ids = []
            self._deepseek_v4_debug_probe_announced = False

    if not finished or not isinstance(tool_parser, DeepSeekV4ToolParser):
        delta_message = _merge_delta_messages(
            prefix_delta,
            _original_delegating_parse_delta(
                self,
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            ),
        )
        _debug_delegating_output(self, delta_message, tool_parser)
        returned_content = delta_message.content if delta_message is not None else None
        if finished:
            _dsml_debug(
                "stream-finished-non-deepseek-path",
                delegating_parser_id=id(self),
                returned_content_has_dsml=bool(
                    returned_content and "<｜DSML｜" in returned_content
                ),
            )
            self._deepseek_v4_debug_stream_announced = False
            self._deepseek_v4_debug_returned_content_probe = ""
            self._deepseek_v4_debug_content_leak_announced = False
        return delta_message

    delta_message = _original_delegating_parse_delta(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids,
        finished=False,
    )
    delta_message = _merge_delta_messages(prefix_delta, delta_message)
    finish_delta = tool_parser.finish_streaming(request)
    delta_message = _merge_delta_messages(delta_message, finish_delta)

    try:
        self._append_unstreamed_tool_args(delta_message)
        _debug_delegating_output(self, delta_message, tool_parser)
        returned_content = delta_message.content if delta_message is not None else None
        _dsml_debug(
            "stream-finished-deepseek-path",
            delegating_parser_id=id(self),
            returned_content_has_dsml=bool(
                returned_content and "<｜DSML｜" in returned_content
            ),
            returned_content_preview=_dsml_tag_preview(returned_content or ""),
            returned_tool_call_count=len(delta_message.tool_calls or [])
            if delta_message is not None
            else 0,
        )
        return delta_message
    finally:
        self._deepseek_v4_dsml_probe_text = ""
        self._deepseek_v4_dsml_probe_token_ids = []
        self._deepseek_v4_debug_probe_announced = False
        self._deepseek_v4_debug_stream_announced = False
        self._deepseek_v4_debug_returned_content_probe = ""
        self._deepseek_v4_debug_content_leak_announced = False
        tool_parser._reset_streaming_state()


# Backward-compatible monkey patches.
DeepSeekV4ToolParser._ensure_streaming_attrs = _ensure_streaming_attrs
DeepSeekV4ToolParser._function_name = _function_name
DeepSeekV4ToolParser._function_parameters = _function_parameters
DeepSeekV4ToolParser._convert_param_value = _convert_param_value
DeepSeekV4ToolParser._extract_param_name = _extract_param_name
DeepSeekV4ToolParser._get_param_config = _get_param_config
DeepSeekV4ToolParser._coerce_param_value = _coerce_param_value
DeepSeekV4ToolParser._repair_param_dict = _repair_param_dict
DeepSeekV4ToolParser._parse_invoke_params = _parse_invoke_params
DeepSeekV4ToolParser.extract_tool_calls = _patched_extract_tool_calls
DeepSeekV4ToolParser._reset_streaming_state = _reset_streaming_state
DeepSeekV4ToolParser._json_escape_string_content = _json_escape_string_content
DeepSeekV4ToolParser.drain_pending_tool_call_deltas = _drain_pending_tool_call_deltas
DeepSeekV4ToolParser._pop_pending_delta_message = _pop_pending_delta_message
DeepSeekV4ToolParser._queue_delta_message = _queue_delta_message
DeepSeekV4ToolParser._emit_tool_name_delta = _emit_tool_name_delta
DeepSeekV4ToolParser._emit_tool_args_delta = _emit_tool_args_delta
DeepSeekV4ToolParser._begin_streaming_tool_call = _begin_streaming_tool_call
DeepSeekV4ToolParser._append_param_prefix = _append_param_prefix
DeepSeekV4ToolParser._append_json_param_value = _append_json_param_value
DeepSeekV4ToolParser._append_raw_param_value = _append_raw_param_value
DeepSeekV4ToolParser._param_types_for_name = _param_types_for_name
DeepSeekV4ToolParser._can_stream_raw_param = _can_stream_raw_param
DeepSeekV4ToolParser._finish_buffered_param = _finish_buffered_param
DeepSeekV4ToolParser._should_buffer_wrapper_param = _should_buffer_wrapper_param
DeepSeekV4ToolParser._finish_buffered_wrapper_param = _finish_buffered_wrapper_param
DeepSeekV4ToolParser._close_streaming_tool_call = _close_streaming_tool_call
DeepSeekV4ToolParser._safe_content_len_before_tag_end = _safe_content_len_before_tag_end
DeepSeekV4ToolParser._process_streaming_buffer = _process_streaming_buffer
DeepSeekV4ToolParser.finish_streaming = _finish_streaming
DeepSeekV4ToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
DelegatingParser.parse_delta = _patched_delegating_parse_delta
_dsml_debug(
    "patch-installed",
    delegating_parse_delta_module=DelegatingParser.parse_delta.__module__,
    delegating_parse_delta_name=DelegatingParser.parse_delta.__name__,
)
