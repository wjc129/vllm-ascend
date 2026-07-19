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

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Any: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class LoptConfig:
    enabled: bool = False
    thread_workers: int = 4
    min_chars: int = 32768
    chunk_chars: int = 32768
    overlap_chars: int = 512
    min_match_tokens: int = 2
    max_retries: int = 3
    verify: bool = False

    def __post_init__(self) -> None:
        if self.thread_workers < 1:
            raise ValueError("LoPT thread_workers must be at least 1")
        if self.min_chars < 1:
            raise ValueError("LoPT min_chars must be at least 1")
        if self.chunk_chars < 1:
            raise ValueError("LoPT chunk_chars must be at least 1")
        if not 0 < self.overlap_chars <= self.chunk_chars:
            raise ValueError("LoPT overlap_chars must be in [1, chunk_chars]")
        if self.min_match_tokens < 1:
            raise ValueError("LoPT min_match_tokens must be at least 1")
        if self.max_retries < 0:
            raise ValueError("LoPT max_retries cannot be negative")


@dataclass(frozen=True)
class TextChunk:
    index: int
    global_start: int
    global_end: int
    text: str


@dataclass(frozen=True)
class ChunkEncoding:
    index: int
    global_start: int
    global_end: int
    token_ids: tuple[int, ...]
    local_offsets: tuple[tuple[int, int], ...]
    global_offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OverlapMatch:
    left_start: int
    right_start: int
    token_count: int
    char_start: int
    char_end: int


class LoptError(RuntimeError):
    """Base exception for a safely recoverable LoPT failure."""


class ChunkEncodingError(LoptError):
    """A tokenizer failed to produce a valid offset-aware chunk encoding."""


class OverlapMatchError(LoptError):
    """Adjacent chunks could not be joined without ambiguity."""


def _split_overlapping(text: str, chunk_chars: int, overlap_chars: int) -> list[TextChunk]:
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    if not 0 < overlap_chars <= chunk_chars:
        raise ValueError("overlap_chars must be in [1, chunk_chars]")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars + overlap_chars)
        chunks.append(
            TextChunk(
                index=index,
                global_start=start,
                global_end=end,
                text=text[start:end],
            )
        )
        start += chunk_chars
        index += 1
    return chunks


def _as_token_ids(value: Any) -> tuple[int, ...]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ChunkEncodingError("tokenizer input_ids is not a sequence") from exc

    if values and isinstance(values[0], (list, tuple)):
        raise ChunkEncodingError("tokenizer unexpectedly returned batched input_ids")
    if any(not isinstance(token_id, Integral) for token_id in values):
        raise ChunkEncodingError("tokenizer returned a non-integer token ID")
    return tuple(int(token_id) for token_id in values)


def _as_offsets(value: Any, text_length: int) -> tuple[tuple[int, int], ...]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ChunkEncodingError("tokenizer offset_mapping is not a sequence") from exc

    offsets: list[tuple[int, int]] = []
    previous_start = 0
    previous_end = 0
    for raw_offset in values:
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise ChunkEncodingError("tokenizer returned an invalid offset pair")
        start, end = raw_offset
        if not isinstance(start, Integral) or not isinstance(end, Integral):
            raise ChunkEncodingError("tokenizer returned a non-integer offset")
        start = int(start)
        end = int(end)
        if not 0 <= start <= end <= text_length:
            raise ChunkEncodingError("tokenizer returned an out-of-range offset")
        if start < previous_start or end < previous_end:
            raise ChunkEncodingError("tokenizer offsets are not monotonic")
        offsets.append((start, end))
        previous_start = start
        previous_end = end
    return tuple(offsets)


def _encoding_value(encoded: Any, key: str) -> Any:
    try:
        return encoded[key]
    except (KeyError, TypeError):
        value = getattr(encoded, key, None)
        if value is None:
            raise ChunkEncodingError(f"tokenizer result does not contain {key}") from None
        return value


def _encode_chunk(tokenizer: _Tokenizer, chunk: TextChunk) -> ChunkEncoding:
    try:
        encoded = tokenizer(
            chunk.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception as exc:
        raise ChunkEncodingError(f"chunk {chunk.index} tokenization failed") from exc

    token_ids = _as_token_ids(_encoding_value(encoded, "input_ids"))
    local_offsets = _as_offsets(_encoding_value(encoded, "offset_mapping"), len(chunk.text))
    if len(token_ids) != len(local_offsets):
        raise ChunkEncodingError("input_ids and offset_mapping lengths differ")

    global_offsets = tuple(
        (chunk.global_start + start, chunk.global_start + end) for start, end in local_offsets
    )
    return ChunkEncoding(
        index=chunk.index,
        global_start=chunk.global_start,
        global_end=chunk.global_end,
        token_ids=token_ids,
        local_offsets=local_offsets,
        global_offsets=global_offsets,
    )


def _parallel_encode_chunks(
    tokenizer: _Tokenizer,
    chunks: list[TextChunk],
    executor: Executor,
) -> list[ChunkEncoding]:
    futures: list[Future[ChunkEncoding]] = [executor.submit(_encode_chunk, tokenizer, chunk) for chunk in chunks]
    try:
        encodings = [future.result() for future in futures]
    except Exception as exc:
        for future in futures:
            future.cancel()
        if isinstance(exc, LoptError):
            raise
        raise ChunkEncodingError("parallel chunk tokenization failed") from exc

    encodings.sort(key=lambda encoding: encoding.index)
    return encodings


def _record(encoding: ChunkEncoding, index: int) -> tuple[int, int, int]:
    start, end = encoding.global_offsets[index]
    return start, end, encoding.token_ids[index]


def _in_overlap(record: tuple[int, int, int], overlap_start: int, overlap_end: int) -> bool:
    start, end, _ = record
    return end > start and start >= overlap_start and end <= overlap_end


def _find_position_overlap(
    left: ChunkEncoding,
    right: ChunkEncoding,
    min_match_tokens: int,
) -> OverlapMatch | None:
    overlap_start = max(left.global_start, right.global_start)
    overlap_end = min(left.global_end, right.global_end)
    if overlap_start >= overlap_end:
        return None

    right_candidates: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for right_index in range(len(right.token_ids)):
        record = _record(right, right_index)
        if _in_overlap(record, overlap_start, overlap_end):
            right_candidates[record].append(right_index)

    matches: list[OverlapMatch] = []
    for left_index in range(len(left.token_ids)):
        first_record = _record(left, left_index)
        if not _in_overlap(first_record, overlap_start, overlap_end):
            continue
        for right_index in right_candidates.get(first_record, ()):
            token_count = 0
            char_end = first_record[1]
            while left_index + token_count < len(left.token_ids) and right_index + token_count < len(right.token_ids):
                left_record = _record(left, left_index + token_count)
                right_record = _record(right, right_index + token_count)
                if left_record != right_record or not _in_overlap(left_record, overlap_start, overlap_end):
                    break
                char_end = max(char_end, left_record[1])
                token_count += 1

            if token_count >= min_match_tokens and char_end > first_record[0]:
                matches.append(
                    OverlapMatch(
                        left_start=left_index,
                        right_start=right_index,
                        token_count=token_count,
                        char_start=first_record[0],
                        char_end=char_end,
                    )
                )

    if not matches:
        return None

    def rank(match: OverlapMatch) -> tuple[int, int, int]:
        safety_margin = min(match.char_start - overlap_start, overlap_end - match.char_end)
        return match.token_count, safety_margin, match.char_end - match.char_start

    matches.sort(key=rank, reverse=True)
    best = matches[0]
    if len(matches) > 1 and rank(matches[1]) == rank(best):
        return None
    return best


def _merge_pair(left: ChunkEncoding, right: ChunkEncoding, min_match_tokens: int) -> ChunkEncoding:
    match = _find_position_overlap(left, right, min_match_tokens)
    if match is None:
        raise OverlapMatchError(f"no unambiguous position-aware overlap between chunks {left.index} and {right.index}")

    left_end = match.left_start + match.token_count
    right_end = match.right_start + match.token_count
    token_ids = left.token_ids[:left_end] + right.token_ids[right_end:]
    global_offsets = left.global_offsets[:left_end] + right.global_offsets[right_end:]
    if not token_ids or len(token_ids) != len(global_offsets):
        raise OverlapMatchError("merged chunk encoding is empty or inconsistent")

    global_start = min(left.global_start, right.global_start)
    local_offsets = tuple((start - global_start, end - global_start) for start, end in global_offsets)
    return ChunkEncoding(
        index=right.index,
        global_start=global_start,
        global_end=max(left.global_end, right.global_end),
        token_ids=token_ids,
        local_offsets=local_offsets,
        global_offsets=global_offsets,
    )


def _merge_chunk_encodings(encodings: list[ChunkEncoding], min_match_tokens: int) -> ChunkEncoding:
    if not encodings:
        raise OverlapMatchError("cannot merge an empty chunk list")

    merged = encodings[0]
    for encoding in encodings[1:]:
        merged = _merge_pair(merged, encoding, min_match_tokens)
    return merged


class LosslessParallelTokenizer:
    """Offset-aware overlapping parallel tokenization with safe fallback."""

    _SUPPORTED_ENCODE_KWARGS = frozenset({"add_special_tokens", "truncation", "max_length"})

    def __init__(
        self,
        tokenizer: _Tokenizer,
        config: LoptConfig,
        executor: Executor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self._executor = executor or ThreadPoolExecutor(
            max_workers=config.thread_workers,
            thread_name_prefix="vllm-ascend-lopt",
        )
        self._owns_executor = executor is None
        self._state_lock = threading.Lock()
        self._shutdown = False
        self._compatibility: dict[tuple[bool, bool, int | None], bool] = {}
        self._warned_fallbacks: set[str] = set()

    def can_use(self, text: str, encode_kwargs: dict[str, Any] | None = None) -> bool:
        encode_kwargs = encode_kwargs or {}
        with self._state_lock:
            is_shutdown = self._shutdown
        if is_shutdown or not self.config.enabled:
            return False
        if getattr(self.tokenizer, "is_fast", False) is not True:
            return False
        if len(text) < self.config.min_chars:
            return False
        if len(text) <= self.config.chunk_chars + self.config.overlap_chars:
            return False
        if set(encode_kwargs) - self._SUPPORTED_ENCODE_KWARGS:
            return False
        if encode_kwargs.get("truncation", False) and not hasattr(self.tokenizer, "prepare_for_model"):
            return False
        if not callable(getattr(self.tokenizer, "encode", None)) or not callable(self.tokenizer):
            return False
        return self._is_compatible(encode_kwargs)

    def encode(self, text: str, **encode_kwargs: Any) -> list[int]:
        if not self.can_use(text, encode_kwargs):
            return self._standard_encode(text, encode_kwargs)

        started_at = time.perf_counter()
        try:
            token_ids, attempts, chunk_count = self._encode_with_retries(text, encode_kwargs)
            if self.config.verify:
                standard_ids = self._standard_encode(text, encode_kwargs)
                if token_ids != standard_ids:
                    self._log_fallback_once("verification", "LoPT verification failed; using standard tokenization")
                    return standard_ids
        except Exception as exc:
            self._log_fallback_once(
                type(exc).__name__,
                "LoPT tokenization failed (%s); using standard tokenization",
                exc,
            )
            return self._standard_encode(text, encode_kwargs)

        logger.debug(
            "LoPT tokenized %d characters into %d tokens using %d chunks and %d attempt(s) in %.3f ms",
            len(text),
            len(token_ids),
            chunk_count,
            attempts,
            (time.perf_counter() - started_at) * 1000,
        )
        return token_ids

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def _standard_encode(self, text: str, encode_kwargs: dict[str, Any]) -> list[int]:
        return list(self.tokenizer.encode(text, **encode_kwargs))

    def _log_fallback_once(self, key: str, message: str, *args: Any) -> None:
        with self._state_lock:
            first_occurrence = key not in self._warned_fallbacks
            self._warned_fallbacks.add(key)
        if first_occurrence:
            logger.warning(message, *args)
        else:
            logger.debug(message, *args)

    def _is_compatible(self, encode_kwargs: dict[str, Any]) -> bool:
        add_special_tokens = bool(encode_kwargs.get("add_special_tokens", True))
        truncation = bool(encode_kwargs.get("truncation", False))
        max_length = encode_kwargs.get("max_length")
        key = (add_special_tokens, truncation, max_length)
        with self._state_lock:
            cached = self._compatibility.get(key)
        if cached is not None:
            return cached

        probe = "LoPT compatibility probe: ASCII 中文 🙂 e\u0301 0123456789. " * 2
        probe_kwargs: dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "max_length": None,
        }
        if truncation:
            probe_kwargs["max_length"] = min(max_length, 16) if isinstance(max_length, int) else 16

        try:
            chunk = TextChunk(index=0, global_start=0, global_end=len(probe), text=probe)
            content_ids = _encode_chunk(self.tokenizer, chunk).token_ids
            prepared_ids = self._prepare_for_model(content_ids, probe_kwargs)
            standard_ids = self._standard_encode(probe, probe_kwargs)
            compatible = prepared_ids == standard_ids
        except Exception:
            compatible = False

        with self._state_lock:
            self._compatibility[key] = compatible
        if not compatible:
            self._log_fallback_once(
                f"compatibility-{key}",
                "LoPT compatibility probe failed for add_special_tokens=%s, truncation=%s; using standard tokenization",
                add_special_tokens,
                truncation,
            )
        return compatible

    def _encode_with_retries(
        self,
        text: str,
        encode_kwargs: dict[str, Any],
    ) -> tuple[list[int], int, int]:
        chunk_chars = self.config.chunk_chars
        last_error: OverlapMatchError | None = None

        for attempt in range(self.config.max_retries + 1):
            chunks = _split_overlapping(text, chunk_chars, self.config.overlap_chars)
            if len(chunks) < 2:
                raise LoptError("prompt does not produce multiple chunks")

            encodings = _parallel_encode_chunks(self.tokenizer, chunks, self._executor)
            try:
                merged = _merge_chunk_encodings(encodings, self.config.min_match_tokens)
            except OverlapMatchError as exc:
                last_error = exc
                logger.debug(
                    "LoPT overlap matching attempt %d failed with chunk_chars=%d: %s",
                    attempt + 1,
                    chunk_chars,
                    exc,
                )
                chunk_chars *= 2
                continue

            token_ids = self._prepare_for_model(merged.token_ids, encode_kwargs)
            return token_ids, attempt + 1, len(chunks)

        raise last_error or OverlapMatchError("LoPT overlap matching failed")

    def _prepare_for_model(self, token_ids: tuple[int, ...], encode_kwargs: dict[str, Any]) -> list[int]:
        add_special_tokens = bool(encode_kwargs.get("add_special_tokens", True))
        truncation = encode_kwargs.get("truncation", False)
        max_length = encode_kwargs.get("max_length")

        prepare_for_model = getattr(self.tokenizer, "prepare_for_model", None)
        if callable(prepare_for_model):
            prepared = prepare_for_model(
                list(token_ids),
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return list(_as_token_ids(_encoding_value(prepared, "input_ids")))

        if truncation:
            raise LoptError("tokenizer cannot reproduce truncation from pre-tokenized IDs")
        if not add_special_tokens:
            return list(token_ids)

        build_inputs = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
        if not callable(build_inputs):
            raise LoptError("tokenizer cannot add special tokens to merged IDs")
        return list(build_inputs(list(token_ids)))
