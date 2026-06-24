from __future__ import annotations

from itertools import zip_longest
from typing import Any

from .rollout import ROLLOUT_TYPE_IFBENCH_MULTITURN
from .specs import TEST_SPEC_INDEX

COMMON_PARAM_ALIASES = {
    "n": ("N",),
    "phrase": ("end_phrase",),
    "word": ("first_word",),
    "paragraph_index": ("nth_paragraph",),
    "word_index": ("m",),
    "source_text": ("reference_text", "prompt_to_repeat"),
    "sep": ("section_spliter",),
}

INSTRUCTION_ID_ALIASES = {
    "count:numbers": "count.numbers_exact",
    "count:unique_word_count": "count.unique_word_count",
}
for spec_id in TEST_SPEC_INDEX:
    INSTRUCTION_ID_ALIASES.setdefault(spec_id, spec_id)
    INSTRUCTION_ID_ALIASES.setdefault(spec_id.replace(".", ":"), spec_id)


def normalize_example_payload(payload: dict[str, Any]) -> dict[str, Any]:
    example_id = payload.get("example_id", payload.get("key", payload.get("id")))
    if example_id is None:
        raise ValueError("Example payload must include example_id, key, or id.")

    rendered_prompt = payload.get("prompt")
    prompt_ar = payload.get("prompt_ar", rendered_prompt or "")
    raw_instruction_ids = payload.get("instruction_id_list", [])
    instruction_id_list = [normalize_instruction_id(item) for item in raw_instruction_ids]
    raw_kwargs_list = payload.get("kwargs_list", payload.get("kwargs", []))
    kwargs_list = [
        normalize_kwargs(instruction_id, raw_kwargs or {})
        for instruction_id, raw_kwargs in zip_longest(instruction_id_list, raw_kwargs_list, fillvalue={})
    ]
    raw_messages = payload.get("messages")
    messages = normalize_messages(raw_messages) if raw_messages is not None else None
    rollout_type = normalize_rollout_type(payload.get("rollout_type"))

    return {
        "example_id": str(example_id),
        "split": payload.get("split", "test_private"),
        "source_dataset": payload.get("source_dataset", "wildchat"),
        "source_row_id": str(
            payload.get("source_row_id", payload.get("wildchat_id", payload.get("source_id", example_id)))
        ),
        "prompt_ar": prompt_ar,
        "prompt": rendered_prompt,
        "messages": messages,
        "rollout_type": rollout_type,
        "instruction_id_list": instruction_id_list,
        "kwargs_list": kwargs_list,
        "prompt_type": payload.get("prompt_type"),
        "dialect_tag": payload.get("dialect_tag"),
    }


def normalize_instruction_id(identifier: str) -> str:
    normalized = identifier.strip().lower()
    mapped = INSTRUCTION_ID_ALIASES.get(normalized)
    if mapped:
        return mapped
    if normalized in TEST_SPEC_INDEX:
        return normalized
    raise KeyError(f"Unknown instruction id: {identifier}")


def normalize_kwargs(instruction_id: str, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    spec = TEST_SPEC_INDEX[instruction_id]
    cleaned = {key: value for key, value in raw_kwargs.items() if value is not None}
    normalized: dict[str, Any] = {}
    for param in spec.params_schema:
        for candidate_key in _candidate_keys(param):
            if candidate_key in cleaned:
                normalized[param] = cleaned[candidate_key]
                break
    return normalized


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list when provided.")
    normalized: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a dict.")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if not role:
            raise ValueError("Each message must include a non-empty role.")
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_rollout_type(raw_rollout_type: Any) -> str | None:
    if raw_rollout_type is None:
        return None
    normalized = str(raw_rollout_type).strip().lower().replace("-", "_")
    aliases = {
        "ifbench_multiturn_sequential": ROLLOUT_TYPE_IFBENCH_MULTITURN,
        "ifbench_multiturn_rewrite": ROLLOUT_TYPE_IFBENCH_MULTITURN,
        "paper_multiturn": ROLLOUT_TYPE_IFBENCH_MULTITURN,
    }
    return aliases.get(normalized, str(raw_rollout_type).strip())


def _candidate_keys(param: str) -> tuple[str, ...]:
    candidates = [param]
    candidates.extend(COMMON_PARAM_ALIASES.get(param, ()))
    return tuple(candidates)
