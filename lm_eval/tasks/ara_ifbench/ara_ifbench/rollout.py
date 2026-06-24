from __future__ import annotations

from .schemas import ExampleRecord

ROLLOUT_TYPE_SINGLE_TURN = "single_turn"
ROLLOUT_TYPE_PREFILLED_MESSAGES = "prefilled_messages"
ROLLOUT_TYPE_IFBENCH_MULTITURN = "ifbench_multiturn_sequential"


def resolve_rollout_type(example: ExampleRecord) -> str:
    if example.rollout_type:
        return example.rollout_type
    if example.messages is not None:
        return ROLLOUT_TYPE_PREFILLED_MESSAGES
    return ROLLOUT_TYPE_SINGLE_TURN


def validate_ifbench_multiturn_example(example: ExampleRecord) -> None:
    if example.messages is not None:
        raise ValueError(
            "Paper-style IFBench multi-turn examples must not include a prefilled assistant turn in messages."
        )
    if not example.prompt_ar.strip():
        raise ValueError("Paper-style IFBench multi-turn examples require a non-empty turn-1 user prompt in prompt_ar.")
    if not example.prompt or not example.prompt.strip():
        raise ValueError("Paper-style IFBench multi-turn examples require a non-empty follow-up user prompt in prompt.")


def build_multiturn_rewrite_messages(
    *,
    initial_prompt: str,
    assistant_response: str,
    followup_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": followup_prompt},
    ]
