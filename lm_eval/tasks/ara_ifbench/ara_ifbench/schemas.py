from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_name: str
    mode: str
    taxonomy_size: int
    max_constraints_per_turn: int
    allowed_constraints_per_turn: tuple[int, ...]


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    category: str
    split_role: str
    description_templates_ar: tuple[str, ...]
    params_schema: dict[str, str]
    normalization_mode: str
    conflict_tags: tuple[str, ...]
    single_only: bool
    handler: str
    loose_enabled: bool = True


@dataclass
class ExampleRecord:
    example_id: str
    split: str
    source_dataset: str
    source_row_id: str
    prompt_ar: str
    instruction_id_list: list[str]
    kwargs_list: list[dict[str, Any]]
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    rollout_type: str | None = None
    prompt_type: str | None = None
    dialect_tag: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExampleRecord":
        return cls(**payload)


@dataclass
class VerifierResult:
    instruction_id: str
    passed_strict: bool
    passed_loose: bool
    failure_code: str
    debug_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleEvalResult:
    example_id: str
    prompt_passed_strict: bool
    prompt_passed_loose: bool
    verifier_results: list[VerifierResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verifier_results"] = [item.to_dict() for item in self.verifier_results]
        return payload


@dataclass
class EvalSummary:
    prompt_level_strict: float
    prompt_level_loose: float
    instruction_level_strict: float
    instruction_level_loose: float
    per_category: dict[str, dict[str, float]]
    per_family: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
