from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets

from lm_eval.tasks.ara_ifbench.ara_ifbench.example_normalize import (
    normalize_example_payload,
)
from lm_eval.tasks.ara_ifbench.ara_ifbench.normalize import loose_variants
from lm_eval.tasks.ara_ifbench.ara_ifbench.prompting import render_example_prompt
from lm_eval.tasks.ara_ifbench.ara_ifbench.rollout import (
    ROLLOUT_TYPE_IFBENCH_MULTITURN,
)
from lm_eval.tasks.ara_ifbench.ara_ifbench.schemas import ExampleRecord
from lm_eval.tasks.ara_ifbench.ara_ifbench.specs import TEST_SPEC_INDEX, TEST_SPECS
from lm_eval.tasks.ara_ifbench.ara_ifbench.verifiers import verify_constraint


DEFAULT_SPLIT_NAME = "test"
DEFAULT_SMOKE_DATA = Path(__file__).with_name("examples") / "smoke_test.jsonl"
CATEGORY_NAMES = tuple(dict.fromkeys(spec.category for spec in TEST_SPECS))


def load_dataset(
    input_data: str | list[str] | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    repo_id: str | None = None,
    dataset_repo_id: str | None = None,
    config_name: str | None = None,
    hf_dataset_kwargs: dict[str, Any] | None = None,
    hf_split: str = DEFAULT_SPLIT_NAME,
    split_name: str = DEFAULT_SPLIT_NAME,
    **kwargs,
):
    """Load Ara-IFBench records from JSONL, HF Hub, or the bundled smoke set."""
    records: list[dict[str, Any]]
    source_files = data_files if data_files is not None else input_data
    repo = repo_id or dataset_repo_id

    if source_files is not None:
        records = _read_json_records(_flatten_data_files(source_files))
    elif repo:
        records = [
            dict(item)
            for item in datasets.load_dataset(
                repo,
                config_name,
                split=hf_split,
                **(hf_dataset_kwargs or {}),
            )
        ]
    else:
        records = _read_json_records([DEFAULT_SMOKE_DATA])

    normalized_records = [_prepare_record(record) for record in records]
    return {split_name: datasets.Dataset.from_list(normalized_records)}


def doc_to_text(doc: dict[str, Any]) -> str:
    record = _example_record_from_doc(doc)
    if record.rollout_type == ROLLOUT_TYPE_IFBENCH_MULTITURN:
        raise ValueError(
            "Ara-IFBench sequential multi-turn examples require a model-generated "
            "first turn and are not supported by this lm-eval task. Use a "
            "single-turn or pre-rendered manifest for lm-evaluation-harness."
        )
    if record.messages is not None:
        raise ValueError(
            "Ara-IFBench prefilled message examples are not supported by this "
            "YAML task because lm-eval task prompts are rendered as a single "
            "string. Use a single-turn or pre-rendered manifest."
        )
    return render_example_prompt(record)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    record = _example_record_from_doc(doc)
    response = results[0] if results else ""
    strict_results: list[bool] = []
    loose_results: list[bool] = []
    category_strict_results: dict[str, list[bool]] = {
        category: [] for category in CATEGORY_NAMES
    }
    category_loose_results: dict[str, list[bool]] = {
        category: [] for category in CATEGORY_NAMES
    }

    for instruction_id, kwargs in zip(record.instruction_id_list, record.kwargs_list):
        spec = TEST_SPEC_INDEX[instruction_id]
        clean_kwargs = _clean_kwargs(kwargs)
        strict_passed, _, _ = verify_constraint(
            spec,
            response,
            clean_kwargs,
            match_mode="strict",
        )
        loose_passed = strict_passed
        if not loose_passed and spec.loose_enabled:
            for candidate in loose_variants(response):
                candidate_passed, _, _ = verify_constraint(
                    spec,
                    candidate,
                    clean_kwargs,
                    match_mode="loose",
                )
                if candidate_passed:
                    loose_passed = True
                    break

        strict_results.append(strict_passed)
        loose_results.append(loose_passed)
        category_strict_results[spec.category].append(strict_passed)
        category_loose_results[spec.category].append(loose_passed)

    metrics = {
        "prompt_level_strict_acc": all(strict_results),
        "inst_level_strict_acc": strict_results,
        "prompt_level_loose_acc": all(loose_results),
        "inst_level_loose_acc": loose_results,
    }
    for category in CATEGORY_NAMES:
        metrics[f"category_{category}_strict_acc"] = category_strict_results[
            category
        ]
        metrics[f"category_{category}_loose_acc"] = category_loose_results[category]
    return metrics


def agg_inst_level_acc(items: list[list[bool]]) -> float:
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items) if flat_items else float("nan")


def _prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_example_payload(record)
    example = ExampleRecord.from_dict(normalized)
    normalized["prompt"] = render_example_prompt(example)
    normalized["key"] = normalized["example_id"]
    normalized["kwargs"] = normalized["kwargs_list"]
    return normalized


def _example_record_from_doc(doc: dict[str, Any]) -> ExampleRecord:
    return ExampleRecord.from_dict(normalize_example_payload(dict(doc)))


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _flatten_data_files(
    data_files: str | list[str] | dict[str, str | list[str]],
) -> list[Path]:
    if isinstance(data_files, dict):
        items: list[str] = []
        for value in data_files.values():
            if isinstance(value, list):
                items.extend(value)
            else:
                items.append(value)
    elif isinstance(data_files, list):
        items = data_files
    else:
        items = [data_files]
    return [Path(item).expanduser() for item in items]


def _read_json_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Ara-IFBench data file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                records.extend(json.loads(line) for line in handle if line.strip())
            else:
                payload = json.load(handle)
                if isinstance(payload, list):
                    records.extend(payload)
                else:
                    records.append(payload)
    return records
