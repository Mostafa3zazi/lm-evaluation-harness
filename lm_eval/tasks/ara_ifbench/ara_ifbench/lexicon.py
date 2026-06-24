from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_lexicon(name: str) -> dict[str, list[str]]:
    path = _project_root() / "data" / "lexicons" / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: list(value) for key, value in payload.items()}
