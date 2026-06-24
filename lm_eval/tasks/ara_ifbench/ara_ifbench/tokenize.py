from __future__ import annotations

import re

from .closed_lists import normalized_token_for_match
from .normalize import normalize


def whitespace_tokens(text: str, normalization_mode: str = "strict_surface") -> list[str]:
    normalized = normalize(text, normalization_mode).strip()
    if not normalized:
        return []
    return re.split(r"\s+", normalized)


def comparable_tokens(text: str, normalization_mode: str = "strict_surface") -> list[str]:
    return [
        token
        for token in (normalized_token_for_match(item) for item in whitespace_tokens(text, normalization_mode))
        if token
    ]


def token_count(text: str, normalization_mode: str = "strict_surface") -> int:
    return len(whitespace_tokens(text, normalization_mode))


def unique_token_count(text: str, normalization_mode: str = "strict_surface") -> int:
    return len(set(comparable_tokens(text, normalization_mode)))
