from __future__ import annotations

import re

from .normalize import normalize


SENTENCE_ENDERS = {".", "!", "?", "؟"}
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+|[٠-٩]+)\s*$")
BARE_ORDERED_LIST_SENTENCE_RE = re.compile(r"^\s*(?:\d+|[٠-٩]+)[\.\)]\s*$")


def split_paragraphs(text: str, normalization_mode: str = "strict_surface") -> list[str]:
    normalized = normalize(text, normalization_mode).strip()
    if not normalized:
        return []
    return [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]


def split_lines(text: str, normalization_mode: str = "strict_surface") -> list[str]:
    normalized = normalize(text, normalization_mode)
    return [line.rstrip("\n") for line in normalized.splitlines() if line.strip()]


def split_sentences(text: str, normalization_mode: str = "strict_surface") -> list[str]:
    normalized = normalize(text, normalization_mode).strip()
    if not normalized:
        return []
    sentences: list[str] = []
    current: list[str] = []
    for idx, char in enumerate(normalized):
        current.append(char)
        if char in SENTENCE_ENDERS and _is_sentence_boundary(normalized, idx, current):
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return _merge_bare_ordered_list_sentences(sentences)


def _is_sentence_boundary(text: str, idx: int, current: list[str]) -> bool:
    char = text[idx]
    if char != ".":
        return True

    prev_char = text[idx - 1] if idx > 0 else ""
    next_char = text[idx + 1] if idx + 1 < len(text) else ""
    if prev_char.isdigit() and next_char.isdigit():
        return False

    prior_text = "".join(current[:-1])
    line_prefix = prior_text.splitlines()[-1] if prior_text.splitlines() else prior_text
    return not ORDERED_LIST_MARKER_RE.fullmatch(line_prefix)


def _merge_bare_ordered_list_sentences(sentences: list[str]) -> list[str]:
    merged: list[str] = []
    idx = 0
    while idx < len(sentences):
        current = sentences[idx]
        if BARE_ORDERED_LIST_SENTENCE_RE.fullmatch(current) and idx + 1 < len(sentences):
            merged.append(f"{current.rstrip()} {sentences[idx + 1].lstrip()}")
            idx += 2
            continue
        merged.append(current)
        idx += 1
    return merged
