from __future__ import annotations

import re
import unicodedata


_MARKUP_PATTERNS = (
    (re.compile(r"</?(?:i|b|em|strong)>", re.IGNORECASE), ""),
    (re.compile(r"(\*\*|\*|__|_)(.*?)\1"), r"\2"),
)


def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_formatting_wrappers(text: str) -> str:
    output = text
    for pattern, repl in _MARKUP_PATTERNS:
        output = pattern.sub(repl, output)
    return output


def strict_surface(text: str) -> str:
    return to_nfc(text)


def soft_surface(text: str) -> str:
    return collapse_whitespace(strip_formatting_wrappers(to_nfc(text)))


def normalize(text: str, mode: str) -> str:
    if mode == "strict_surface":
        return strict_surface(text)
    if mode == "soft_surface":
        return soft_surface(text)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def loose_variants(text: str) -> list[str]:
    base = soft_surface(text)
    lines = [line for line in to_nfc(text).splitlines() if line.strip()]
    variants = [base]
    if lines:
        variants.append(soft_surface("\n".join(lines[1:])))
        variants.append(soft_surface("\n".join(lines[:-1])))
    if len(lines) > 1:
        variants.append(soft_surface("\n".join(lines[1:-1])))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped
