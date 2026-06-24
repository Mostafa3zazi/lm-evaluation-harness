"""eval-360 `mmlu_zeroshot` scoring, reimplemented as an lm-eval filter.

This task reproduces LLM360 eval-360's `mmlu_zeroshot` benchmark inside the
jais-3 lm-eval harness. eval-360 scores multiple-choice in two stages:

  1. a *parse* stage runs the ``mc_answer`` parser on each generation, producing
     a single letter A-E (or None);
  2. the ``multiple_choice`` grader runs ``_extract_choice_symbol`` on that parse
     (falling back to the raw generation when the parse is None), i.e. it takes
     the FIRST alphabetic char (1-5 -> A-E), and compares that letter to the
     normalized ground-truth letter.

Unlike the earlier ``mmlu_us_gen`` task -- which only *approximated* the parse
stage with a "take the text after the last 'answer is'" heuristic because the
real parser was unavailable -- this file copies eval-360's actual ``mc_answer``
parser (Eval360-V2 ``scheduler/grader/base_parsers.py``) and ``multiple_choice``
grader helpers (``scheduler/grader/multiple_choice.py``) VERBATIM, so the parse
and grade stages are identical to eval-360.

The filter emits the picked letter; it is compared against a bare-letter
``doc_to_target`` via ``exact_match`` (ignore_case/ignore_punctuation), which
reproduces eval-360's ``expected_symbol == picked`` check.
"""

from __future__ import annotations

import re
from collections import deque

from lm_eval.api.filter import Filter


# ===========================================================================
# eval-360 `mc_answer` parser  (verbatim: scheduler/grader/base_parsers.py)
# ===========================================================================

def _normalize_mc_match(value: str) -> str:
    candidate = value.strip().upper()
    if candidate in {"1", "2", "3", "4", "5"}:
        return "ABCDE"[int(candidate) - 1]
    return candidate


_MC_EMPHASIS = r"(?:\*\*|__|`)?"
_MC_CHOICE = r"([A-E1-5])"
_MC_LINE_SCAN_LIMIT = 96
_MC_TAIL_SCAN_CHARS = 4096
_THINK_CLOSE_RE = re.compile(r"</think[^>]*>", flags=re.IGNORECASE)

_MC_TAG_PATTERNS = (
    re.compile(
        rf"<(?:final\s*)?answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</(?:final\s*)?answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"<answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\\(?:boxed|fbox)\s*\{{\s*{_MC_CHOICE}\s*\}}",
        flags=re.IGNORECASE,
    ),
)

_MC_STRONG_LINE_PATTERNS = (
    re.compile(
        rf"^\s*(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
)

_MC_WEAK_LINE_PATTERNS = (
    re.compile(
        rf"\b(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:match(?:es|ed)?|coincides?\s+with|corresponds?\s+to|gives?|yields?|confirms?)\s+(?:option|choice)\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bonly\s+(?:option\s*)?{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:option|choice)\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s+(?:is|would\s+be|remains?|works?|fits?|correct|best|right|compatible|consistent|larger)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s+is\s+(?:correct|best|right|compatible|consistent)\b",
        flags=re.IGNORECASE,
    ),
)


def _mc_match_last(pattern: re.Pattern, text: str):
    last = None
    for match in pattern.finditer(text):
        last = match
    if last is None:
        return None
    return _normalize_mc_match(last.group(1))


def _collect_mc_scan_lines(text: str):
    first = None
    trailing = deque(maxlen=_MC_LINE_SCAN_LIMIT)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if first is None:
            first = line
        trailing.append(line)
    return first, list(trailing)


def _extract_mc_from_line(line: str, *, allow_weak: bool):
    for pattern in _MC_TAG_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    for pattern in _MC_STRONG_LINE_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    if not allow_weak:
        return None

    for pattern in _MC_WEAK_LINE_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    return None


def _extract_mc_letter(text: str):
    first_line, trailing_lines = _collect_mc_scan_lines(text)

    if first_line is not None:
        first_line_match = _extract_mc_from_line(first_line, allow_weak=False)
        if first_line_match is not None:
            return first_line_match

    for line in reversed(trailing_lines):
        line_match = _extract_mc_from_line(line, allow_weak=True)
        if line_match is not None:
            return line_match

    # Some models inline the answer tag or boxed answer inside a paragraph. Keep
    # this search bounded to the tail, where the final answer typically appears.
    tail = text[-_MC_TAIL_SCAN_CHARS:]
    for pattern in _MC_TAG_PATTERNS:
        tail_match = _mc_match_last(pattern, tail)
        if tail_match is not None:
            return tail_match

    for pattern in _MC_WEAK_LINE_PATTERNS:
        tail_match = _mc_match_last(pattern, tail)
        if tail_match is not None:
            return tail_match

    return None


def _mc_answer_parser(generation: str):
    if generation is None:
        return None

    text = generation.strip()
    if not text:
        return None

    # In thinking-mode outputs, prefer the portion after the final </think> tag.
    candidate = text
    last_close = None
    for match in _THINK_CLOSE_RE.finditer(text):
        last_close = match
    if last_close is not None:
        candidate = text[last_close.end():].strip()

    extracted = _extract_mc_letter(candidate)
    if extracted is not None:
        return extracted

    # Fallback to the full text if extraction after </think> fails.
    return _extract_mc_letter(text)


# ===========================================================================
# eval-360 `multiple_choice` grade stage  (verbatim: grader/multiple_choice.py)
# ===========================================================================

_MC_DIGIT_TO_LETTER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}


def _normalize_choice_symbol(value):
    if value is None:
        return None
    candidate = value.strip().upper()
    if not candidate:
        return None
    return _MC_DIGIT_TO_LETTER.get(candidate, candidate)


def _extract_choice_symbol(sampled):
    if sampled is None:
        return None

    stripped = sampled.strip()
    if not stripped:
        return None

    letter = re.search(r"[A-Za-z]", stripped)
    if letter:
        return letter.group(0).upper()

    digit = re.search(r"[1-5]", stripped)
    if digit:
        return _MC_DIGIT_TO_LETTER[digit.group(0)]

    return stripped[0].upper()


# ===========================================================================
# lm-eval filter tying the two stages together
# ===========================================================================

class Eval360MCAnswerFilter(Filter):
    """Replicates eval-360 `mmlu_zeroshot` grading as an lm-eval filter.

    Per response: run the real `mc_answer` parse stage; fall back to the raw
    generation when the parse is None (mirrors the grader's
    ``parsed_generations[i] is None`` branch); then apply ``_extract_choice_symbol``.
    The emitted symbol is compared against a bare-letter ``doc_to_target`` via
    ``exact_match``, reproducing eval-360's ``expected_symbol == picked`` check.
    """

    def apply(self, resps, docs):
        filtered_resps = []
        for resp_list in resps:
            picks = []
            for resp in resp_list:
                parsed = _mc_answer_parser(resp)
                sampled = parsed if parsed is not None else resp
                picked = _extract_choice_symbol(sampled)
                # picked is None only for empty output; emit a token that can
                # never match a gold letter so it scores as incorrect, matching
                # eval-360's `picked is not None and ...` guard.
                picks.append(picked if picked is not None else "[invalid]")
            filtered_resps.append(picks)
        return filtered_resps
