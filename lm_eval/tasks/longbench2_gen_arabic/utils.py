"""Helpers for the generative (``generate_until``) Arabic LongBench v2 suite.

Each task reads a pre-filtered Arabic JSON file whose gold ``answer`` is an
option letter ("A"-"D") and whose options live in ``choice_A``..``choice_D``.
The model is prompted (in Arabic) to emit its choice as a bracketed *English*
letter, e.g. ``[[B]]``, so the labels line up with the gold answer.
``extract_bracketed_label`` pulls that letter out of the raw generation (used as
a ``custom`` filter) and ``exact_match_normalized_label`` scores it.
"""

import re


LABELS = ["A", "B", "C", "D"]


def add_options_with_text(dataset):
    """``process_docs`` hook: attach a formatted ``options_with_text`` field.

    The four options live in ``choice_A``..``choice_D``. This bundles them into
    a single human-readable block so the downstream LLM-as-judge step
    (``judge_vllm.py``) can read ``doc["options_with_text"]`` straight from the
    written samples without re-running inference. It does not affect prompting
    or the exact-match scoring.
    """

    def _add_field(doc):
        doc["options_with_text"] = (
            f"A. {doc['choice_A']}\n"
            f"B. {doc['choice_B']}\n"
            f"C. {doc['choice_C']}\n"
            f"D. {doc['choice_D']}"
        )
        return doc

    return dataset.map(_add_field)


def add_options_with_text_en(dataset):
    """English counterpart of :func:`add_options_with_text`.

    Uses the ``choice_*_en`` fields so the judge sees English option text for
    the English LongBench tasks.
    """

    def _add_field(doc):
        doc["options_with_text"] = (
            f"A. {doc['choice_A_en']}\n"
            f"B. {doc['choice_B_en']}\n"
            f"C. {doc['choice_C_en']}\n"
            f"D. {doc['choice_D_en']}"
        )
        return doc

    return dataset.map(_add_field)

# Match tiers, tried in order; within a tier the *last* match wins so that a
# model which reasons before concluding is scored on its final commitment.
_BRACKET_RE = re.compile(r"\[\[\s*([A-Da-d])\s*\]\]")
_PAREN_RE = re.compile(r"\(\s*([A-Da-d])\s*\)")
_LONE_RE = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")


def _extract_one(text: str) -> str:
    """Extract a single option letter (A-D) from one generation."""
    text = str(text)
    for pattern in (_BRACKET_RE, _PAREN_RE, _LONE_RE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return text.strip().upper()


def extract_bracketed_label(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """``custom`` filter: reduce each raw generation to its option letter."""
    return [[_extract_one(r) for r in resp] for resp, _ in zip(resps, docs)]


def _normalize_label(value) -> str:
    """Normalize a gold/predicted answer to a canonical option letter."""
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return str(value)
    if isinstance(value, int):
        return LABELS[value] if 0 <= value < len(LABELS) else str(value)
    s = str(value).strip()
    if s.isdigit():
        i = int(s)
        return LABELS[i] if 0 <= i < len(LABELS) else s
    return _extract_one(s)


def exact_match_normalized_label(references, predictions) -> dict:
    """Exact match after normalizing both sides to an option letter.

    The gold ``answer`` is an option letter and the prediction has already been
    reduced to a letter by the ``extract_answer`` filter; normalizing both makes
    the comparison robust. Reported under ``exact_match`` so the group-level
    ``aggregate_metric_list`` can roll it up.
    """
    gold = _normalize_label(references[0])
    pred = _normalize_label(predictions[0])
    return {"exact_match": 1.0 if gold == pred else 0.0}
