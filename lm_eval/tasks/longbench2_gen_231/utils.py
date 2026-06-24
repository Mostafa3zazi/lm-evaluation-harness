"""Helpers for the generative (``generate_until``) variant of LongBench v2.

The dataset (``recursal/longbench-v2``) stores the gold ``answer`` as an integer
index into the 4-element ``choices`` list, while the model is prompted to emit its
choice as a bracketed option letter, e.g. ``[[B]]``. ``extract_bracketed_label``
pulls that letter out of the raw generation (used as a ``custom`` filter) and
``exact_match_normalized_label`` scores it against the normalized gold label.
"""

import re


LABELS = ["A", "B", "C", "D"]

# Match tiers, tried in order; within a tier the *last* match wins so that a
# model which reasons before concluding ("... so the answer is [[C]]") is scored
# on its final commitment rather than an earlier aside.
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

    The gold ``answer`` may be an option letter (local ``longbench_v2.json``) or
    an integer index cast to its string form (the recursal HF mirror), while the
    prediction has already been reduced to a letter by the ``extract_answer``
    filter; normalizing both makes the comparison robust to either
    representation. Reported under ``exact_match`` so the group-level
    ``aggregate_metric_list`` configs can roll it up.
    """
    gold = _normalize_label(references[0])
    pred = _normalize_label(predictions[0])
    return {"exact_match": 1.0 if gold == pred else 0.0}


# ---------------------------------------------------------------------------
# Per-task document selection.
#
# The local ``longbench_v2.json`` is a single file holding all 503 questions,
# so each leaf task carves out its slice by (domain, sub_domain). The filter
# callables below are referenced from the leaf YAMLs as
# ``process_docs: !function utils.filter_<task>``.
# ---------------------------------------------------------------------------

TASK_DOMAINS = {
    # Single-Document QA
    "govt_single": ("Single-Document QA", "Governmental"),
    "legal_single": ("Single-Document QA", "Legal"),
    "lit_single": ("Single-Document QA", "Literary"),
    "fin_single": ("Single-Document QA", "Financial"),
    "event_order": ("Single-Document QA", "Event ordering"),
    "academic_single": ("Single-Document QA", "Academic"),
    "detective": ("Single-Document QA", "Detective"),
    # Multi-Document QA
    "govt_multi": ("Multi-Document QA", "Governmental"),
    "academic_multi": ("Multi-Document QA", "Academic"),
    "fin_multi": ("Multi-Document QA", "Financial"),
    "news_multi": ("Multi-Document QA", "Multi-news"),
    "legal_multi": ("Multi-Document QA", "Legal"),
    # Long In-context Learning
    "user_guide": ("Long In-context Learning", "User guide QA"),
    "translate": ("Long In-context Learning", "New language translation"),
    "many_shot": ("Long In-context Learning", "Many-shot learning"),
    # Long-dialogue History Understanding
    "agent_history": ("Long-dialogue History Understanding", "Agent history QA"),
    "dialogue_history": ("Long-dialogue History Understanding", "Dialogue history QA"),
    # Long Structured Data Understanding
    "graph": ("Long Structured Data Understanding", "Knowledge graph reasoning"),
    "table": ("Long Structured Data Understanding", "Table QA"),
    # Code Repository Understanding
    "code": ("Code Repository Understanding", "Code repo QA"),
}


def _add_options_with_text(doc):
    """Attach a formatted ``options_with_text`` block built from ``choice_*``.

    This bundles the four options into a single human-readable field so the
    downstream LLM-as-judge step (``judge_vllm.py``) can read
    ``doc["options_with_text"]`` straight from the written samples without
    re-running inference. It does not affect prompting or exact-match scoring.
    """
    doc["options_with_text"] = (
        f"A. {doc['choice_A']}\n"
        f"B. {doc['choice_B']}\n"
        f"C. {doc['choice_C']}\n"
        f"D. {doc['choice_D']}"
    )
    return doc


def _make_domain_filter(domain: str, sub_domain: str):
    def _filter(dataset):
        dataset = dataset.filter(
            lambda doc: doc["domain"] == domain and doc["sub_domain"] == sub_domain
        )
        return dataset.map(_add_options_with_text)

    return _filter


# Expose one ``filter_<task>`` callable per task at module scope so the YAMLs
# can reference them via ``!function utils.filter_<task>``.
for _suffix, (_domain, _sub_domain) in TASK_DOMAINS.items():
    globals()[f"filter_{_suffix}"] = _make_domain_filter(_domain, _sub_domain)
