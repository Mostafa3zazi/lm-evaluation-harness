import re
import string

# SQuAD-style normalization (English): lowercase, drop punctuation + articles, collapse spaces.
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text) -> str:
    text = str(text).lower()
    text = text.translate(_PUNCT_TABLE)   # strip punctuation
    text = _ARTICLES.sub(" ", text)       # strip articles (a / an / the)
    return " ".join(text.split())         # collapse whitespace


def simpleqa_automated_metrics(predictions, references, **kwargs):
    """Automated (non-LLM) baseline metrics for SimpleQA-Verified, reported ALONGSIDE the
    LLM judge (judge_vllm.py writes acc / correct / incorrect / not_attempted /
    correct_given_attempted / f_score under separate keys, so these survive the merge).

    Returns two keys (lm-eval propagates the metric_list aggregation/higher_is_better to each):
      - exact_match : normalized response EQUALS normalized gold (strict; ~0 for verbose models)
      - cover_match : normalized gold appears as a whitespace-bounded substring of the response
                      ("cover-EM"; the useful proxy for free-form chat answers)

    Mirrors the metric-function signature used by the other `_gen` tasks
    (e.g. arabicmmlu_gen.exact_match_normalized_label).
    """
    pred = _normalize(predictions[0] if predictions else "")
    gold = _normalize(references[0] if references else "")
    if not gold:
        return {"exact_match": 0.0, "cover_match": 0.0}
    return {
        "exact_match": float(pred == gold),
        "cover_match": float(f" {gold} " in f" {pred} "),
    }
