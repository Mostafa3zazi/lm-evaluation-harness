import re
import string

# Arabic-aware normalization for the AUTOMATED baseline only (the real score comes from the LLM
# judge -> ternary A/B/C). Strip diacritics + tatweel, unify alef/ya/ta-marbuta, drop punctuation,
# collapse spaces. Mirrors halluscore/utils.py.
_ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")  # harakat + tatweel
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "؟،؛")


def _normalize(text) -> str:
    text = str(text).lower()
    text = _ARABIC_DIACRITICS.sub("", text)           # strip harakat + tatweel
    text = text.translate(_PUNCT_TABLE)               # strip ASCII + Arabic punctuation
    text = re.sub(r"[إأآا]", "ا", text)               # unify alef forms
    text = text.replace("ى", "ي").replace("ة", "ه")  # ya / ta-marbuta variants
    return " ".join(text.split())                     # collapse whitespace


def simpleqa_ar_automated_metrics(predictions, references, **kwargs):
    """Automated (non-LLM) baseline for Arabic SimpleQA-Verified, reported ALONGSIDE the LLM judge
    (judge_vllm.py writes acc / correct / incorrect / not_attempted / correct_given_attempted /
    f_score from the ternary grade under separate keys, so these survive the merge).

    Returns:
      - exact_match : normalized response EQUALS normalized gold (strict; ~0 for verbose models)
      - cover_match : normalized gold appears as a whitespace-bounded substring of the response
                      (cover-EM; the useful proxy for free-form Arabic answers). Higher is better.
    """
    pred = _normalize(predictions[0] if predictions else "")
    gold = _normalize(references[0] if references else "")
    if not gold:
        return {"exact_match": 0.0, "cover_match": 0.0}
    return {
        "exact_match": float(pred == gold),
        "cover_match": float(f" {gold} " in f" {pred} "),
    }
