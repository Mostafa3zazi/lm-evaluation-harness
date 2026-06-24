import re
import string

# Arabic-aware normalization for the AUTOMATED baseline only (the real HalluScore score comes from
# the LLM judge -> ternary A/B/C). Strip diacritics + tatweel, unify alef/ya/ta-marbuta variants,
# drop punctuation, collapse spaces.
_ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")  # harakat + tatweel
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "؟،؛")


def _normalize(text) -> str:
    text = str(text).lower()
    text = _ARABIC_DIACRITICS.sub("", text)           # strip harakat + tatweel
    text = text.translate(_PUNCT_TABLE)               # strip ASCII + Arabic punctuation
    text = re.sub(r"[إأآا]", "ا", text)               # unify alef forms
    text = text.replace("ى", "ي").replace("ة", "ه")  # ya / ta-marbuta variants
    return " ".join(text.split())                     # collapse whitespace


def halluscore_automated_metrics(predictions, references, **kwargs):
    """Automated (non-LLM) baseline for HalluScore, reported ALONGSIDE the LLM judge
    (judge_vllm.py writes acc / correct / incorrect / not_attempted / correct_given_attempted /
    f_score from the ternary grade under separate keys, so this survives the merge).

    Returns:
      - cover_match : normalized gold answer appears as a whitespace-bounded substring of the
                      response (cover-EM; a coarse proxy for the gold being present in free-form
                      Arabic answers). Higher is better.
    """
    pred = _normalize(predictions[0] if predictions else "")
    gold = _normalize(references[0] if references else "")
    if not gold:
        return {"cover_match": 0.0}
    return {"cover_match": float(f" {gold} " in f" {pred} ")}
