import re
import sys
import unicodedata

from lm_eval.api.filter import Filter
from lm_eval.filters.extraction import RegexFilter


# ---------------------------------------------------------------------------
# LLM360 eval-360 `multiple_choice` grader, reimplemented as an lm-eval filter.
#
# In eval-360 scoring is two stages:
#   1. a parse stage produces `parsed_generations`;
#   2. the `multiple_choice` grader runs `_extract_choice_symbol` on
#      `parsed_generations[i]` (or the raw generation when the parse is None),
#      i.e. it takes the FIRST alphabetic char (1-5 -> A-E), and compares that
#      letter to the normalized ground-truth letter.
#
# The three helpers below are copied verbatim from their grader.py so the
# grade stage is identical. We only have to approximate the parse stage, which
# their frozen jsonl baked in and we don't have the code for. The prompt tells
# the model to end with "the answer is (X)", so the parse keeps the text after
# the final "answer is"; if that marker is absent the parse "fails" (None) and
# we fall back to the raw generation exactly as their grader does.
# ---------------------------------------------------------------------------

_MC_DIGIT_TO_LETTER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}


def _normalize_choice_symbol(value):  # verbatim from eval-360 grader.py
    if value is None:
        return None
    candidate = value.strip().upper()
    if not candidate:
        return None
    return _MC_DIGIT_TO_LETTER.get(candidate, candidate)


def _extract_choice_symbol(sampled):  # verbatim from eval-360 grader.py
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


_ANSWER_MARKER = re.compile(r"answer is", re.IGNORECASE)


def _parse_generation(text):
    """Approximates eval-360's parse stage (what fills `parsed_generations`).

    Returns the substring after the final "answer is" marker, or None if the
    marker is absent (a "failed" parse -> grader falls back to the raw text).
    """
    matches = list(_ANSWER_MARKER.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end() :]


class LLM360MultipleChoiceFilter(Filter):
    """Replicates eval-360 `multiple_choice` grading as an lm-eval filter.

    For each response: run the parse stage, fall back to the raw generation
    when the parse is None (mirrors the grader's `parsed_generations[i] is None`
    branch), then apply `_extract_choice_symbol`. The emitted symbol is compared
    against a bare-letter `doc_to_target` via `exact_match`, which reproduces
    their `expected_symbol == picked` check.
    """

    def apply(self, resps, docs):
        filtered_resps = []
        for resp_list in resps:
            picks = []
            for resp in resp_list:
                parsed = _parse_generation(resp)
                sampled = parsed if parsed is not None else resp
                picked = _extract_choice_symbol(sampled)
                # picked is None only for empty output; emit a token that can
                # never match a gold letter so it scores as incorrect, matching
                # their `picked is not None and ...` guard.
                picks.append(picked if picked is not None else "[invalid]")
            filtered_resps.append(picks)
        return filtered_resps


class MultiChoiceRegexFilter(RegexFilter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        r"""
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


def add_options_with_text(dataset):
    """Adds an `options_with_text` field (A./B./C./D. block) to each doc.

    Mirrors mmlu_gen's `process_docs` so the rendered options are available on
    the logged sample for downstream inspection / judging.
    """

    def _add_field(doc):
        choices = doc["choices"]
        doc["options_with_text"] = (
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}"
        )
        return doc

    return dataset.map(_add_field)
