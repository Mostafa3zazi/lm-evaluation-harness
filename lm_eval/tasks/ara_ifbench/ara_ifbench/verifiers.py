from __future__ import annotations

import re
from typing import Callable

from .closed_lists import (
    ARABIC_DIACRITICS,
    ARABIC_INDIC_DIGITS,
    ASCII_PUNCT_EQUIVALENTS,
    ASCII_DIGITS,
    CONJUNCTIONS,
    COPY_SEPARATOR,
    HAMZA_CHARS,
    HAMZA_FORMS,
    OUTPUT_TEMPLATE_PREFIXES,
    PERSON_NAMES,
    PERSIAN_VARIANT_CHARS,
    PRONOUNS,
    SECTION_PREFIX,
    TANWEEN_CHARS,
    ZERO_WIDTH_CHARS,
    first_significant_char,
    in_arabic_presentation_block,
    is_arabic_letter,
    is_latin_letter,
    normalized_token_for_match,
    strip_edge_punctuation,
)
from .lexicon import load_lexicon
from .normalize import normalize, strip_formatting_wrappers
from .schemas import ConstraintSpec
from .splitters import split_lines, split_paragraphs, split_sentences
from .tokenize import comparable_tokens, whitespace_tokens


VerifierFn = Callable[[ConstraintSpec, str, dict], tuple[bool, str, dict]]
ARABIC_INDIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PROCLITIC_PREFIXES = ("و", "ف", "ب", "ك", "ل")
CONJUNCTION_CLITIC_PREFIXES = ("و", "ف")
SHORT_VOWEL_CHARS = set("َُِ")
SECTION_ORDINAL_INDEX = {
    "الأول": 1,
    "الاول": 1,
    "الثاني": 2,
    "الثالث": 3,
}
TERMINAL_END_PHRASE_CHARS = " \t\r\n.!?؟،؛:;…\"'»”’)]}*"
LEADING_SEQUENCE_MARKER_RE = re.compile(r"^\s*(?:#+\s*)?(?:(?:\d+|[٠-٩]+)[\.\)]|[-*•])\s*")
MATCH_MODE_KEY = "__match_mode__"


def verify_constraint(spec: ConstraintSpec, response_text: str, kwargs: dict, *, match_mode: str = "strict") -> tuple[bool, str, dict]:
    if match_mode not in {"strict", "loose"}:
        raise ValueError(f"Unsupported match mode: {match_mode}")
    handler = HANDLERS[spec.handler]
    payload = dict(kwargs)
    payload[MATCH_MODE_KEY] = match_mode
    return handler(spec, response_text, payload)


def exact_sentence_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    count = len(split_sentences(text, spec.normalization_mode))
    expected = kwargs["n"]
    return count == expected, _code(count == expected, "wrong_sentence_count"), {"observed": count, "expected": expected}


def exact_paragraph_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    count = len(split_paragraphs(text, spec.normalization_mode))
    expected = kwargs["n"]
    return count == expected, _code(count == expected, "wrong_paragraph_count"), {"observed": count, "expected": expected}


def exact_word_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    count = len(whitespace_tokens(text, spec.normalization_mode))
    expected = kwargs["n"]
    return count == expected, _code(count == expected, "wrong_word_count"), {"observed": count, "expected": expected}


def word_count_range(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    count = len(whitespace_tokens(text, spec.normalization_mode))
    passed = kwargs["min_words"] <= count <= kwargs["max_words"]
    return passed, _code(passed, "word_count_out_of_range"), {"observed": count, "expected_min": kwargs["min_words"], "expected_max": kwargs["max_words"]}


def unique_word_count_min(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = len(set(comparable_tokens(text, spec.normalization_mode)))
    expected = kwargs["n"]
    return observed >= expected, _code(observed >= expected, "unique_word_count_too_small"), {"observed": observed, "expected": expected}


def numeric_spans_exact(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = len(re.findall(r"[0-9٠-٩]+", normalize(text, spec.normalization_mode)))
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_numeric_span_count"), {"observed": observed, "expected": expected}


def keywords_multiple_exact(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    tokens = comparable_tokens(text, spec.normalization_mode)
    observed = {}
    passed = True
    for keyword, expected_count in kwargs["keyword_counts"].items():
        observed_count = tokens.count(normalized_token_for_match(keyword))
        observed[keyword] = observed_count
        if observed_count != expected_count:
            passed = False
    return passed, _code(passed, "keyword_count_mismatch"), {"observed": observed, "expected": kwargs["keyword_counts"]}


def closed_list_min(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    list_values = _resolve_closed_list(spec.id)
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    observed = sum(
        bool(candidates & list_values)
        for candidates in _semantic_token_candidate_sets(
            text,
            spec.normalization_mode,
            strip_diacritics=strip_diacritics,
            **_closed_list_match_options(spec.id),
        )
    )
    expected = kwargs["n"]
    return observed >= expected, _code(observed >= expected, "closed_list_count_too_small"), {"observed": observed, "expected": expected}


def closed_list_distinct_min(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    list_values = _resolve_closed_list(spec.id)
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    observed_values: set[str] = set()
    for candidates in _semantic_token_candidate_sets(
        text,
        spec.normalization_mode,
        strip_diacritics=strip_diacritics,
        **_closed_list_match_options(spec.id),
    ):
        observed_values.update(candidates & list_values)
    expected = kwargs["n"]
    observed = len(observed_values)
    return observed >= expected, _code(observed >= expected, "closed_list_distinct_count_too_small"), {"observed": observed, "expected": expected, "values": sorted(observed_values)}


def title_wrapped(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    first_line = _normalize_layout_line(lines[0]) if lines else ""
    passed = bool(first_line and first_line.startswith("<<") and first_line.endswith(">>") and len(first_line) > 4)
    return passed, _code(passed, "missing_wrapped_title"), {"first_line": first_line}


def output_template_three_fields(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    fields = _extract_output_template_fields(lines)
    passed = len(fields) == len(OUTPUT_TEMPLATE_PREFIXES) and [prefix for prefix, _ in fields] == list(OUTPUT_TEMPLATE_PREFIXES) and all(
        content for _, content in fields
    )
    return passed, _code(passed, "wrong_output_template"), {"lines": lines, "fields": fields}


def newline_words(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    passed = bool(lines) and all(len(whitespace_tokens(line, spec.normalization_mode)) == 1 for line in lines)
    return passed, _code(passed, "not_one_word_per_line"), {"lines": lines}


def custom_separator_list(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    items = [item.strip() for item in normalized.split(kwargs["sep"])]
    passed = len(items) == kwargs["n"] and all(item for item in items)
    return passed, _code(passed, "wrong_custom_separator_list"), {"observed_items": items, "separator": kwargs["sep"]}


def line_indent_stairs(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    indents = [len(line) - len(line.lstrip(" ")) for line in lines]
    passed = len(lines) >= 2 and all(curr > prev for prev, curr in zip(indents, indents[1:]))
    return passed, _code(passed, "indentation_not_strictly_increasing"), {"indents": indents}


def exact_bullet_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    observed = sum(_is_bullet_line(line) for line in lines)
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_bullet_count"), {"observed": observed, "expected": expected}


def multiple_sections(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    lines = split_lines(text, spec.normalization_mode)
    section_lines = [line for line in lines if _parse_section_heading_index(line) is not None]
    expected = kwargs["n"]
    observed = [_parse_section_heading_index(line) for line in section_lines[:expected]]
    wanted = list(range(1, expected + 1))
    passed = len(section_lines) == expected and observed == wanted
    return passed, _code(passed, "wrong_section_layout"), {"observed": section_lines, "expected": wanted}


def wrapped_in_guillemets(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode).strip()
    passed = normalized.startswith("«") and normalized.endswith("»") and len(normalized) >= 2
    return passed, _code(passed, "missing_guillemets"), {"observed": normalized}


def nested_quotes(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    depth = kwargs.get("depth", 2)
    normalized = normalize(text, spec.normalization_mode)
    if depth <= 2:
        passed = any(re.search(pattern, normalized) for pattern in _depth_two_quote_patterns()) or _has_nested_blockquote_depth(normalized, 2)
    else:
        passed = any(re.search(pattern, normalized) for pattern in _depth_three_quote_patterns()) or _has_nested_blockquote_depth(normalized, 3)
    return passed, _code(passed, "missing_nested_quotes"), {"depth": depth}


def nested_parentheses(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    depth = kwargs["depth"]
    observed = _max_bracket_depth(normalize(text, spec.normalization_mode))
    return observed >= depth, _code(observed >= depth, "nested_parentheses_too_shallow"), {"observed": observed, "expected_min": depth}


def no_whitespace(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    passed = not any(char.isspace() for char in normalize(text, spec.normalization_mode))
    return passed, _code(passed, "whitespace_present"), {}


def keyword_specific_position(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    s_idx = kwargs["sentence_index"] - 1
    w_idx = kwargs["word_index"] - 1
    if not (0 <= s_idx < len(sentences)):
        return False, "sentence_index_out_of_range", {"observed_sentences": len(sentences)}
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        sentences[s_idx],
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        sentences[s_idx],
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["keyword"], strip_diacritics=strip_diacritics)
    passed = 0 <= w_idx < len(token_sets) and expected in token_sets[w_idx]
    return passed, _code(passed, "keyword_not_in_required_position"), {"observed_tokens": tokens}


def nth_paragraph_first_word(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    paragraphs = split_paragraphs(text, spec.normalization_mode)
    p_idx = kwargs["paragraph_index"] - 1
    if not (0 <= p_idx < len(paragraphs)):
        return False, "paragraph_index_out_of_range", {"observed_paragraphs": len(paragraphs)}
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        paragraphs[p_idx],
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        paragraphs[p_idx],
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["word"], strip_diacritics=strip_diacritics)
    passed = bool(token_sets) and expected in token_sets[0]
    return passed, _code(passed, "paragraph_first_word_mismatch"), {"observed_tokens": tokens}


def second_and_second_last_word(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["keyword"], strip_diacritics=strip_diacritics)
    passed = len(token_sets) >= 2 and expected in token_sets[1] and expected in token_sets[-2]
    return passed, _code(passed, "second_or_second_last_word_mismatch"), {"observed_tokens": tokens}


def ends_with_phrase(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode).rstrip()
    phrase = kwargs["phrase"]
    stripped_terminal = normalized.rstrip(TERMINAL_END_PHRASE_CHARS)
    passed = normalized.endswith(phrase) or stripped_terminal.endswith(phrase)
    return passed, _code(passed, "end_phrase_mismatch"), {"observed": normalized[-40:], "stripped": stripped_terminal[-40:]}


def same_start_end_word(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    passed = len(token_sets) >= 1 and bool(token_sets[0] & token_sets[-1])
    return passed, _code(passed, "start_end_word_mismatch"), {"observed_tokens": tokens[:3] + tokens[-3:]}


def exact_first_token(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["word"], strip_diacritics=strip_diacritics)
    passed = bool(token_sets) and expected in token_sets[0]
    return passed, _code(passed, "first_token_mismatch"), {"observed": tokens[0] if tokens else ""}


def exact_last_token(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        text,
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["word"], strip_diacritics=strip_diacritics)
    passed = bool(token_sets) and expected in token_sets[-1]
    return passed, _code(passed, "last_token_mismatch"), {"observed": tokens[-1] if tokens else ""}


def exact_source_copy(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode)
    expected = kwargs["source_text"]
    passed = observed == expected
    return passed, _code(passed, "source_copy_mismatch"), {"observed": observed, "expected": expected}


def copy_change_first_word(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    source_tokens = whitespace_tokens(kwargs["source_text"], "strict_surface")
    if not source_tokens:
        return False, "empty_source_text", {}
    expected = " ".join([kwargs["replacement"], *source_tokens[1:]])
    observed = normalize(text, spec.normalization_mode)
    passed = observed == expected
    return passed, _code(passed, "copy_change_first_word_mismatch"), {"observed": observed, "expected": expected}


def copy_span_char_idx(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    source = kwargs["source_text"]
    expected = source[kwargs["start_idx"] : kwargs["end_idx"] + 1]
    observed = normalize(text, spec.normalization_mode)
    passed = observed == expected
    return passed, _code(passed, "copy_span_mismatch"), {"observed": observed, "expected": expected}


def copy_request_n_times(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    expected = COPY_SEPARATOR.join([kwargs["source_text"]] * kwargs["n"])
    observed = normalize(text, spec.normalization_mode)
    passed = observed == expected
    return passed, _code(passed, "copy_n_times_mismatch"), {"observed": observed, "expected": expected}


def reverse_word_order(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    expected = " ".join(reversed(whitespace_tokens(kwargs["source_text"], "strict_surface")))
    observed = normalize(text, spec.normalization_mode)
    passed = observed == expected
    return passed, _code(passed, "reverse_word_order_mismatch"), {"observed": observed, "expected": expected}


def reverse_char_order(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    expected = kwargs["source_text"][::-1]
    observed = normalize(text, spec.normalization_mode)
    passed = observed == expected
    return passed, _code(passed, "reverse_char_order_mismatch"), {"observed": observed, "expected": expected}


def square_brackets_every_token(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    raw_tokens = whitespace_tokens(text, spec.normalization_mode)
    passed = bool(raw_tokens) and all(token.startswith("[") and token.endswith("]") and len(token) > 2 for token in raw_tokens)
    return passed, _code(passed, "token_not_wrapped_in_square_brackets"), {"observed_tokens": raw_tokens}


def wrap_every_bigram(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode).strip()
    groups = re.findall(r"«([^»]+)»", normalized)
    if not groups:
        return False, "missing_bigram_wrapping", {"observed": normalized}
    reconstructed = []
    for group in groups:
        reconstructed.extend(group.split())
    observed_tokens = whitespace_tokens(normalized.replace("«", "").replace("»", ""), spec.normalization_mode)
    passed = reconstructed == observed_tokens and all(1 <= len(group.split()) <= 2 for group in groups)
    return passed, _code(passed, "bad_bigram_wrapping"), {"groups": groups}


def keyword_in_nth_sentence(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    s_idx = kwargs["sentence_index"] - 1
    if not (0 <= s_idx < len(sentences)):
        return False, "sentence_index_out_of_range", {"observed_sentences": len(sentences)}
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    token_sets = _semantic_token_candidate_sets(
        sentences[s_idx],
        spec.normalization_mode,
        split_proclitics=True,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    tokens = _semantic_token_bases(
        sentences[s_idx],
        spec.normalization_mode,
        strip_sequence_marker=True,
        strip_diacritics=strip_diacritics,
    )
    expected = _semantic_token_base(kwargs["keyword"], strip_diacritics=strip_diacritics)
    passed = any(expected in candidates for candidates in token_sets)
    return passed, _code(passed, "keyword_missing_from_nth_sentence"), {"observed_tokens": tokens}


def increment_word_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    counts = [len(_semantic_token_bases(sentence, spec.normalization_mode, strip_sequence_marker=True)) for sentence in sentences]
    delta = kwargs["delta"]
    passed = len(counts) >= 2 and all(curr - prev == delta for prev, curr in zip(counts, counts[1:]))
    return passed, _code(passed, "sentence_word_increment_mismatch"), {"counts": counts, "delta": delta}


def last_word_first_next_sentence(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    chains = []
    passed = len(sentences) >= 2
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    for left, right in zip(sentences, sentences[1:]):
        left_token_sets = _semantic_token_candidate_sets(
            left,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        right_token_sets = _semantic_token_candidate_sets(
            right,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        left_tokens = _semantic_token_bases(
            left,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        right_tokens = _semantic_token_bases(
            right,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        if not left_token_sets or not right_token_sets or not (left_token_sets[-1] & right_token_sets[0]):
            passed = False
        chains.append((left_tokens[-1] if left_tokens else "", right_tokens[0] if right_tokens else ""))
    return passed, _code(passed, "sentence_chain_mismatch"), {"chains": chains}


def paragraph_last_first(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    paragraphs = split_paragraphs(text, spec.normalization_mode)
    observed = []
    passed = bool(paragraphs)
    strip_diacritics = _match_uses_diacritic_relaxation(kwargs)
    for paragraph in paragraphs:
        token_sets = _semantic_token_candidate_sets(
            paragraph,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        tokens = _semantic_token_bases(
            paragraph,
            spec.normalization_mode,
            strip_sequence_marker=True,
            strip_diacritics=strip_diacritics,
        )
        if not token_sets or not (token_sets[0] & token_sets[-1]):
            passed = False
        observed.append(tokens[:1] + tokens[-1:])
    return passed, _code(passed, "paragraph_start_end_mismatch"), {"observed": observed}


def same_char_count_three_sentences(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    counts = [_normalized_sentence_char_count(sentence) for sentence in sentences]
    passed = len(sentences) == 3 and len(set(counts)) == 1
    return passed, _code(passed, "sentence_char_counts_not_equal"), {"counts": counts}


def no_consecutive_same_first_letter(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    tokens = whitespace_tokens(text, spec.normalization_mode)
    first_letters = [first_significant_char(token) for token in tokens if first_significant_char(token)]
    passed = all(left != right for left, right in zip(first_letters, first_letters[1:]))
    return passed, _code(passed, "consecutive_same_first_letter"), {"letters": first_letters}


def only_arabic_letters(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    letters = [char for char in normalize(text, spec.normalization_mode) if char.isalpha()]
    passed = bool(letters) and all(is_arabic_letter(char) for char in letters)
    return passed, _code(passed, "non_arabic_letter_present"), {"letters": "".join(letters)}


def no_latin_letters(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    latin = [char for char in normalize(text, spec.normalization_mode) if is_latin_letter(char)]
    passed = not latin
    return passed, _code(passed, "latin_letter_present"), {"observed": "".join(latin)}


def no_persian_variants(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if char in PERSIAN_VARIANT_CHARS]
    passed = not observed
    return passed, _code(passed, "persian_variant_present"), {"observed": "".join(observed)}


def arabic_indic_only(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    digits = [char for char in normalize(text, spec.normalization_mode) if char.isdigit()]
    passed = all(char in ARABIC_INDIC_DIGITS for char in digits)
    return passed, _code(passed, "non_arabic_indic_digit_present"), {"digits": "".join(digits)}


def arabic_indic_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = sum(char in ARABIC_INDIC_DIGITS for char in normalize(text, spec.normalization_mode))
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_arabic_indic_digit_count"), {"observed": observed, "expected": expected}


def no_ascii_digits(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if char in ASCII_DIGITS]
    passed = not observed
    return passed, _code(passed, "ascii_digit_present"), {"observed": "".join(observed)}


def mixed_digit_system_forbidden(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    has_ascii = any(char in ASCII_DIGITS for char in normalized)
    has_indic = any(char in ARABIC_INDIC_DIGITS for char in normalized)
    passed = not (has_ascii and has_indic)
    return passed, _code(passed, "mixed_digit_system_present"), {"has_ascii": has_ascii, "has_indic": has_indic}


def arabic_comma_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("،")
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_arabic_comma_count"), {"observed": observed, "expected": expected}


def arabic_semicolon_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("؛")
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_arabic_semicolon_count"), {"observed": observed, "expected": expected}


def arabic_question_mark_presence(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("؟")
    passed = observed >= 1
    return passed, _code(passed, "missing_arabic_question_mark"), {"observed": observed}


def question_mark_every_sentence(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    sentences = split_sentences(text, spec.normalization_mode)
    passed = bool(sentences) and all(sentence.rstrip().endswith("؟") for sentence in sentences)
    return passed, _code(passed, "sentence_missing_arabic_question_mark"), {"sentences": sentences}


def no_ascii_punct_equivalents(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if char in ASCII_PUNCT_EQUIVALENTS]
    passed = not observed
    return passed, _code(passed, "ascii_punct_equivalent_present"), {"observed": "".join(observed)}


def cover_arabic_punct_set(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    missing = [char for char in ("،", "؛", "؟") if char not in normalized]
    passed = not missing
    return passed, _code(passed, "missing_arabic_punct_set"), {"missing": missing}


def shadda_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("ّ")
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_shadda_count"), {"observed": observed, "expected": expected}


def sukun_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("ْ")
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_sukun_count"), {"observed": observed, "expected": expected}


def cover_tanween_set(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    missing = [char for char in sorted(TANWEEN_CHARS) if char not in normalized]
    passed = not missing
    return passed, _code(passed, "missing_tanween_set"), {"missing": missing}


def no_diacritics(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if char in ARABIC_DIACRITICS]
    passed = not observed
    return passed, _code(passed, "arabic_diacritic_present"), {"observed": "".join(observed)}


def kashida_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = normalize(text, spec.normalization_mode).count("ـ")
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_kashida_count"), {"observed": observed, "expected": expected}


def no_zero_width_chars(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if char in ZERO_WIDTH_CHARS]
    passed = not observed
    return passed, _code(passed, "zero_width_char_present"), {"observed": "".join(observed)}


def no_presentation_forms(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = [char for char in normalize(text, spec.normalization_mode) if in_arabic_presentation_block(char)]
    passed = not observed
    return passed, _code(passed, "arabic_presentation_form_present"), {"observed": "".join(observed)}


def arabic_diacritics_exact_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = sum(char in ARABIC_DIACRITICS for char in normalize(text, spec.normalization_mode))
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_arabic_diacritic_count"), {"observed": observed, "expected": expected}


def token_suffix_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    suffix = "ة" if spec.id.endswith("ta_marbuta_token_count") else "ى"
    observed = sum(strip_edge_punctuation(token).endswith(suffix) for token in whitespace_tokens(text, spec.normalization_mode))
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_token_suffix_count"), {"observed": observed, "expected": expected, "suffix": suffix}


def definite_article_prefix_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = sum(
        stripped.startswith("ال") and len(stripped) > 2
        for token in whitespace_tokens(text, spec.normalization_mode)
        for stripped in [strip_edge_punctuation(token)]
    )
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_definite_article_prefix_count"), {"observed": observed, "expected": expected}


def hamza_char_count(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    observed = sum(char in HAMZA_CHARS for char in normalize(text, spec.normalization_mode))
    expected = kwargs["n"]
    return observed == expected, _code(observed == expected, "wrong_hamza_char_count"), {"observed": observed, "expected": expected}


def hamza_forms_cover_all(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    missing = [form for form in HAMZA_FORMS if form not in normalized]
    passed = not missing
    return passed, _code(passed, "missing_hamza_forms"), {"missing": missing}


def hamza_forms_exact_once_each(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    normalized = normalize(text, spec.normalization_mode)
    counts = {form: normalized.count(form) for form in HAMZA_FORMS}
    passed = all(count == 1 for count in counts.values())
    return passed, _code(passed, "wrong_hamza_form_profile"), {"counts": counts}


def morph_lookup_contains_target(spec: ConstraintSpec, text: str, kwargs: dict) -> tuple[bool, str, dict]:
    loose = _is_loose_match(kwargs)
    lexicon_name = _lexicon_name_for_spec(spec.id)
    lexicon = load_lexicon(lexicon_name)
    source_word = kwargs["source_word"]
    expected_targets = [
        _semantic_token_base(item, strip_diacritics=loose, strip_morph_tanween_alif=loose)
        for item in lexicon.get(source_word, [])
    ]
    tokens = _semantic_token_bases(
        text,
        spec.normalization_mode,
        strip_diacritics=loose,
        strip_morph_tanween_alif=loose,
    )
    passed = bool(expected_targets) and any(
        _morph_token_matches(spec.id, source_word, token, expected_targets, loose=loose)
        for token in _semantic_raw_tokens(text, spec.normalization_mode)
    )
    return passed, _code(passed, "lexicon_target_missing"), {"source_word": source_word, "expected_targets": expected_targets, "observed_tokens": tokens}


def _lexicon_name_for_spec(spec_id: str) -> str:
    mapping = (
        ("dual_from_singular", "dual_from_singular"),
        ("present_from_past", "present_from_past"),
        ("broken_plural_from_singular", "broken_plural_from_singular"),
        ("feminine_from_masculine_adj", "feminine_from_masculine_adj"),
        ("active_participle_from_past", "active_participle_from_past"),
        ("masdar_from_past", "masdar_from_past"),
        ("sound_masc_plural_from_singular", "sound_masc_plural_from_singular"),
        ("nisba_from_noun", "nisba_from_noun"),
    )
    for suffix, lexicon_name in mapping:
        if spec_id.endswith(suffix):
            return lexicon_name
    raise ValueError(f"No lexicon configured for spec {spec_id}")


def _resolve_closed_list(spec_id: str) -> set[str]:
    if spec_id.endswith("pronouns_min_closed_list"):
        return {item.casefold() for item in PRONOUNS}
    if spec_id.endswith("conjunctions_distinct_closed_list"):
        return {item.casefold() for item in CONJUNCTIONS}
    if spec_id.endswith("person_names_from_closed_list"):
        return {item.casefold() for item in PERSON_NAMES}
    raise ValueError(f"No closed list configured for spec {spec_id}")


def _closed_list_match_options(spec_id: str) -> dict[str, bool]:
    if spec_id.endswith("pronouns_min_closed_list"):
        return {"split_proclitics": True}
    if spec_id.endswith("conjunctions_distinct_closed_list"):
        return {"include_conjunction_clitic": True}
    if spec_id.endswith("person_names_from_closed_list"):
        return {"split_proclitics": True}
    return {}


def _is_loose_match(kwargs: dict) -> bool:
    return kwargs.get(MATCH_MODE_KEY) == "loose"


def _match_uses_diacritic_relaxation(kwargs: dict) -> bool:
    return _is_loose_match(kwargs)


def _semantic_text(text: str, normalization_mode: str) -> str:
    normalized = strip_formatting_wrappers(normalize(text, normalization_mode))
    return "".join(char for char in normalized if char not in ZERO_WIDTH_CHARS).replace("ـ", "")


def _strip_leading_sequence_marker(text: str) -> str:
    return LEADING_SEQUENCE_MARKER_RE.sub("", text, count=1)


def _normalize_morph_tanween_alif(token: str) -> str:
    return re.sub(r"اً$", "", token)


def _semantic_raw_tokens(text: str, normalization_mode: str, *, strip_sequence_marker: bool = False) -> list[str]:
    normalized = _semantic_text(text, normalization_mode)
    if strip_sequence_marker:
        normalized = _strip_leading_sequence_marker(normalized)
    return [token for token in re.split(r"\s+", normalized.strip()) if token]


def _semantic_token_base(
    token: str,
    *,
    strip_diacritics: bool = False,
    strip_morph_tanween_alif: bool = False,
) -> str:
    stripped = strip_edge_punctuation(token)
    if strip_morph_tanween_alif:
        stripped = _normalize_morph_tanween_alif(stripped)
    if strip_diacritics:
        stripped = "".join(char for char in stripped if char not in ARABIC_DIACRITICS)
    return stripped.casefold()


def _semantic_token_candidates(
    token: str,
    *,
    strip_diacritics: bool = False,
    split_proclitics: bool = False,
    strip_definite_article: bool = False,
    include_conjunction_clitic: bool = False,
    strip_morph_tanween_alif: bool = False,
) -> set[str]:
    base = _semantic_token_base(
        token,
        strip_diacritics=strip_diacritics,
        strip_morph_tanween_alif=strip_morph_tanween_alif,
    )
    if not base:
        return set()
    variants = {base}
    if split_proclitics:
        for item in list(variants):
            if item[:1] in PROCLITIC_PREFIXES and len(item) > 1:
                variants.add(item[1:])
    if include_conjunction_clitic:
        for item in list(variants):
            if item[:1] in CONJUNCTION_CLITIC_PREFIXES and len(item) > 1:
                variants.add(item[0])
    if strip_definite_article:
        for item in list(variants):
            if item.startswith("ال") and len(item) > 2:
                variants.add(item[2:])
    return {item for item in variants if item}


def _semantic_token_bases(
    text: str,
    normalization_mode: str,
    *,
    strip_sequence_marker: bool = False,
    strip_diacritics: bool = False,
    strip_morph_tanween_alif: bool = False,
) -> list[str]:
    normalized = _semantic_text(text, normalization_mode)
    if strip_sequence_marker:
        normalized = _strip_leading_sequence_marker(normalized)
    tokens: list[str] = []
    for token in re.split(r"\s+", normalized.strip()):
        base = _semantic_token_base(
            token,
            strip_diacritics=strip_diacritics,
            strip_morph_tanween_alif=strip_morph_tanween_alif,
        )
        if base:
            tokens.append(base)
    return tokens


def _semantic_token_candidate_sets(
    text: str,
    normalization_mode: str,
    *,
    strip_sequence_marker: bool = False,
    strip_diacritics: bool = False,
    split_proclitics: bool = False,
    strip_definite_article: bool = False,
    include_conjunction_clitic: bool = False,
    strip_morph_tanween_alif: bool = False,
) -> list[set[str]]:
    normalized = _semantic_text(text, normalization_mode)
    if strip_sequence_marker:
        normalized = _strip_leading_sequence_marker(normalized)
    token_sets: list[set[str]] = []
    for token in re.split(r"\s+", normalized.strip()):
        candidates = _semantic_token_candidates(
            token,
            strip_diacritics=strip_diacritics,
            split_proclitics=split_proclitics,
            strip_definite_article=strip_definite_article,
            include_conjunction_clitic=include_conjunction_clitic,
            strip_morph_tanween_alif=strip_morph_tanween_alif,
        )
        if candidates:
            token_sets.append(candidates)
    return token_sets


def _morph_token_matches(spec_id: str, source_word: str, token: str, expected_targets: list[str], *, loose: bool) -> bool:
    candidates = _semantic_token_candidates(
        token,
        strip_diacritics=loose,
        split_proclitics=True,
        strip_definite_article=True,
        strip_morph_tanween_alif=loose,
    )
    matched_targets = candidates & set(expected_targets)
    if not matched_targets:
        return False
    if loose and spec_id.endswith("masdar_from_past"):
        source_base = _semantic_token_base(source_word, strip_diacritics=True)
        if source_base in matched_targets and _looks_like_explicit_past_verb_vocalization(token, source_word):
            return False
    return True


def _looks_like_explicit_past_verb_vocalization(token: str, source_word: str) -> bool:
    stripped = strip_edge_punctuation(token)
    if not any(char in ARABIC_DIACRITICS for char in stripped):
        return False
    letters = "".join(char for char in stripped if is_arabic_letter(char))
    if letters.startswith("ال") and len(letters) > 2:
        return False
    if letters != source_word:
        return False
    marks_by_letter: list[list[str]] = []
    for char in stripped:
        if is_arabic_letter(char):
            marks_by_letter.append([])
        elif char in ARABIC_DIACRITICS and marks_by_letter:
            marks_by_letter[-1].append(char)
    if len(marks_by_letter) < 3:
        return False
    if any(mark in TANWEEN_CHARS for marks in marks_by_letter for mark in marks):
        return False
    first_has_short = any(mark in SHORT_VOWEL_CHARS for mark in marks_by_letter[0])
    second_has_short = any(mark in SHORT_VOWEL_CHARS for mark in marks_by_letter[1])
    second_has_sukun = "ْ" in marks_by_letter[1]
    last_has_short = any(mark in SHORT_VOWEL_CHARS for mark in marks_by_letter[-1])
    return first_has_short and second_has_short and not second_has_sukun and last_has_short


def _normalized_sentence_char_count(sentence: str) -> int:
    normalized = _strip_leading_sequence_marker(_semantic_text(sentence, "soft_surface")).strip().rstrip(".!?؟")
    return sum(not char.isspace() for char in normalized)


def _max_bracket_depth(text: str) -> int:
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = {value: key for key, value in openers.items()}
    stack: list[str] = []
    max_depth = 0
    for char in text:
        if char in openers:
            stack.append(char)
            max_depth = max(max_depth, len(stack))
        elif char in closers:
            if not stack or stack[-1] != closers[char]:
                return 0
            stack.pop()
    return max_depth if not stack else 0


def _code(passed: bool, failure_code: str) -> str:
    return "ok" if passed else failure_code


def _normalize_layout_line(line: str) -> str:
    return strip_formatting_wrappers(line.strip()).strip()


def _extract_output_template_fields(lines: list[str]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    current_prefix: str | None = None
    current_content: list[str] = []
    for line in lines:
        normalized = _normalize_layout_line(line)
        matched_prefix = next((prefix for prefix in OUTPUT_TEMPLATE_PREFIXES if normalized.startswith(prefix)), None)
        if matched_prefix is not None:
            if current_prefix is not None:
                fields.append((current_prefix, " ".join(part for part in current_content if part).strip()))
            remainder = normalized[len(matched_prefix) :].strip()
            current_prefix = matched_prefix
            current_content = [remainder] if remainder else []
            continue
        if current_prefix is None:
            return []
        current_content.append(normalized)
    if current_prefix is not None:
        fields.append((current_prefix, " ".join(part for part in current_content if part).strip()))
    return fields


def _is_bullet_line(line: str) -> bool:
    stripped = _normalize_layout_line(line)
    if stripped.startswith(("-", "*", "•")):
        return True
    return bool(re.match(r"^(?:\d+|[٠-٩]+)[\.\)\-:]\s*", stripped))


def _parse_section_heading_index(line: str) -> int | None:
    stripped = _normalize_layout_line(line)
    stripped = re.sub(r"^#+\s*", "", stripped)
    stripped = stripped.rstrip(":").strip()
    if not stripped.startswith(SECTION_PREFIX):
        return None
    remainder = stripped[len(SECTION_PREFIX) :].strip().translate(ARABIC_INDIC_TO_ASCII)
    if remainder.isdigit():
        return int(remainder)
    return SECTION_ORDINAL_INDEX.get(remainder)


def _depth_two_quote_patterns() -> tuple[str, ...]:
    return (
        r"«[^»]*[\"“”][^\"“”]+[\"“”][^»]*»",
        r"«[^»]*«[^»]+»[^»]*»",
        r"[\"“”][^\"“”]*['‘’][^'‘’]+['‘’][^\"“”]*[\"“”]",
    )


def _depth_three_quote_patterns() -> tuple[str, ...]:
    return (
        r"«[^»]*[\"“”][^\"“”]*['‘’][^'‘’]+['‘’][^\"“”]*[\"“”][^»]*»",
        r"[\"“”][^\"“”]*«[^»]*['‘’][^'‘’]+['‘’][^»]*»[^\"“”]*[\"“”]",
    )


def _has_nested_blockquote_depth(text: str, depth: int) -> bool:
    depths = set()
    for line in text.splitlines():
        stripped = line.lstrip()
        current_depth = 0
        while stripped.startswith(">"):
            current_depth += 1
            stripped = stripped[1:]
        if current_depth:
            depths.add(current_depth)
    return set(range(1, depth + 1)).issubset(depths)


HANDLERS: dict[str, VerifierFn] = {
    "exact_sentence_count": exact_sentence_count,
    "exact_paragraph_count": exact_paragraph_count,
    "exact_word_count": exact_word_count,
    "word_count_range": word_count_range,
    "unique_word_count_min": unique_word_count_min,
    "numeric_spans_exact": numeric_spans_exact,
    "keywords_multiple_exact": keywords_multiple_exact,
    "closed_list_min": closed_list_min,
    "closed_list_distinct_min": closed_list_distinct_min,
    "title_wrapped": title_wrapped,
    "output_template_three_fields": output_template_three_fields,
    "newline_words": newline_words,
    "custom_separator_list": custom_separator_list,
    "line_indent_stairs": line_indent_stairs,
    "exact_bullet_count": exact_bullet_count,
    "multiple_sections": multiple_sections,
    "wrapped_in_guillemets": wrapped_in_guillemets,
    "nested_quotes": nested_quotes,
    "nested_parentheses": nested_parentheses,
    "no_whitespace": no_whitespace,
    "keyword_specific_position": keyword_specific_position,
    "nth_paragraph_first_word": nth_paragraph_first_word,
    "second_and_second_last_word": second_and_second_last_word,
    "ends_with_phrase": ends_with_phrase,
    "same_start_end_word": same_start_end_word,
    "exact_first_token": exact_first_token,
    "exact_last_token": exact_last_token,
    "exact_source_copy": exact_source_copy,
    "copy_change_first_word": copy_change_first_word,
    "copy_span_char_idx": copy_span_char_idx,
    "copy_request_n_times": copy_request_n_times,
    "reverse_word_order": reverse_word_order,
    "reverse_char_order": reverse_char_order,
    "square_brackets_every_token": square_brackets_every_token,
    "wrap_every_bigram": wrap_every_bigram,
    "keyword_in_nth_sentence": keyword_in_nth_sentence,
    "increment_word_count": increment_word_count,
    "last_word_first_next_sentence": last_word_first_next_sentence,
    "paragraph_last_first": paragraph_last_first,
    "same_char_count_three_sentences": same_char_count_three_sentences,
    "no_consecutive_same_first_letter": no_consecutive_same_first_letter,
    "only_arabic_letters": only_arabic_letters,
    "no_latin_letters": no_latin_letters,
    "no_persian_variants": no_persian_variants,
    "arabic_indic_only": arabic_indic_only,
    "arabic_indic_exact_count": arabic_indic_exact_count,
    "no_ascii_digits": no_ascii_digits,
    "mixed_digit_system_forbidden": mixed_digit_system_forbidden,
    "arabic_comma_exact_count": arabic_comma_exact_count,
    "arabic_semicolon_exact_count": arabic_semicolon_exact_count,
    "arabic_question_mark_presence": arabic_question_mark_presence,
    "question_mark_every_sentence": question_mark_every_sentence,
    "no_ascii_punct_equivalents": no_ascii_punct_equivalents,
    "cover_arabic_punct_set": cover_arabic_punct_set,
    "shadda_exact_count": shadda_exact_count,
    "sukun_exact_count": sukun_exact_count,
    "cover_tanween_set": cover_tanween_set,
    "no_diacritics": no_diacritics,
    "kashida_exact_count": kashida_exact_count,
    "no_zero_width_chars": no_zero_width_chars,
    "no_presentation_forms": no_presentation_forms,
    "arabic_diacritics_exact_count": arabic_diacritics_exact_count,
    "token_suffix_count": token_suffix_count,
    "definite_article_prefix_count": definite_article_prefix_count,
    "hamza_char_count": hamza_char_count,
    "hamza_forms_cover_all": hamza_forms_cover_all,
    "hamza_forms_exact_once_each": hamza_forms_exact_once_each,
    "morph_lookup_contains_target": morph_lookup_contains_target,
}
