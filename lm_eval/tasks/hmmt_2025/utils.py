import re
from fractions import Fraction

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def doc_to_text(doc):
    question = doc["messages"][-1]["content"]
    return f"{question}\n\nYou MUST put your answer in \\boxed{{}}"


def doc_to_target(doc):
    return str(doc["ground_truth"]).strip()


def extract_last_boxed_expression(text):
    pattern = r"\\boxed\{"
    if text is None:
        return ""

    matches = list(re.finditer(pattern, text))
    if not matches:
        return ""

    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    i = start

    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start : i - 1].strip()
    return ""


def extract_final_answer(response):
    if not response:
        return ""

    match_boxed = extract_last_boxed_expression(response)
    if match_boxed:
        return match_boxed

    pattern = r"(?i)(?:\*\*)?answer:(?:\*\*)?\s*(.+)"
    match_answer = re.findall(pattern, response)
    if match_answer:
        return match_answer[-1].strip()

    match_number = re.search(r"(\d+)(?!.*\d)", response)
    if match_number:
        return match_number.group(1).strip()

    return "No answer given"


@register_filter("hmmt_final_answer")
class HMMTFinalAnswerFilter(Filter):
    def apply(self, resps, docs):
        filtered = []
        for instance_resps in resps:
            response = instance_resps[0] if instance_resps else ""
            filtered.append(extract_final_answer(response))
        return filtered


def _strip_math_text(text):
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    return text


def _simple_latex_to_text(text):
    text = _strip_math_text(text)
    text = text.replace("\\cdot", "*")
    text = text.replace("\\pi", "pi")

    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        new_text = frac_pattern.sub(r"(\1)/(\2)", text)
        if new_text == text:
            break
        text = new_text

    sqrt_pattern = re.compile(r"\\sqrt\{([^{}]+)\}")
    while True:
        new_text = sqrt_pattern.sub(r"sqrt(\1)", text)
        if new_text == text:
            break
        text = new_text

    return text


def _as_fraction(text):
    text = _simple_latex_to_text(text)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return Fraction(text)
    if re.fullmatch(r"\(?[+-]?\d+\)?/\(?[+-]?\d+\)?", text):
        numerator, denominator = text.replace("(", "").replace(")", "").split("/")
        return Fraction(int(numerator), int(denominator))
    return None


def _split_answers(text):
    return [part.strip() for part in str(text).split(",")]


def _math_verify_equiv(reference, prediction):
    try:
        from math_verify import parse, verify
    except Exception:
        return None

    try:
        return bool(verify(parse(str(reference)), parse(str(prediction))))
    except Exception:
        return None


def _sympy_equiv(reference, prediction):
    try:
        import sympy as sp
    except Exception:
        return None

    try:
        ref = sp.sympify(_simple_latex_to_text(reference))
        pred = sp.sympify(_simple_latex_to_text(prediction))
        return bool(sp.simplify(ref - pred) == 0)
    except Exception:
        return None


def math_equivalent(reference, prediction):
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    verified = _math_verify_equiv(reference, prediction)
    if verified is not None:
        return verified

    ref_parts = _split_answers(reference)
    pred_parts = _split_answers(prediction)
    if len(ref_parts) == len(pred_parts) and len(ref_parts) > 1:
        unmatched = pred_parts[:]
        for ref_part in ref_parts:
            for idx, pred_part in enumerate(unmatched):
                if math_equivalent(ref_part, pred_part):
                    unmatched.pop(idx)
                    break
            else:
                return False
        return True

    ref_frac = _as_fraction(reference)
    pred_frac = _as_fraction(prediction)
    if ref_frac is not None and pred_frac is not None:
        return ref_frac == pred_frac

    sympy_result = _sympy_equiv(reference, prediction)
    if sympy_result is not None:
        return sympy_result

    return _strip_math_text(reference) == _strip_math_text(prediction)


def process_results(doc, results):
    prediction = results[0]
    reference = doc_to_target(doc)
    return {"math_equiv": float(math_equivalent(reference, prediction))}
