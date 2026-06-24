"""
Copied from https://github.com/LLM360/Eval360-V2/blob/master/scheduler/grader/math.py
"""
import re
from math_verify import parse, verify


def parse_answer_with_verify(text):
    """Parse answer using math_verify library with fallback to regex patterns."""
    # ---- Fix 4: formatting normalization ----
    text = re.sub(r"(?s).*#### ", "", text)          # keep only what's after the last "#### "
    text = re.sub(r"\\text\s*\{[^{}]*\}", "", text)  # drop \text{...} labels/units   (doc 17)
    text = text.replace("\\$", "$")                  # \$9.00 -> $9.00                (1193/1299/1312)
    text = re.sub(r"(\d)\{,?\}?(\d)", r"\1\2", text) # 2{,}400 -> 2400
    text = re.sub(r"(\d)\{(\d+)\}", r"\1\2", text)   # 2{400}  -> 2400                (1197)
    text = re.sub(r"\.$", "", text)                  # strip a trailing period
    
    try:
        # Prefer $...$ wrapping: raw LaTeX like \sqrt{51} or \left(...\right)
        # parses incorrectly or not at all without math delimiters
        parsed = parse(f"${text}$")
        if parsed:
            return parsed
        parsed = parse(text)
        if parsed:
            return parsed
    except Exception:
        pass

    # Fallback to regex patterns for extraction
    answer_patterns = [
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        r"=\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?\.?\s*(?:The answer|$)",
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # take the last match (usually the final answer)
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                try:
                    return parse(ans)
                except Exception:
                    return ans

    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    m = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if m:
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            try:
                return parse(ans)
            except:
                return ans

    # last fallback: find the last number in the last complete sentence
    sentences = text.split('.')
    for sent in reversed(sentences):
        # skip sentences containing Human/Assistant (possibly irrelevant content)
        if 'Human:' in sent or 'Assistant:' in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            num = numbers[-1].lstrip('0') or '0'
            try:
                return parse(num)
            except Exception:
                return num
    return None


def compare_answers(answer, gold_raw):
    """Compare parsed prediction against raw ground truth string."""
    if not answer or gold_raw is None:
        return False
    try:
        answer_parsed = parse_answer_with_verify(answer)
        # Try with gold as raw string (verify can parse simple expressions internally)
        if verify(answer, gold_raw):
            return True
        # For complex LaTeX (tuples, intervals, radicals with \left/\right),
        # parse the ground truth explicitly and call verify(gold, pred)
        gold_parsed = parse_answer_with_verify(gold_raw)
        if gold_parsed:
            return verify(gold_parsed, answer_parsed)
    except Exception:
        pass
    return False


#Before any changes
# def math_verify_grader(reference, prediction=None):
#     # print("REF:",references,"\nPRED:",predictions)
#     # print('\n')
#     if prediction is None: return 0
#     paresed_prediction = parse_answer_with_verify(prediction)
#     if paresed_prediction is None:
#         return 0
#     return compare_answers(paresed_prediction, reference)

#Fix Issue 1: Wrong function signature
# def math_verify_grader(references=None, predictions=None, **kwargs):
#     # lm-eval calls: math_verify_grader(references=[gold], predictions=[pred], **kwargs)
#     def _unwrap(x):
#         return x[0] if isinstance(x, (list, tuple)) and x else x

#     reference = _unwrap(references)
#     prediction = _unwrap(predictions)
#     if prediction is None or reference is None:
#         return 0

#     parsed_prediction = parse_answer_with_verify(prediction)
#     if parsed_prediction is None:
#         return 0
#     return 1 if compare_answers(parsed_prediction, reference) else 0

#Issue 2 was with the Q: in the yaml

# Fix Issue 3: parse each side once and verify directly (no compare_answers re-parse)
def math_verify_grader(references=None, predictions=None, **kwargs):
    # lm-eval calls: math_verify_grader(references=[gold], predictions=[pred], **kwargs)
    def _unwrap(x):
        return x[0] if isinstance(x, (list, tuple)) and x else x

    reference = _unwrap(references)
    prediction = _unwrap(predictions)
    if prediction is None or reference is None:
        return 0

    pred_parsed = parse_answer_with_verify(prediction)
    gold_parsed = parse_answer_with_verify(reference)
    if not pred_parsed or not gold_parsed:
        return 0
    try:
        return 1 if verify(gold_parsed, pred_parsed) else 0
    except Exception:
        return 0