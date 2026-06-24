import re
from math_verify import parse, verify


def doc_to_text(doc: dict) -> str:
    question_key = next(k for k in doc.keys() if k.lower() in {"question", "problem"})
    return (
        "Solve the problem. After your full response, add this exact final line:\n"
        "The final answer in double square brackets is: [[FINAL_ANSWER]].\n\n"
        "Replace FINAL_ANSWER with the final numeric answer.\n\n"
        f"Question: {doc[question_key]}\n"
        "Answer:"
    )


def _extract_answer_from_response(response: str) -> str:
    # Try to extract answer from $...$ format first
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    # Extract from \\boxed{} if present
    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass

    bracketed_answer = last_double_bracketed_string(response)
    if bracketed_answer is not None:
        try:
            bracketed_content = remove_double_brackets(bracketed_answer)
            if bracketed_content is not None:
                answer = bracketed_content
        except (AssertionError, IndexError):
            pass

    return answer


def extract_answer(resps, docs):
    def extract(resp):
        response = resp[0] if resp else ""
        if not isinstance(response, str):
            response = ""
        return _extract_answer_from_response(response)

    return map(extract, resps)


def parse_answer_with_verify(text):
    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    try:
        parsed = parse(f"${text}$")
        if parsed:
            return parsed
        parsed = parse(text)
        if parsed:
            return parsed
    except Exception:
        pass

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
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer = matches[-1].replace(",", "").strip().rstrip(".")
            if answer:
                try:
                    return parse(answer)
                except Exception:
                    return answer

    sentence_end_pattern = (
        r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)"
        r"\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    )
    match = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if match:
        answer = match.group(1).replace(",", "").strip().rstrip(".")
        if answer:
            try:
                return parse(answer)
            except Exception:
                return answer

    for sentence in reversed(text.split(".")):
        if "Human:" in sentence or "Assistant:" in sentence:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sentence)
        if numbers:
            number = numbers[-1].lstrip("0") or "0"
            try:
                return parse(number)
            except Exception:
                return number

    return None


def compare_answers_with_verify(prediction, reference):
    parsed_prediction = parse_answer_with_verify(prediction)
    if not parsed_prediction or reference is None:
        return False

    try:
        if verify(parsed_prediction, reference):
            return True
    except Exception:
        pass

    parsed_reference = parse_answer_with_verify(reference)
    if not parsed_reference:
        return False

    for gold, target in (
        (parsed_reference, parsed_prediction),
        (parsed_prediction, parsed_reference),
    ):
        try:
            if verify(gold, target):
                return True
        except Exception:
            pass

    return False


def aime_math_verify(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else ""
    reference = references[0] if references else ""
    return {"exact_match": int(compare_answers_with_verify(prediction, reference))}


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def remove_double_brackets(s):
    left = "[["
    right = "]]"

    assert s[: len(left)] == left
    assert s[-len(right) :] == right

    return s[len(left) : -len(right)]


def last_double_bracketed_string(string):
    start_idx = string.rfind("[[")
    if start_idx < 0:
        return None

    end_idx = string.find("]]", start_idx)
    if end_idx < 0:
        return None

    return string[start_idx : end_idx + 2]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
