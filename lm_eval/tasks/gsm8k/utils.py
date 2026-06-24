from typing import Dict, List, Optional, Tuple

from math_verify import parse, verify
import re


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


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


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    # Extract from \\boxed{} if present
    boxed_answer = last_boxed_only_string(candidates)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                candidates = boxed_content
        except (AssertionError, IndexError):
            pass

    # math_verify
    # print("="*100)
    # print(candidates, doc["answer"].split('####')[-1].strip())
    # print("="*100)

    res = verify(parse(doc["answer"].split('####')[-1].strip()), parse(candidates))
    mathval = 1 if res else 0
    results = {
        "math_verify": mathval,
    }
    return results