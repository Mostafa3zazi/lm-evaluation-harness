
# def extract_code_blocks(text: str) -> str:
#     # Pattern to match ```...``` blocks
#     pattern = r"```(?:\w+)?\n?(.*?)\n?```"
#     # (+ ```) as we add the opening "```python" to the gen_prefix
#     matches = re.findall(pattern, r"```" + text, re.DOTALL)
#     # if no matches, try to match ```...``` blocks (after removing the language)
#     if not matches:
#         text_without_lang = re.sub(r"```python", "```", text)
#         matches = re.findall(pattern, text_without_lang, re.DOTALL)
#     if not matches:
#         return ""
#     else:
#         return matches[0]

# adapting the code from Eval360-V2/grader/mbpp_local
# Compile once for efficiency
import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]],
    predictions: Union[str, list[list[str]]],
) -> float:
    if isinstance(references, str):
        references = [references]

    if isinstance(predictions, str):
        predictions = [[predictions]]
    elif predictions and isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]

    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n?```",
    re.DOTALL,
)

_OPEN_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n",
)

_PYTHON_LANGS = {"python", "py", "python3", "python-3"}

_UNSAFE_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bsubprocess\.",
    r"\bPopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\b__import__\s*\(",
    r"\bimportlib\.",
    r"\bopen\s*\(",
    r"\bos\.(remove|unlink|rmdir|removedirs|rename|replace|chmod|chown|kill|killpg|fork|forkpty|chdir|fchdir|putenv|setuid|chroot|truncate)\s*\(",
    r"\bshutil\.(rmtree|move|chown)\s*\(",
    r"\b(socket|requests|urllib|httpx|ftplib|telnetlib|paramiko)\b",
    r"\b(resource|psutil|tkinter|joblib|ipdb)\b",
]

_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS))


def is_safe_code(code: str) -> bool:
    return bool(code) and _UNSAFE_RE.search(code) is None


def _is_python_lang(lang: str) -> bool:
    return lang.strip().lower() in _PYTHON_LANGS


def _strip_leading_comments_and_whitespace(code: str) -> str:
    lines = code.splitlines()
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("#"):
            i += 1
            continue

        break

    return "\n".join(lines[i:]).lstrip()


def _starts_with(code: str, prefixes: tuple[str, ...]) -> bool:
    cleaned = _strip_leading_comments_and_whitespace(code)
    return cleaned.startswith(prefixes)


def _choose_by_priority(blocks: list[str]) -> str:
    for prefixes in [
        ("import ", "import\n"),
        ("from ",),
        ("def ",),
        ("async def ",),
        ("class ",),
        ("@",),
    ]:
        matches = [block for block in blocks if _starts_with(block, prefixes)]
        if matches:
            return matches[-1]

    return blocks[-1] if blocks else ""


def extract_code_blocks(text: str) -> str:
    if not text:
        return ""

    matches = _CODE_BLOCK_RE.findall(text)

    if matches:
        python_blocks = [
            code.strip()
            for lang, code in matches
            if _is_python_lang(lang)
        ]

        if python_blocks:
            code = _choose_by_priority(python_blocks)
        else:
            all_blocks = [code.strip() for _, code in matches]
            code = _choose_by_priority(all_blocks)

    else:
        open_match = _OPEN_CODE_BLOCK_RE.search(text)

        if open_match:
            code = text[open_match.end():].strip()
        else:
            return ""

    if not is_safe_code(code):
        return ""

    return code


def build_predictions(
    resps: list[list[str]],
    docs: list[dict],
) -> list[list[str]]:
    return [
        [extract_code_blocks(r) for r in resp]
        for resp in resps
    ]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
