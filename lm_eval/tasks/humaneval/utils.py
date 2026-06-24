import re
import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n?```",
    re.DOTALL,
)

_UNSAFE_RE = re.compile(
    "|".join(
        [
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
    )
)


_PYTHON_LANGS = {"python", "py", "python3", "python-3"}


def is_safe_code(code: str) -> bool:
    return bool(code) and _UNSAFE_RE.search(code) is None


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


def _is_python_lang(lang: str) -> bool:
    return lang.strip().lower() in _PYTHON_LANGS


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
        open_match = re.search(
            r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n",
            text,
        )

        if open_match:
            code = text[open_match.end():].strip()
        else:
            code = text.strip()

    if not is_safe_code(code):
        return ""

    return code


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
):
    global compute_

    assert k is not None

    if isinstance(k, int):
        k = [k]

    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )

    return res[0]


def build_predictions_instruct(
    resps: list[list[str]],
    docs: list[dict],
) -> list[list[str]]:
    return [
        [
            extract_code_blocks(r)
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions(
    resps: list[list[str]],
    docs: list[dict],
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + r
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]