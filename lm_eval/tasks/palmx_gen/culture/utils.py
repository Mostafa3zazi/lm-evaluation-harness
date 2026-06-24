"""Scoring helpers for the PalmX 2025 subtask 1 (culture) generation task.

Each example has the columns: id, question, A, B, C, D, answer
where `answer` is one of the letters "A", "B", "C", "D".
This is the open-generation variant: the model is asked to emit the answer
letter inside double square brackets (e.g. "[[D]]"); the filter extracts that
label and it is compared against the gold letter with exact match.
"""

CHOICES = ["A", "B", "C", "D"]


def doc_to_target(doc):
    return doc["answer"].strip()


def doc_to_text(doc):
    header = build_prompt(instruction_lang="ar", labels_lang="en", num_choices=4)
    return (
        f"{header}\n"
        f"السؤال: {doc['question']}\n"
        f"A. {doc['A']}\n"
        f"B. {doc['B']}\n"
        f"C. {doc['C']}\n"
        f"D. {doc['D']}\n"
        f"الإجابة:"
    )


import re
BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def normalize_label(label):
    return str(label).strip().lower()


def extract_bracketed_label(resps, docs):
    def extract(resp):
        if not isinstance(resp, str):
            return "[invalid]"

        matches = BRACKETED_LABEL_RE.findall(resp)
        if not matches:
            return "[invalid]"

        return matches[-1]

    return map(lambda r: extract(r[0] if r else ""), resps)


def exact_match_normalized_label(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else "[invalid]"
    reference = references[0] if references else "[invalid]"
    return {
        "exact_match": float(
            normalize_label(prediction) == normalize_label(reference)
        )
    }


def build_prompt(instruction_lang: str, labels_lang: str, num_choices: int) -> str:
    if instruction_lang not in ["en", "ar"]:
        raise ValueError("instruction_lang must be 'en' or 'ar'")
    if labels_lang not in ["en", "ar"]:
        raise ValueError("labels_lang must be 'en' or 'ar'")
    if num_choices < 2 or num_choices > 26:
        raise ValueError("num_choices must be between 2 and 26")

    # Generate letters
    letters_en = [chr(ord('A') + i) for i in range(num_choices)]
    letters_ar = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي",
                  "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر",
                  "ش", "ت", "ث", "خ", "ذ", "ض"][:num_choices]

    # Choose label set
    letters = letters_en if labels_lang == "en" else letters_ar

    # Instruction text
    if instruction_lang == "en":
        instruction = "# Instruction\nReturn only the final answer as the option letter inside double square brackets."
        sep = ", "
        or_word = "or"
        valid_prefix = "Valid outputs are only:\n"
    else:
        instruction = "# التعليمات\nأعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة."
        sep = "، "
        or_word = "أو"
        valid_prefix = "المخرجات المسموح بها فقط هي:\n"

    # Build options string
    if num_choices == 2:
        options = f"[[{letters[0]}]] {or_word} [[{letters[1]}]]"
    else:
        options = sep.join(f"[[{l}]]" for l in letters[:-1])
        options += f"{sep}{or_word} [[{letters[-1]}]]"

    valid_line = f"{valid_prefix}{options}"

    return f"{instruction}\n{valid_line}"


# Builds the `options_with_text` field consumed by the LLM-as-a-judge step
# (judge_vllm.py -> build_judge_prompt). The judge reads this block plus the
# gold letter (doc_to_target) and the model's raw response to decide correctness.
def add_options_with_text(dataset):
    def _add_field(doc):
        doc["options_with_text"] = (
            f"A. {doc['A']}\n"
            f"B. {doc['B']}\n"
            f"C. {doc['C']}\n"
            f"D. {doc['D']}"
        )
        return doc
    return dataset.map(_add_field)
