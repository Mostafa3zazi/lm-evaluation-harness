import random
import re

import datasets

def doc_to_target(doc):
    return doc["answer"].strip()[1]


def doc_to_text(doc):
    header = build_prompt(instruction_lang="en", labels_lang="en", num_choices=4)
    return (
        f"{header}\n"
        f"What is the correct answer to this question: {doc['Question']}\n"
        f"Choices:\n"
        f"(A) {doc['choice1']}\n"
        f"(B) {doc['choice2']}\n"
        f"(C) {doc['choice3']}\n"
        f"(D) {doc['choice4']}\n"
        "Answer:"
    )

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
            "options_with_text": "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)),
        }
        return out_doc

    return dataset.map(_process_doc)

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
