def doc_to_text(doc):
    question = doc["question"]
    choices = doc["choices"]
    instructions = build_prompt('en',4)
    return (
        f"{instructions}\n"
        f"{question.strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )

def map_answer_to_label(answer_idx, lang):
    en_labels = ["A", "B", "C", "D", "E"]
    ar_labels = ["أ", "ب", "ج", "د", "هـ"]

    labels = en_labels if lang == "en" else ar_labels
    return labels[answer_idx]


def doc_to_target(doc):
    # return doc["answer"]
    return map_answer_to_label(doc["answer"], lang="en")

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

def build_prompt(lang: str, num_choices: int) -> str:
    if lang not in ["en", "ar"]:
        raise ValueError("lang must be 'en' or 'ar'")
    if num_choices < 2 or num_choices > 26:
        raise ValueError("num_choices must be between 2 and 26")

    # Generate letters
    letters_en = [chr(ord('A') + i) for i in range(num_choices)]
    letters_ar = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي",
                  "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر",
                  "ش", "ت", "ث", "خ", "ذ", "ض"][:num_choices]

    if lang == "en":
        instruction = "# Instruction\nReturn only the final answer as the English option letter inside double square brackets."
        options = ", ".join(f"[[{l}]]" for l in letters_en[:-1])
        if num_choices == 2:
            options = f"[[{letters_en[0]}]] or [[{letters_en[1]}]]"
        else:
            options += f", or [[{letters_en[-1]}]]"
        valid_line = f"Valid outputs are only:\n{options}"
    else:
        instruction = "# التعليمات\nأعد فقط الإجابة النهائية على شكل حرف الخيار باللغة العربية داخل أقواس مربعة مزدوجة."
        options = "، ".join(f"[[{l}]]" for l in letters_ar[:-1])
        if num_choices == 2:
            options = f"[[{letters_ar[0]}]] أو [[{letters_ar[1]}]]"
        else:
            options += f"، أو [[{letters_ar[-1]}]]"
        valid_line = f"المخرجات المسموح بها فقط هي:\n{options}"

    return f"{instruction}\n{valid_line}"


def add_options_with_text(dataset):
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