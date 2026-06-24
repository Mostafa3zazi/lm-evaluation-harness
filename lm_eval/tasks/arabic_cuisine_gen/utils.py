import evaluate


def doc_to_text(doc):
    options = "\n".join(doc["options"])
    instruction = build_prompt(instruction_lang="ar", labels_lang="en", num_choices=len(doc["options"]))
    return (
        f"{instruction}\n"
        "أجب عن السؤال باستخدام الخيار المناسب من بين A أو B أو C أو D أو E. "
        "الرجاء الرد بالحرف الصحيح فقط: A أو B أو C أو D أو E دون أي شرح أو معلومات إضافية.\n"
        f"السؤال: {doc['question']}\n"
        "الاختيارات:\n"
        f"{options}\n"
        "الإجابة:"
    )


def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

alpha = ['A', 'B', 'C', 'D', 'E']
def doc_to_target(doc):
    options = doc_to_choice(doc)
    index = options.index(doc["correct_answer"][0])
    return alpha[index]


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


#! make sure to add process_docs: !function utils.add_options_with_text and make sure there is no conflict in all subtasks.
def add_options_with_text(dataset):
    def _add_field(doc):
        # implement the logic to convert options to text (done)
        options_text = "\n".join(doc["options"])
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)
