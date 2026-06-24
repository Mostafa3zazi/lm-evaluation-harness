PROMPT = """Example 1:
    Question: ما مدة المسح على الخفين للمقيم؟
    A) يوم وليلة
    B) ثلاثة أيام بلياليهن
    C) يومان وليلتان
    D) أسبوع كامل
    Answer: [[A]]
    
    Example 2:
    Question: توفي عن أب، وأخوين شقيقين، وابن أخ شقيق، وعمين شقيقين، وأم، وبنتين، و زوجة، فما نصيب الأم؟
    A) الثلث
    B) الربع
    C) السدس
    D) الثمن
    E) النصف
    F) لا شيء
    Answer: [[C]]
    
    Now answer the following question:


You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.

Question: {}

{}

Please respond using **only one English letter** from the following: {}
Do not write any explanation or additional text."""

alpa = ["A", "B", "C", "D", "E", "F"]


def doc_to_text(doc):
    """
    Converts a document row from the CSV file into the formatted prompt text.
    Expected keys: id_question, question, option1–option6, label, level.
    """
    options = []
    valid_letters = []
    for i, opt_key in enumerate(
        ["option1", "option2", "option3", "option4", "option5", "option6"]
    ):
        if opt_key in doc:
            options.append(f"{alpa[i]}) {doc[opt_key]}")
            valid_letters.append(alpa[i])

    options_text = "\n".join(options)
    valid_letters_str = "/".join(valid_letters)

    doc_text = PROMPT.format(doc["question"], options_text, valid_letters_str)
    return (build_prompt(instruction_lang='en',labels_lang='en', num_choices= len(options)) + "\n" + doc_text)

def doc_to_choice(doc):
    """
    Returns list of all option letters for LM Harness evaluation.
    """
    return [alpa[i] for i in range(6) if f"option{i+1}" in doc]


def doc_to_target(doc):
    """
    Returns the correct answer letter (e.g., 'A', 'B', ...).
    """
    return doc["label"].strip()


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

def add_options_with_text(dataset):
    def _add_field(doc):
        options = []
        for i, opt_key in enumerate(
            ["option1", "option2", "option3", "option4", "option5", "option6"]
        ):
            if opt_key in doc:
                options.append(f"{alpa[i]}) {doc[opt_key]}")

        options_text = "\n".join(options)
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)