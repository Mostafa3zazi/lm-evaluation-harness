def doc_to_text(doc):
    flores_passage = doc["flores_passage"]
    question = doc["question"]
    mc_answer1 = doc["mc_answer1"]
    mc_answer2 = doc["mc_answer2"]
    mc_answer3 = doc["mc_answer3"]
    mc_answer4 = doc["mc_answer4"]
    instruction = build_prompt(instruction_lang="ar", labels_lang="en", num_choices=4)
    return (
        f"{instruction}\n"
        f"P: {flores_passage}\n"
        f"Q: {question.strip()}\n"
        f"A: {mc_answer1}\n"
        f"B: {mc_answer2}\n"
        f"C: {mc_answer3}\n"
        f"D: {mc_answer4}\n"
        "Answer:"
    )

def doc_to_target(doc):
    correct_answer_num = doc["correct_answer_num"]
    index = ["1", "2", "3", "4"].index(correct_answer_num)
    label_list = ["A", "B", "C", "D"]
    return label_list[index]

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
        mc_answer1 = doc["mc_answer1"]
        mc_answer2 = doc["mc_answer2"]
        mc_answer3 = doc["mc_answer3"]
        mc_answer4 = doc["mc_answer4"]
        doc["options_with_text"] = f"A. {mc_answer1}\nB. {mc_answer2}\nC. {mc_answer3}\nD. {mc_answer4}"
        return doc
    return dataset.map(_add_field)