labels = ["A", "B", "C", "D", "E"]
def doc_to_text(doc):
    subject = doc.get("subject", "subject")
    level = doc.get("level", "level")
    question = doc["question"]

    # Collect non-empty options
    options = []
    for i in range(5):
        opt = doc.get(f"option_{i}", "").strip()
        if opt:
            options.append(f"{labels[i]}) {opt}")

    options_text = "\n".join(options)

    prompt = (
        f"You are an expert in {subject} at the {level} level. Analyze the given multiple-choice question and "
        f"provide the correct answer using this approach:\n"
        f"1. Carefully read the question and options\n"
        f"2. Identify core {subject} concepts and required knowledge\n"
        f"3. Analyze each option for relevance, accuracy, and consistency\n"
        f"4. Consider {subject}-specific context and factors\n"
        f"5. Use elimination and comparative analysis\n"
        f"6. Select the most accurate answer\n"
        f"Maintain objectivity, consider {subject}-specific sensitivities, and base your decision on verifiable facts "
        f"and sound logical reasoning within {subject} at the {level}.\n"
        f"Question: {question}\n{options_text}\nCorrect option letter is:"
    )
    instruction = build_prompt(instruction_lang="en", labels_lang="en", num_choices=len(options))
    return f"{instruction}\n{prompt}"


def doc_to_choice(doc):
    """Return non-empty choices in order."""
    choices = []
    for i in range(5):
        choice = doc.get(f"option_{i}", "").strip()
        if choice:
            choices.append(str(i))
    return choices


def doc_to_target(doc):
    """Return the correct option number as a string if the option is non-empty."""
    correct = doc.get("correct_option")
    return labels[correct]


def add_options_with_text(dataset):
    def _add_field(doc):
        options = []
        for i in range(5):
            opt = doc.get(f"option_{i}", "").strip()
            if opt:
                options.append(f"{labels[i]}) {opt}")

        options_text = "\n".join(options)
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)

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