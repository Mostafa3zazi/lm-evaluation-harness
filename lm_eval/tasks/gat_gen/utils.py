PROMPTS = {
    "default": "التعليمات: فيما يلي سؤال، يتبعه أربعة اختيارات. المطلوب هو: اختيار الإجابة الصحيحة.",
    "analogy": "التعليمات: في السؤال، كلمتان ترتبطان بعلاقة معينة، تتبعهما أربعة أزواج من الكلمات، أحدها ترتبط فيه الكلمتانبعلاقة مشابهة للعلاقة التي بين الكلمتين في بداية السؤال. المطلوب هو: اختيار الإجابة الصحيحة.",
    "completion": "التعليمات: في السؤال جملة، تليها أربعة اختيارات، أحدها يكمل الفراغ أو الفراغات في الجملة إكمالاً صحيحاً. المطلوب هو: اختيار الإجابة الصحيحة.",
    "contextual": "التعليمات: في السؤال جملة، تليها أربع كلمات من الجملة. المطلوب هو: تحديد الكلمة التي لا يتفق معناها مع المعنى العام للجملة. (الخطأ ليس إملائياً ولا نحوياً)",
    "reading": "التعليمات: السؤال يتعلق بالنص الذي يسبقه، بعد السؤال يوجد أربعة اختيارات، أحدها صحيح. المطلوب هو: قراءة النص بعناية، واختيار الإجابة الصحيحة."
}

def make_prompt(key):
    prompt_template = PROMPTS.get(key, PROMPTS["default"])
    def doc_to_text(doc):
        rubric = ""
        try:
            rubric_text = doc.get("النص", "")
            if rubric_text:
                rubric = f"النص: {rubric_text}\n"
        except Exception:
            pass

        question = f"السؤال: {doc['السؤال']}"
        a = str(doc.get("أ", ""))
        b = str(doc.get("ب", ""))
        c = str(doc.get("ج", ""))
        d = str(doc.get("د", ""))
        
        text = (
            f"{prompt_template}\n"
            f"{rubric}"
            f"{question}\n"
            f"أ: {a}\n"
            f"ب: {b}\n"
            f"ج: {c}\n"
            f"د: {d}\n"
        )
        instruction = build_prompt(labels_lang="ar", instruction_lang="ar", num_choices=4)
        return f"{instruction}\n{text}"
    return doc_to_text


doc_to_text_default = make_prompt("default")
doc_to_text_analogy = make_prompt("analogy")
doc_to_text_reading = make_prompt("reading")
doc_to_text_contextual = make_prompt("contextual")
doc_to_text_completion = make_prompt("completion")

def add_options_with_text(dataset):
    def _add_field(doc):
        a = str(doc.get("أ", ""))
        b = str(doc.get("ب", ""))
        c = str(doc.get("ج", ""))
        d = str(doc.get("د", ""))
        doc["options_with_text"] = f"أ: {a}\nب: {b}\nج: {c}\nد: {d}"
        return doc
    return dataset.map(_add_field)


def doc_to_choice(doc):
    """
    Returns the available multiple-choice labels.
    The evaluation framework will score each one separately.
    """
    # Your dataset always includes four options: أ, ب, ج, د
    return ["أ", "ب", "ج", "د"]


def doc_to_target(doc):
    """
    Returns the correct answer label exactly as it appears in the dataset.
    Example: "ب"
    """
    return doc["الإجابة_الصحيحة"].strip()

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