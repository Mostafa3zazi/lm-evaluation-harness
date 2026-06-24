# for subtask 1A
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import re

# def post_process(doc, results):
#     gold = doc["label_word"]
#     label = results[0].strip()
#     label = re.sub(r"\s+", "", label)
#     return {"eval": (label, gold)}



# def evaluate(items):
#     predicted_labels, true_labels = zip(*items)
#     return {"Accuracy": accuracy_score(true_labels, predicted_labels)}

# def evaluate(items):
#     return {"acc": 0}


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




# for subtask 1B
options = ['CorrectAyah', 'WrongAyah', 'CorrectHadith', 'WrongHadith']
options_letter = ['A', 'B', 'C', 'D']

def doc_to_choice(doc):
    return options

def doc_to_target(doc):
    return options_letter[options.index(doc["Label"])]

def add_options_with_text(dataset):
    def _add_field(doc):
        doc["options_with_text"] = (
        "A. CorrectAyah\n"
        "B. WrongAyah\n"
        "C. CorrectHadith\n"
        "D. WrongHadith"
        )
        return doc
    return dataset.map(_add_field)

def doc_to_text(doc):
    full_span = doc["full_span"]

    return (
        f"{build_prompt(instruction_lang='en',labels_lang='en', num_choices= 4)}\n"
        "Classify the following text into one of these four options:\n\n"
        "- A. CorrectAyah: exact Qur'an verse, written correctly\n"
        "- B. WrongAyah: Qur'an verse but incorrect/incomplete\n"
        "- C. CorrectHadith: authentic Hadith, written correctly\n"
        "- D. WrongHadith: Hadith but incorrect/incomplete\n\n"
        f'Text: "{full_span}"\n\n'
        "Output only one option letter. A, B, C or D.\n"
        "The chosen option is:"
    )