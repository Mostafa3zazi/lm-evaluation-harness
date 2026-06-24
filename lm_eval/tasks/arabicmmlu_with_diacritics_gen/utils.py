

import re
PROMPT = "This is a {}. Select the correct answer!\n\nQuestion: {}\n{}\n\nAnswer:"

level_en = {
    "Primary": "primary school",
    "Middle": "middle school",
    "High": "high school",
    "Univ": "university",
    "Prof": "professional",
}

alpa = ["A.", "B.", "C.", "D.", "E."]
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

def doc_to_text(doc):
    """
    Refactoring `prepare_data_en` to fit with the lm harness framework.
    https://github.com/mbzuai-nlp/ArabicMMLU/blob/main/util_prompt.py
    """

    level = "" if not doc["Level"] else " for " + level_en[doc["Level"]]
    country = "" if not doc["Country"] else " in " + doc["Country"]
    main_meta_data = f"{doc['Subject']} question{level}{country}"

    question = (
        doc["Question"]
        if not doc["Context"]
        else f"{doc['Context']}\n\n{doc['Question']}"
    )

    options = []
    for i, opt in enumerate(
        ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    ):
        if not doc[opt]:
            break
        options.append(f"{alpa[i]} {doc[opt]}")

    doc_text = PROMPT.format(main_meta_data, question, "\n".join(options))
    instruction = build_prompt(lang="en", num_choices=len(options))
    return f"{instruction}\n{doc_text}"

def doc_to_choice(doc):
    return [alpa[i][0] for i in range(5) if doc[f"Option {i + 1}"]]


import re
def filter_arabic_mmlU(**kwarg):
    """
    Standard filter signature for many harness tasks.
    It usually takes a list of strings and returns a list of processed strings.
    """
    print(kwarg)
    results = []
    processed_results = []
    for res in results:
        # 1. Clean whitespace
        cleaned = res.strip()
        # 2. Extract first A-D or أ-د
        match = re.search(r'[A-Dأ-د]', cleaned)
        if match:
            processed_results.append(match.group(0))
        else:
            processed_results.append(cleaned)
    return processed_results

def process_results(*args, **kwargs):
    # This will never throw a TypeError because it accepts everything.
    # We just need to find 'doc' and 'results' inside them.
    
    # Usually: args[0] is doc, args[1] is results
    doc = kwargs.get('doc', args[0] if len(args) > 0 else None)
    results = kwargs.get('results', args[1] if len(args) > 1 else None)
    
    # ... (rest of extraction logic) ...
def add_options_with_text(dataset):
    def _add_field(doc):
        doc["options_with_text"] = '\n'.join(
            f"{alpa[i]} {doc[f'Option {i + 1}']}"
            for i in range(5) if doc[f'Option {i + 1}']
        )
        return doc
    return dataset.map(_add_field)