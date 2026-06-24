import re
import ast

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


def doc_to_text(query):
    instruction = build_prompt(instruction_lang="ar", labels_lang="ar", num_choices=4)
    return f"{instruction}\n{query}"

def doc_to_choice(doc):
    return ["أ", "ب", "ج", "د"]

def doc_to_target(doc):
    try:
        doc['correct_choice'] = doc['self_answer']
    except:
        pass
    correct = doc.get("correct_choice").strip()
    return str(correct)



def add_options_with_text_native(dataset):
    def _add_field(doc):
        labels, texts = native_parse_choices(doc)
        options_str = ""
        options_str += "".join([f"{label}. {text}\n" for label, text in zip(labels, texts)])
        doc["options_with_text"] = options_str
        return doc
    return dataset.map(_add_field)
#Native
def process_docs(domain):
    def filter_fn(dataset):
        def _add_field(doc):
            labels, texts = native_parse_choices(doc)
            options_str = ""
            options_str += "".join([f"{label}. {text}\n" for label, text in zip(labels, texts)])
            doc["options_with_text"] = options_str
            return doc
        dataset = dataset.filter(lambda doc: doc.get("domain") == domain)
        return dataset.map(_add_field)
    return filter_fn

process_docs_biology = process_docs("Biology")
process_docs_math = process_docs("Math")
process_docs_physics = process_docs("Physics")
process_docs_chemistry = process_docs("Chemistry")
process_docs_geography = process_docs("Geography")

def native_parse_choices(doc):
    raw_choices_input = doc["choices"]
    raw_choices = ast.literal_eval(raw_choices_input) if isinstance(raw_choices_input, str) else raw_choices_input
    labels, texts = [], []
    for i, choice in enumerate(raw_choices):
        match = re.match(r"^\((.)\)\s*(.*)", choice)
        if match:
            labels.append(match.group(1).strip())
            texts.append(match.group(2).strip())
        else:
            raise ValueError(f"Malformed choice: {choice}")
    labels = [re.sub(r"^\)?\s*", "", l).strip(" []'\"\n") for l in labels]
    return labels, texts

def doc_to_text_native(doc):
    labels, texts = native_parse_choices(doc)
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"
    question_text = doc["question_text"].strip()

    query = f"{instruction}{question_text}\n"
    query += "".join([f"{label}. {text}\n" for label, text in zip(labels, texts)])
    query += "الإجابة:"

    return doc_to_text(query)

#Synthetic
def synthetic_parse_choices(raw, labels):
    if all(lbl in raw for lbl in labels):
        positions_and_labels = sorted((raw.find(lbl), lbl) for lbl in labels if raw.find(lbl) != -1)
        label_to_text = {
            lbl: raw[pos + len(lbl): positions_and_labels[i + 1][0] - 1 if i < 3 else len(raw)].strip()
            for i, (pos, lbl) in enumerate(positions_and_labels)
        }
        return [label_to_text[lbl] for lbl in labels]
    elif "," in raw:
        parts, buffer, depth = [], "", 0
        for ch in raw:
            if ch == "(": depth += 1
            elif ch == ")": depth = max(depth - 1, 0)
            if ch == "," and depth == 0:
                parts.append(buffer.strip()); buffer = ""
            else:
                buffer += ch
        parts.append(buffer.strip())

        if len(parts) != 4:
            raise ValueError(f"Expected 4 top-level commas, got {len(parts)}: {parts!r}")

        return [part[2:].strip() for part in parts]
    else:
        raise ValueError(f"Cannot determine how to split choices: {raw!r}")


def doc_to_text_synthetic(doc):
    labels = ["أ)", "ب)", "ج)", "د)"]
    raw = doc["choices"]
    choices = synthetic_parse_choices(raw, labels)
    choices = [re.sub(r"^\)?\s*", "", c).strip(" []'\"\n") for c in choices]
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
    arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
    valid_keys_arabic = list(latin_to_arabic.values())
    self_answer_arabic = doc["self_answer"].strip()
    self_answer_latin = arabic_to_latin.get(self_answer_arabic)        
        
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    question = doc["question"]
    query = f"{instruction}{question}\n"
    for arab_label, choice_text in zip(valid_keys_arabic, choices):
        choice_text = re.sub(r"^\)?\s*", "", choice_text).strip(" []'\"\n")
        query += f"{arab_label}. {choice_text}\n"
    query += "الإجابة:"

    return doc_to_text(query)

def add_options_with_text_synthetic(dataset):
    def _add_field(doc):
        labels = ["أ)", "ب)", "ج)", "د)"]
        raw = doc["choices"]
        choices = synthetic_parse_choices(raw, labels)
        choices = [re.sub(r"^\)?\s*", "", c).strip(" []'\"\n") for c in choices]
        latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
        arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
        valid_keys_arabic = list(latin_to_arabic.values())
        options_str = ""
        for arab_label, choice_text in zip(valid_keys_arabic, choices):
            choice_text = re.sub(r"^\)?\s*", "", choice_text).strip(" []'\"\n")
            options_str += f"{arab_label}. {choice_text}\n"
        doc["options_with_text"] = options_str
        return doc
    return dataset.map(_add_field)