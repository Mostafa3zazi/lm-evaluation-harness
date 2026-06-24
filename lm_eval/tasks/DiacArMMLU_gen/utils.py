"""
Utility functions for Dialectal Arabic MMLU (DiacArMMLU) task.
Provides process_docs functions to filter examples by dialect and domain.
"""


def _make_process_docs(dialect: str, domain: str):
    """Factory function to create a process_docs filter for a specific dialect and domain."""
    def process_docs_fn(dataset):
        def _add_field(doc):
            choices = doc["choices"]
            doc["options_with_text"] = (
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}"
            )
            return doc
        dataset = dataset.filter(lambda doc: doc["dialect"] == dialect and doc["domain"] == domain)
        return dataset.map(_add_field)
    return process_docs_fn


# Pre-generate process_docs functions for all dialect+domain combinations
DIALECTS = ["EGY", "ENG", "KSA", "MAG", "MSA", "SYR", "UAE"]

DOMAINS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_computer_science",
    "college_medicine",
    "conceptual_physics",
    "elementary_mathematics",
    "global_facts",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_psychology",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "international_law",
    "management",
    "marketing",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_law",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


# Dynamically create process_docs functions for each dialect+domain combo
# These will be called as: process_docs_EGY_astronomy, process_docs_EGY_anatomy, etc.
for _dialect in DIALECTS:
    for _domain in DOMAINS:
        # Create function name like "process_docs_EGY_astronomy"
        func_name = f"process_docs_{_dialect}_{_domain}"
        # Create the filter function and add to module globals
        globals()[func_name] = _make_process_docs(_dialect, _domain)


# Also create per-dialect process_docs (for dialect-level aggregation)
def _make_dialect_process_docs(dialect: str):
    """Factory function to create a process_docs for a specific dialect (all domains)."""
    def process_docs_fn(dataset):
        return dataset.filter(lambda doc: doc["dialect"] == dialect)
    return process_docs_fn


for _dialect in DIALECTS:
    func_name = f"process_docs_{_dialect}"
    globals()[func_name] = _make_dialect_process_docs(_dialect)

def doc_to_target(doc):
    labels = ["A", "B", "C", "D"]
    return labels[int(doc["answer"])]

def doc_to_text(doc):
    question = doc["question"]
    choices = doc["choices"]
    instructions = build_prompt(instruction_lang='en', labels_lang='en', num_choices=4)

    return (
        f"{instructions}\n"
        f"{question.strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )

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

# def add_options_with_text(dataset):
#     def _add_field(doc):
#         choices = doc["choices"]
#         doc["options_with_text"] = (
#             f"A. {choices[0]}\n"
#             f"B. {choices[1]}\n"
#             f"C. {choices[2]}\n"
#             f"D. {choices[3]}"
#         )
#         return doc
#     return dataset.map(_add_field)