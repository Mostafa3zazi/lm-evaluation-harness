import os
import re

from lm_eval.tasks.arab_culture.prompts import (
    BASE_PROMPT,
    BASE_PROMPT_AR,
    JAIS_CHAT_AR,
    JAIS_CHAT_EN,
    REGION_COUNTRY_PROMPT,
    REGION_COUNTRY_PROMPT_AR,
    REGION_PROMPT,
    REGION_PROMPT_AR,
)

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

### get the conutry variable from environment

### Set this to one to add the country and region information to the prompt
COUNTRY = True if os.getenv("COUNTRY", True) == "True" else False
### Set this to one to add the region information to the prompt
REGION = True if os.getenv("REGION", True) == "True" else False
### Set this to change between Arabic and English for the answer keys and the choices keys
ARABIC = True if os.getenv("ARABIC", True) == "True" else False
### Get the model name
MODEL_NAME = os.getenv("MODEL_NAME")
## Uncomment this to check if the environment variables are set correctly
# print(f'Task settings: COUNTRY: {COUNTRY}, REGION: {REGION}, ARABIC: {ARABIC}', MODEL_NAME: {MODEL_NAME})

en_ar_countries_regions = {
    "Egypt": "مصر",
    "Morocco": "المغرب",
    "Algeria": "الجزائر",
    "Libya": "ليبيا",
    "Sudan": "السودان",
    "Tunisia": "تونس",
    "Jordan": "الأردن",
    "Lebanon": "لبنان",
    "Syria": "سوريا",
    "Palestine": "فلسطين",
    "Yemen": "اليمن",
    "UAE": "الإمارات",
    "KSA": "السعودية",
    "Gulf": "الخليج",
    "Levant": "الشام",
    "North Africa": "شمال أفريقيا",
    "Nile Valley": "وادي النيل",
}


def doc_to_text(doc):
    country = "" if not doc["country"] else doc["country"]
    region = "" if not doc["region"] else doc["region"]
    first_statement = doc["first_statement"].strip()

    ## We don't have a setting for only information about the country without the region
    if COUNTRY:
        assert REGION, (
            "If you want to add the country information, you must also add the region information"
        )

    ## convert contry and region name to arabic if the language is arabic
    if ARABIC:
        country = en_ar_countries_regions[country]
        region = en_ar_countries_regions[region]

    choices = doc["options"]
    choices_str = ""
    for i in range(3):
        key = choices["arabic_keys"][i] if ARABIC else choices["english_keys"][i]
        choice_str = key + ". " + choices["text"][i].strip() + "\n"
        choices_str += choice_str

    if COUNTRY and REGION:
        cur_prompt = REGION_COUNTRY_PROMPT_AR if ARABIC else REGION_COUNTRY_PROMPT
        doc_text = cur_prompt.format(
            country=country,
            region=region,
            first_statement=first_statement,
            choices=choices_str,
        )
    elif REGION:
        cur_prompt = REGION_PROMPT_AR if ARABIC else REGION_PROMPT
        doc_text = cur_prompt.format(
            region=region, first_statement=first_statement, choices=choices_str
        )
    else:
        cur_prompt = BASE_PROMPT_AR if ARABIC else BASE_PROMPT
        doc_text = cur_prompt.format(
            first_statement=first_statement, choices=choices_str
        )

    ### apply jais chat template
    if MODEL_NAME and "jais" in MODEL_NAME and "chat" in MODEL_NAME:
        if ARABIC:
            doc_text = JAIS_CHAT_AR.format(question=doc_text)
        else:
            doc_text = JAIS_CHAT_EN.format(question=doc_text)

    return (build_prompt("en",3)+ "\n" + doc_text)


def doc_to_choice(doc):
    choices = doc["options"]
    choices_str = ""
    for i in range(3):
        key = choices["arabic_keys"][i] if ARABIC else choices["english_keys"][i]
        choice_str = key + ". " + choices["text"][i].strip() + "\n"
        choices_str += choice_str
    return choices_str


def doc_to_target(doc):
    ans = (
        doc["answer_key"]["arabic_answer_key"]
        if ARABIC
        else doc["answer_key"]["english_answer_key"]
    )
    ans = ans.strip()
    return ans

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
        doc["options_with_text"] = doc_to_choice(doc)
        return doc
    return dataset.map(_add_field)