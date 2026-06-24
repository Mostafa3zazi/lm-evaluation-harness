from __future__ import annotations

import string
import unicodedata

ARABIC_INDIC_DIGITS = set("٠١٢٣٤٥٦٧٨٩")
ASCII_DIGITS = set(string.digits)
HAMZA_CHARS = set("ءأإؤئ")
ARABIC_PUNCTUATION = {"،", "؛", "؟"}
ASCII_PUNCT_EQUIVALENTS = {",", ";", "?"}
ARABIC_DIACRITICS = set("ًٌٍَُِّْٰ")
TANWEEN_CHARS = set("ًٌٍ")
PERSIAN_VARIANT_CHARS = {"ک", "ی", "ے", "ہ"}
ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u200e", "\u200f"}
EDGE_PUNCTUATION = set(string.punctuation) | ARABIC_PUNCTUATION | {"«", "»", "“", "”", "‘", "’", "…", "ـ"}

PRONOUN_LIST = (
    "أنا",
    "نحن",
    "أنت",
    "أنتِ",
    "أنتما",
    "أنتم",
    "أنتن",
    "هو",
    "هي",
    "هما",
    "هم",
    "هن",
)
PRONOUNS = set(PRONOUN_LIST)

CONJUNCTION_LIST = ("و", "أو", "ثم", "لكن", "بل", "ف", "أم")
CONJUNCTIONS = set(CONJUNCTION_LIST)

PERSON_NAME_LIST = (
    "أحمد",
    "محمد",
    "علي",
    "خالد",
    "سارة",
    "مريم",
    "نور",
    "ليلى",
    "يوسف",
    "عمر",
    "هند",
    "ريم",
    "فاطمة",
    "كريم",
    "سلمان",
)
PERSON_NAMES = set(PERSON_NAME_LIST)

HAMZA_FORMS = ("ء", "أ", "إ", "ؤ", "ئ")

SECTION_PREFIX = "القسم"
OUTPUT_TEMPLATE_PREFIXES = ("الإجابة:", "الخلاصة:", "النظرة المستقبلية:")
COPY_SEPARATOR = " ****** "

ARABIC_BLOCKS = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

ARABIC_PRESENTATION_BLOCKS = (
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def in_arabic_block(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ARABIC_BLOCKS)


def is_arabic_letter(char: str) -> bool:
    return in_arabic_block(char) and unicodedata.category(char).startswith("L")


def is_latin_letter(char: str) -> bool:
    if not unicodedata.category(char).startswith("L"):
        return False
    try:
        return "LATIN" in unicodedata.name(char)
    except ValueError:
        return False


def in_arabic_presentation_block(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ARABIC_PRESENTATION_BLOCKS)


def strip_edge_punctuation(token: str) -> str:
    start = 0
    end = len(token)
    while start < end and token[start] in EDGE_PUNCTUATION:
        start += 1
    while end > start and token[end - 1] in EDGE_PUNCTUATION:
        end -= 1
    return token[start:end]


def first_significant_char(token: str) -> str:
    stripped = strip_edge_punctuation(token)
    return stripped[0] if stripped else ""


def normalized_token_for_match(token: str) -> str:
    return strip_edge_punctuation(token).casefold()
