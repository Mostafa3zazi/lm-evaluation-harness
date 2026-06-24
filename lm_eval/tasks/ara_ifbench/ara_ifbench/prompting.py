from __future__ import annotations

from .closed_lists import CONJUNCTION_LIST, COPY_SEPARATOR, HAMZA_FORMS, OUTPUT_TEMPLATE_PREFIXES, PERSON_NAME_LIST, PRONOUN_LIST, SECTION_PREFIX
from .schemas import ExampleRecord
from .specs import TEST_SPEC_INDEX


def render_constraint_text(instruction_id: str, kwargs: dict[str, object]) -> str:
    spec = TEST_SPEC_INDEX[instruction_id]
    template = spec.description_templates_ar[0]
    base_text = template.format(**kwargs)
    return _augment_constraint_text(instruction_id, base_text, kwargs)


def render_example_prompt(example: ExampleRecord) -> str:
    if example.prompt:
        return example.prompt.strip()
    constraints = [
        render_constraint_text(instruction_id, kwargs)
        for instruction_id, kwargs in zip(example.instruction_id_list, example.kwargs_list)
    ]
    if not constraints:
        return example.prompt_ar.strip()
    heading = "يرجى الالتزام بما يلي في الإجابة:"
    lines = [example.prompt_ar.strip(), "", heading]
    lines.extend(f"{_arabic_indic_number(index)}. {constraint}" for index, constraint in enumerate(constraints, start=1))
    return "\n".join(lines)


def _arabic_indic_number(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def _augment_constraint_text(instruction_id: str, base_text: str, kwargs: dict[str, object]) -> str:
    if instruction_id == "count.keywords_multiple":
        details = _format_keyword_counts(kwargs["keyword_counts"])
        return f"{base_text} الكلمات المطلوبة وتكراراتها هي: {details}."

    if instruction_id == "format.title":
        return f"{base_text} ويجب أن يكون العنوان في السطر الأول نفسه."

    if instruction_id == "format.output_template":
        labels = _format_quoted_list(OUTPUT_TEMPLATE_PREFIXES)
        return (
            f"{base_text} ويجب أن تظهر العناوين الحرفية التالية بالترتيب: {labels}. "
            "ويجوز أن يأتي محتوى كل حقل في السطر نفسه أو في السطر الذي يليه مباشرة."
        )

    if instruction_id == "format.newline_words":
        return f"{base_text} أي إن كل سطر غير فارغ يجب أن يحتوي كلمة واحدة فقط."

    if instruction_id == "format.list_custom_separator":
        return (
            f"{base_text} ويجب أن تتكوّن الإجابة كلها من {kwargs['n']} عناصر غير فارغة، "
            f"يفصل بينها الرمز «{kwargs['sep']}» حرفياً من دون فاصل زائد في البداية أو النهاية."
        )

    if instruction_id == "format.line_indent_stairs":
        return (
            f"{base_text} ويجب أن تتكوّن الإجابة من سطرين على الأقل، "
            "وأن تُقاس الإزاحة بالمسافات البادئة فقط مع زيادة صارمة من سطر إلى الذي يليه."
        )

    if instruction_id == "format.number_bullets":
        return f"{base_text} ويُقبل بدء كل نقطة بأحد الأنماط الشائعة مثل - أو * أو 1. أو ١."

    if instruction_id == "format.multiple_sections":
        headings = _format_section_headings(int(kwargs["n"]))
        return f"{base_text} واجعل عناوين الأقسام على هذا النحو حرفياً: {headings}."

    if instruction_id == "format.nested_quotes":
        return f"{base_text} ويكفي نمط مثل: {_nested_quote_example(int(kwargs['depth']))}."

    closed_list = _visible_closed_list(instruction_id)
    if closed_list is not None:
        return f"{base_text} والقائمة المغلقة هي: {closed_list}."

    if instruction_id.startswith("copy."):
        source_text = str(kwargs["source_text"])
        output_only_note = "ويجب أن تتكوّن الإجابة كلها من الناتج المطلوب فقط، من دون أي شرح أو نص إضافي."
        if instruction_id == "copy.repeat_span_char_idx":
            return (
                f"{base_text} النص المعطى هو: «{source_text}». "
                f"ويبدأ العد من الصفر، ويُحتسب المؤشران ضمن المدى. {output_only_note}"
            )
        if instruction_id == "copy.copy_request_exact_n_times":
            return (
                f"{base_text} النص المعطى هو: «{source_text}». "
                f"والفاصل الثابت بين كل تكرارين هو السلسلة الحرفية التالية تماماً: «{COPY_SEPARATOR}». "
                f"{output_only_note}"
            )
        return f"{base_text} النص المعطى هو: «{source_text}». {output_only_note}"

    if instruction_id == "transform.square_brackets_every_token":
        return (
            f"{base_text} أي يجب أن تكون كل كلمة أو رمز مستقل في الإجابة محاطاً بالقوسين [ و ]، "
            "من دون أي كلمات خارج الأقواس."
        )

    if instruction_id == "transform.wrap_every_bigram":
        return (
            f"{base_text} أي يجب أن تتكوّن الإجابة كلها من مقاطع متتالية، "
            "يضم كل مقطع كلمة واحدة أو كلمتين فقط، وكل مقطع محاط بالعلامتين « و »، "
            "من دون أي نص خارج هذه المقاطع."
        )

    if instruction_id == "ratio.same_char_count_three_sentences":
        return (
            f"{base_text} ويُحسب عدد المحارف بعد تجاهل المسافات، "
            "ومع تجاهل علامة نهاية الجملة الأخيرة مثل . أو ! أو ؟."
        )

    if instruction_id in {
        "arabic.orth.hamza_char_count",
        "arabic.orth.hamza_forms_cover_all",
        "arabic.orth.hamza_forms_exact_once_each",
    }:
        return f"{base_text} وصور الهمزة المقصودة هي: {_format_list(HAMZA_FORMS)}."

    return base_text


def _visible_closed_list(instruction_id: str) -> str | None:
    if instruction_id == "count.pronouns_min_closed_list":
        return _format_list(PRONOUN_LIST)
    if instruction_id == "count.conjunctions_distinct_closed_list":
        return _format_list(CONJUNCTION_LIST)
    if instruction_id == "count.person_names_from_closed_list":
        return _format_list(PERSON_NAME_LIST)
    return None


def _format_list(values: tuple[str, ...]) -> str:
    return "، ".join(values)


def _format_keyword_counts(payload: object) -> str:
    keyword_counts = payload if isinstance(payload, dict) else {}
    rendered_items = [f"{keyword}: {count}" for keyword, count in keyword_counts.items()]
    return "، ".join(rendered_items)


def _format_section_headings(count: int) -> str:
    return _format_quoted_list(tuple(f"{SECTION_PREFIX} {index}:" for index in range(1, count + 1)))


def _nested_quote_example(depth: int) -> str:
    if depth <= 2:
        return "«قال المتحدث: \"النص الداخلي\"»"
    return "«قال المتحدث: \"قال الشاهد: 'النص الداخلي'\"»"


def _format_quoted_list(values: tuple[str, ...]) -> str:
    return " ثم ".join(f"«{value}»" for value in values)
