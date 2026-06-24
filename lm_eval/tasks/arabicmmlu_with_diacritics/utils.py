PROMPT = "This is a {}. Select the correct answer!\n\nQuestion: {}\n{}\n\nAnswer:"

level_en = {
    "Primary": "primary school",
    "Middle": "middle school",
    "High": "high school",
    "Univ": "university",
    "Prof": "professional",
}

alpa = ["A.", "B.", "C.", "D.", "E."]


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

    return doc_text


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