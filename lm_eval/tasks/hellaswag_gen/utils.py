import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        base_query = preprocess(doc["activity_label"] + ": " + ctx)

        choices = [preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])

        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options_text = "\n".join(
            f"{letters[i]}. {choices[i]}" for i in range(len(choices))
        )

        query = (
            f"{base_query}...\n\n"
            "Choose the most accurate continuation among the options. Respond only with the option letter:\n"
            f"{options_text}\n\n"
            "Answer:"
        )

        return {
            "query": query,
            "choices": choices,
            "gold": gold,
            "label": letters[gold],
        }

    return dataset.map(_process_doc)