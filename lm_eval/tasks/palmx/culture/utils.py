"""Scoring helpers for the PalmX 2025 subtask 1 (culture) MCQ task.

Each example has the columns: id, question, A, B, C, D, answer
where `answer` is one of the letters "A", "B", "C", "D".
The task is scored as a log-likelihood multiple-choice task over the four
answer letters.
"""

CHOICES = ["A", "B", "C", "D"]


def doc_to_choice(doc):
    return CHOICES


def doc_to_target(doc):
    return CHOICES.index(doc["answer"].strip())
