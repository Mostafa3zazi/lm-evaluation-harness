# mmlu_zeroshot_eval360

A faithful in-harness replica of **LLM360 eval-360's `mmlu_zeroshot`** benchmark
(standard MMLU test, 14,042 questions, zero-shot CoT).

It differs from this repo's other MMLU generative tasks (`mmlu_gen`, `mmlu_us_gen`)
in the two factors that actually move the MMLU number (see Eval360-V2/`MMLU_DIFF.md`):

1. **Prompt format** — eval-360's MMLU-Pro-style reprompt, with the subject intro,
   the reasoning instruction, the question, the `A./B./C./D.` options and the
   `Answer: Let's think step by step.` cue all in a **single user turn** (no system
   message). The rendered prompt is **byte-identical** to eval-360's baked
   `mmlu_zeroshot.jsonl` `completion_input`.
2. **Grader** — eval-360's lenient `mc_answer` parser + `multiple_choice` grader
   (exact letter match), **not** an LLM judge and **not** flexible/strict regex.
   `utils.py` copies eval-360's real `_mc_answer_parser`
   (`Eval360-V2/scheduler/grader/base_parsers.py`) and the grade-stage helpers
   (`scheduler/grader/multiple_choice.py`) verbatim, wired as the `eval360-mc`
   lm-eval filter feeding `exact_match` against a bare-letter `doc_to_target`.

## Layout
- `_mmlu_zeroshot_eval360_template_yaml` — shared template (prompt, filter, metric).
- `mmlu_<subject>.yaml` × 57 — per-subject configs (`cais/mmlu` per-subject configs;
  the `"all"` config is not in the offline HF cache). All carry the tag
  `mmlu_zeroshot_eval360_tasks`. No `description` (subject goes in the user turn).
- `_mmlu_zeroshot_eval360.yaml` — group `mmlu_zeroshot_eval360`, `exact_match`
  weighted by size over the 57 subjects → the single headline number.
- `utils.py` — eval-360 parser + grade-stage helpers + `Eval360MCAnswerFilter`.

## Fidelity check
Running this task's filter (parser + grade stage) over eval-360's own saved
bbq-8b-mid3-final generations (`Eval360-V2/output/bbq-8b-mid3-final/`):
- parse stage identical to eval-360 on **14,042/14,042** samples;
- pick stage identical on **14,042/14,042** samples;
- accuracy **0.804373**, matching eval-360's recorded `0.8043725964962256`.

So with the same generations, scoring is exact. Remaining differences vs eval-360
are environmental (serving backend, sampling temperature, `max_model_len`); see
`MMLU_DIFF.md` §3–§4.
