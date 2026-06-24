# Ara-IFBench

Ara-IFBench is an Arabic deterministic instruction-following benchmark. This
task vendors the verifier logic and lexicons from `MBZUAI-IFM/Ara-IFBench` and
uses the same strict and loose scoring scheme as the upstream evaluator.

By default, `ara_ifbench` loads a six-example smoke manifest bundled with the
task. That default is for wiring checks only, not for benchmark reporting. To
evaluate a real Ara-IFBench manifest, pass a JSONL file through CLI metadata:

```bash
conda run -n lm python -m lm_eval \
  --model hf \
  --model_args pretrained=/path/to/model \
  --tasks ara_ifbench \
  --metadata '{"input_data": "/path/to/ara_ifbench_manifest.jsonl"}'
```

You can also load a Hugging Face dataset repository:

```bash
conda run -n lm python -m lm_eval \
  --model hf \
  --model_args pretrained=/path/to/model \
  --tasks ara_ifbench \
  --metadata '{"repo_id": "org/ara-ifbench-test", "config_name": "default", "hf_split": "test"}'
```

The task supports Ara-IFBench single-turn manifests and compatible IFBench-like
records with `prompt`, `prompt_ar`, `instruction_id_list`, and `kwargs_list`.
Paper-style sequential multi-turn records are intentionally rejected because
they require a model-generated first turn before the scored rewrite turn.

The aggregated lm-eval result JSON reports global prompt-level and
instruction-level strict/loose accuracy, plus instruction-level strict/loose
accuracy for each Ara-IFBench category as
`category_<category_name>_strict_acc` and
`category_<category_name>_loose_acc`. Category metrics are `NaN` when a limited
test run contains no examples from that category.
