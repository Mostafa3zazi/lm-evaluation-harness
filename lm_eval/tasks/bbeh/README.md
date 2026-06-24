# BIG-Bench Extra Hard (BBEH)

### Paper

Title: `BIG-Bench Extra Hard`

Abstract: https://arxiv.org/abs/2502.19187

BBEH replaces each of the 23 tasks in BIG-Bench Hard (BBH) with a novel
counterpart that probes a similar reasoning capability but is substantially more
difficult. The benchmark targets skills such as many-hop reasoning, learning on
the fly, finding errors in reasoning traces, long-context needle-finding, going
against strong priors, long-range dependencies, distractor handling, and pattern
induction.

Homepage: https://github.com/google-deepmind/bbeh

### Citation

```bibtex
@article{kazemi2025bbeh,
  title={BIG-Bench Extra Hard},
  author={Kazemi, Mehran and others},
  journal={arXiv preprint arXiv:2502.19187},
  year={2025}
}
```

### Groups and Tasks

#### Groups

* `bbeh`: Runs all 23 BBEH subtasks and reports a size-weighted mean exact-match.

#### Tasks

One task per subtask, named `bbeh_<subtask>` (e.g. `bbeh_web_of_lies`,
`bbeh_zebra_puzzles`).

### Scoring

Tasks are zero-shot chain-of-thought and `generate_until`. The model is asked to
finish with `The answer is: <answer>`. Scoring is ported verbatim from the
official BBEH `evaluate.py` (`extract_answer` + `fuzzy_match`), so reported
numbers reproduce the paper rather than a plain exact-match.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    Scoring is ported from the reference implementation.
