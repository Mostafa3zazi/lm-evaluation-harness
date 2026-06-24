"""Generate the per-task leaf YAMLs for the generative LongBench v2 suite.

Each leaf task ``include``s ``_longbench_common_yaml`` (which loads the local
``longbench_v2.json``, defines the ``generate_until`` prompt, the
``extract_answer`` filter and the ``exact_match_normalized_label`` metric) and
only sets the task name, the grouping tags and a ``process_docs`` callable that
carves out this task's slice of the combined file by (domain, sub_domain). Run
from this directory:

    python _generate_config.py                 # writes *.yaml here
    python _generate_config.py --overwrite     # also overwrite existing files
"""

import argparse
import os


COMMON_YAML = "_longbench_common_yaml"

# (output filename, short task suffix, category)
# ``category`` maps to the ``longbench2_<category>_tasks_gen`` grouping tag; use
# None for tasks (code) that only carry the top-level tag. The suffix also keys
# into ``utils.TASK_DOMAINS`` and selects the ``utils.filter_<suffix>`` callable.
TASKS = [
    # Single-document QA
    ("govt_single_doc", "govt_single", "single"),
    ("legal_single", "legal_single", "single"),
    ("lit_single_doc", "lit_single", "single"),
    ("fin_single_doc", "fin_single", "single"),
    ("event_order", "event_order", "single"),
    ("academic_single", "academic_single", "single"),
    ("detective", "detective", "single"),
    # Multi-document QA
    ("govt_multi_doc", "govt_multi", "multi"),
    ("academic_multi_doc", "academic_multi", "multi"),
    ("fin_multi_doc", "fin_multi", "multi"),
    ("news_multi", "news_multi", "multi"),
    ("legal_multi", "legal_multi", "multi"),
    # Long in-context learning
    ("user_guide", "user_guide", "incontext"),
    ("translate", "translate", "incontext"),
    ("many_shot", "many_shot", "incontext"),
    # Long-dialogue history understanding
    ("agent_history", "agent_history", "history"),
    ("dialogue_history", "dialogue_history", "history"),
    # Long structured data understanding
    ("graph", "graph", "structured"),
    ("table", "table", "structured"),
    # Code repository understanding (no category sub-tag)
    ("longbench2_code", "code", None),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_prefix_path",
        default="",
        help="Directory prefix to write the generated YAMLs to.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite YAMLs that already exist (default: skip them).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for filename, suffix, category in TASKS:
        tags = ["longbench2_tasks_gen"]
        if category is not None:
            tags.append(f"longbench2_{category}_tasks_gen")

        # Emitted as raw text (not yaml.dump) so the ``!function`` tag stays
        # unquoted and is resolved by the harness rather than read as a string.
        lines = [f"include: {COMMON_YAML}", "tag:"]
        lines += [f"  - {t}" for t in tags]
        lines.append(f"task: longbench2_{suffix}_gen")
        lines.append(f"process_docs: !function utils.filter_{suffix}")
        content = "\n".join(lines) + "\n"

        out_path = os.path.join(args.save_prefix_path, f"{filename}.yaml")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip (exists): {out_path}")
            continue
        with open(out_path, "w") as f:
            f.write(content)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
