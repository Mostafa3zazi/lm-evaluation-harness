"""Generate the per-task leaf YAMLs (and groups) for the generative LongBench v2
suite over the filtered translated subdomains.

Each data file under ``DATA_ROOT`` is a single pre-filtered subtask that holds
*both* the Arabic fields (``context``/``question``/``choice_*``) and their
English counterparts (``context_en``/``question_en``/``choice_*_en``), sharing
one gold ``answer`` letter. We therefore emit two tasks per file, one per
language, both pointing at the same JSON; only the ``include``d common YAML
differs (Arabic vs English prompt). This lets the Arabic and English variants be
run together and compared. Run from this directory:

    python _generate_config.py                 # writes *.yaml here
    python _generate_config.py --overwrite     # also overwrite existing files
"""

import argparse
import os


DATA_ROOT = "/home/nurkhan.laiyk/longbenchv2_arabic/translation_out_v2"

# language -> (included common yaml, task name infix, grouping tag, group name)
LANGS = {
    "ar": (
        "_longbench2_ar_gen_common_yaml",
        "ar",
        "longbench2_arabic_tasks_gen",
        "longbench2_gen_arabic",
    ),
    "en": (
        "_longbench2_en_gen_common_yaml",
        "en",
        "longbench2_english_tasks_gen",
        "longbench2_gen_english",
    ),
}

# (subtask name, data file basename)
SUBTASKS = [
    ("academic", "longbench_v2_ar_academic.json"),
    ("academic_mqa", "longbench_v2_ar_academic_mqa.json"),
    ("multinews", "longbench_v2_ar_multinews.json"),
    ("detective", "longbench_v2_ar_detective.json"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_prefix_path", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write(path, content, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f"skip (exists): {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote: {path}")


def main():
    args = parse_args()
    for lang, (common, infix, tag, group_name) in LANGS.items():
        task_names = []
        for subtask, data_file in SUBTASKS:
            task_name = f"longbench2_{infix}_{subtask}_gen"
            task_names.append(task_name)
            lines = [
                f"include: {common}",
                "tag:",
                f"  - {tag}",
                f"task: {task_name}",
                "dataset_kwargs:",
                "  data_files:",
                f"    train: {os.path.join(DATA_ROOT, data_file)}",
            ]
            _write(
                os.path.join(args.save_prefix_path, f"longbench2_{infix}_{subtask}.yaml"),
                "\n".join(lines) + "\n",
                args.overwrite,
            )

        group_lines = [f"group: {group_name}", "task:"]
        group_lines += [f"  - {t}" for t in task_names]
        group_lines += [
            "aggregate_metric_list:",
            "  - metric: exact_match",
            "    weight_by_size: true",
            "metadata:",
            "  version: 0.0",
        ]
        _write(
            os.path.join(args.save_prefix_path, f"_{group_name}.yaml"),
            "\n".join(group_lines) + "\n",
            args.overwrite,
        )


if __name__ == "__main__":
    main()
