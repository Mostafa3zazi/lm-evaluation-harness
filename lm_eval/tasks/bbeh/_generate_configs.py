"""Generate the per-subtask YAML configs and the group config for BBEH.

Run from this directory:  python _generate_configs.py
"""

import yaml


SUBTASKS = [
    "boardgame_qa",
    "boolean_expressions",
    "buggy_tables",
    "causal_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "geometric_shapes",
    "hyperbaton",
    "linguini",
    "movie_recommendation",
    "multistep_arithmetic",
    "nycc",
    "object_counting",
    "object_properties",
    "sarc_triples",
    "shuffled_objects",
    "spatial_reasoning",
    "sportqa",
    "temporal_sequence",
    "time_arithmetic",
    "web_of_lies",
    "word_sorting",
    "zebra_puzzles",
]


def main():
    for subtask in SUBTASKS:
        config = {
            "include": "_bbeh_template_yaml",
            "task": f"bbeh_{subtask}",
            "dataset_name": subtask,
        }
        with open(f"{subtask}.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)

    group_config = {
        "group": "bbeh",
        "task": [f"bbeh_{subtask}" for subtask in SUBTASKS],
        "aggregate_metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "weight_by_size": True,
            }
        ],
        "metadata": {"version": 1.0},
    }
    with open("_bbeh.yaml", "w") as f:
        yaml.dump(group_config, f, sort_keys=False)

    print(f"Wrote {len(SUBTASKS)} subtask configs + _bbeh.yaml")


if __name__ == "__main__":
    main()
