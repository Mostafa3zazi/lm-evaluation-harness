# noqa
"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


DIALECTS = ['EGY', 'ENG', 'KSA', 'MAG', 'MSA', 'SYR', 'UAE']

SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_computer_science": "stem",
    "college_medicine": "other",
    "conceptual_physics": "stem",
    "elementary_mathematics": "stem",
    "global_facts": "other",
    "high_school_chemistry": "stem",
    "high_school_geography": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "international_law": "humanities",
    "management": "other",
    "marketing": "other",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu_ar_dial")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_DIALECTS = []
    for dialect in tqdm(DIALECTS):
        if dialect not in ALL_DIALECTS:
            ALL_DIALECTS.append(dialect)

        for subject, category in tqdm(SUBJECTS.items()):
            if args.cot_prompt_path is not None:
                description = cot_file[subject]
            else:
                description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

            # Build task name
            task_name = f"mmlu_ar_dial_{dialect}_{subject}_{args.task_prefix}" if args.task_prefix != "" else f"mmlu_ar_dial_{dialect}_{subject}"
            tag_name = f"mmlu_ar_dial_{dialect}_{args.task_prefix}" if args.task_prefix != "" else f"mmlu_ar_dial_{dialect}_tasks"
            
            # Write YAML manually to include !function tag for process_docs
            file_save_path = args.save_prefix_path + f"_{dialect}_{subject}.yaml"
            eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
            
            yaml_content = f'''"include": "{base_yaml_name}"
"tag": "{tag_name}"
"task": "{task_name}"
"task_alias": "{dialect}: {subject.replace('_', ' ')}"
"description": "{description}"
process_docs: !function utils.process_docs_{dialect}_{subject}
'''
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml_file.write(yaml_content)

    if args.task_prefix != "":
        mmlu_subcategories = [
            f"mmlu_ar_dial_{dialect}_{args.task_prefix}" for dialect in ALL_DIALECTS
        ]
    else:
        mmlu_subcategories = [f"mmlu_ar_dial_{dialect}" for dialect in ALL_DIALECTS]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_ar_dial_{args.task_prefix}"
                if args.task_prefix != ""
                else "mmlu_ar_dial",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
