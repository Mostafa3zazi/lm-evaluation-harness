"""
Utility functions for Dialectal Arabic MMLU (DiacArMMLU) task.
Provides process_docs functions to filter examples by dialect and domain.
"""


def _make_process_docs(dialect: str, domain: str):
    """Factory function to create a process_docs filter for a specific dialect and domain."""
    def process_docs_fn(dataset):
        return dataset.filter(lambda doc: doc["dialect"] == dialect and doc["domain"] == domain)
    return process_docs_fn


# Pre-generate process_docs functions for all dialect+domain combinations
DIALECTS = ["EGY", "ENG", "KSA", "MAG", "MSA", "SYR", "UAE"]

DOMAINS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_computer_science",
    "college_medicine",
    "conceptual_physics",
    "elementary_mathematics",
    "global_facts",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_psychology",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "international_law",
    "management",
    "marketing",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_law",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


# Dynamically create process_docs functions for each dialect+domain combo
# These will be called as: process_docs_EGY_astronomy, process_docs_EGY_anatomy, etc.
for _dialect in DIALECTS:
    for _domain in DOMAINS:
        # Create function name like "process_docs_EGY_astronomy"
        func_name = f"process_docs_{_dialect}_{_domain}"
        # Create the filter function and add to module globals
        globals()[func_name] = _make_process_docs(_dialect, _domain)


# Also create per-dialect process_docs (for dialect-level aggregation)
def _make_dialect_process_docs(dialect: str):
    """Factory function to create a process_docs for a specific dialect (all domains)."""
    def process_docs_fn(dataset):
        return dataset.filter(lambda doc: doc["dialect"] == dialect)
    return process_docs_fn


for _dialect in DIALECTS:
    func_name = f"process_docs_{_dialect}"
    globals()[func_name] = _make_dialect_process_docs(_dialect)
