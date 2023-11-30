from pathlib import Path

import spacy
import srsly

from neuradicon.custom_pipes import *

# %%
model_name = "en_full_neuro_model"
model_version = "1.4"
ROOT = Path("DATAPATH/Desktop")
data_path = (
    ROOT
    / "neuroData/domains_data/julius_domain_labels/julius_errors_review_150822.jsonl"
)
model_path = (
    ROOT
    / "neuradicon"
    / "packages"
    / f"{model_name}-{model_version}"
    / model_name
    / f"{model_name}-{model_version}"
)
data = list(srsly.read_jsonl(data_path))
model = spacy.load(model_path)

# %%
pathology_classes = [
    "pathology-endocrine",
    "pathology-congenital-developmental",
    "pathology-cerebrovascular",
    "pathology-csf-disorders",
    "pathology-infectious",
    "pathology-inflammatory-autoimmune",
    "pathology-metabolic-nutritional-toxic",
    "pathology-musculoskeletal",
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
]
# %%
comparisons = [
    {
        "text": i["text"],
        "labelled_domain": [k for k in i["accept"] if k in pathology_classes],
        "inferred_domains": [
            l for l in extract_domains(model(i["text"])) if l in pathology_classes
        ],
    }
    for i in data
]
# %%
inconsistent_reports = [
    i
    for i in comparisons
    if set(i["labelled_domain"]).difference(set(i["inferred_domains"]))
]

# %%
with open("inconsistencies.txt", "w") as file:
    for r in inconsistent_reports:
        file.write("\n\ntext\n")
        file.write(r["text"])
        file.write("\nlabels: " + str(r["labelled_domain"]))
        file.write("\npredictions: " + str(r["inferred_domains"]))
