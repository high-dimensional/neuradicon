from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import srsly
from neuroNLP.custom_pipes import *
from neuroNLP.custom_pipes import DomainDetector
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from spacy.tokens import Doc, DocBin
from srsly import read_json
from tqdm import tqdm

name = "en_full_neuro_model"
version = "1.8"
date = "270922"
DATADIR = Path("DATAPATH/Desktop/neuroData")
# path_data = DATADIR / "domains_data/joined_domain_labels.jsonl"
path_data = DATADIR / "domains_data" / "evaluation-data" / "domain-review.jsonl"
# path_patterns = DATADIR / "domain_patterns/pathology_patterns_v4"
path_patterns = DATADIR / "domain_patterns" / "terminology_data"
nlp_dir = "DATAPATH/Desktop/neuroNLP/packages/en_neuro_base-1.0/en_neuro_base/en_neuro_base-1.0"

nlp = spacy.load(nlp_dir)


pathdetector = DomainDetector(
    nlp, class_types=["PATHOLOGY", "DESCRIPTOR"], method="matcher", use_negated=False
)
pathdetector.from_disk(path_patterns)
# %%
name_map_full = {
    "pathology-cerebrovascular": "Cerebrovascular",
    "pathology-mass-effect": "mass",
    "pathology-csf-disorders": "CSF disorders",
    "pathology-morphology": "morphology",
    "pathology-signal-change": "signal",
    "pathology-flow": "flow/csf",
    "pathology-enhancement": "enhancement",
    "pathology-interval-change": "intervalchange",
    "pathology-neoplastic-paraneoplastic": "Neoplastic & paraneoplastic",
    "pathology-infectious": "Infectious",
    "pathology-inflammatory-autoimmune": "Inflammatory & Autoimmune",
    "pathology-cysts": "Cystic",
    "pathology-metabolic-nutritional-toxic": "Metabolic, Nutritional, & Toxic",
    "pathology-neurodegenerative-dementia": "Neurodegenerative & Dementia",
    "pathology-traumatic": "Traumatic",
    "pathology-musculoskeletal": "Musculoskeletal",
    "pathology-treatment": "Interventional - Surgery",
    "pathology-congenital-developmental": "Congenital & Developmental",
    "pathology-opthalmological": "Ophthalmological",
    "pathology-diffusion": "diffusion",
    "pathology-necrosis": "Necrosis",
    "pathology-headache": "Headache",
    "pathology-endocrine": "Endocrine",
    "pathology-haematological": "Haematological",
}
reverse_map = {v: k for k, v in name_map_full.items()}


gold_path_labels = [i for i in srsly.read_jsonl(path_data) if i["answer"] == "accept"]
pred_path_docs = [pathdetector(nlp(d["text"])) for d in tqdm(gold_path_labels)]
gold_path_domains = [sorted(list(set(d["accept"]))) for d in gold_path_labels]
pred_path_domains = [
    sorted(list(set([i for i in d._.domains]))) for d in pred_path_docs
]


path_bina = MultiLabelBinarizer()

path_true = path_bina.fit_transform(gold_path_domains)
path_match_pred = path_bina.transform(pred_path_domains)
path_neuro_pred = path_bina.transform(full_neuro_pred_path_domains)

mtlconf = multilabel_confusion_matrix(path_true, path_neuro_pred)


# %%
def output_to_file(file):
    file.write("Matcher-based performance")
    output = classification_report(
        path_true, path_match_pred, target_names=path_bina.classes_, zero_division=0
    )
    file.write(output)
    file.write("NER-based performance")
    output = classification_report(
        path_true, path_neuro_pred, target_names=path_bina.classes_, zero_division=0
    )
    file.write(output)

    for i, name in enumerate(path_bina.classes_):
        file.write(name + "\n")
        file.write(str(mtlconf[i]) + "\n")

    for i, v in enumerate(path_bina.classes_):
        idx_fp = np.logical_and(path_true[:, i] == 0, path_neuro_pred[:, i] == 1)
        idx_fn = np.logical_and(path_true[:, i] == 1, path_neuro_pred[:, i] == 0)

        fp_docs = [d for d, i in zip(gold_path_labels, idx_fp) if i]
        file.write(f"\nFalse positives for {v}\n")
        for t in fp_docs:
            file.write(t["text"])
            file.write("\n")
            doc = full_nlp(t["text"])
            file.write(
                "ents: "
                + str(
                    [
                        (e.text, e.label_)
                        for e in doc.ents
                        if e.label_ in classes_to_keep
                    ]
                )
            )
            file.write("\n")
            file.write("\n")

        fn_docs = [d for d, i in zip(gold_path_labels, idx_fn) if i]
        file.write(f"\nFalse negatives for {v}\n")
        for t in fn_docs:
            file.write(t["text"])
            file.write("\n")
            doc = full_nlp(t["text"])
            file.write(
                "ents: "
                + str(
                    [
                        (e.text, e.label_)
                        for e in doc.ents
                        if e.label_ in classes_to_keep
                    ]
                )
            )
            file.write("\n")
            file.write("\n")


with open(f"domain-perf_v{version}_{date}.txt", "w") as f:
    output_to_file(f)
