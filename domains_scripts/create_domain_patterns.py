"""Combine domain patterns dictionaries """
# %%
import json
from pathlib import Path

import pandas as pd
from srsly import read_json, write_json

# %%
DATADIR = Path("DATAPATH/Desktop/neuroData")
BASE_PATTERNS_DIR = DATADIR / "domain_patterns" / "corrected_pathology_patterns_v2"
BASE_PATTERNS = BASE_PATTERNS_DIR / "patterns.json"
ENTITY_TYPE = "PATHOLOGY"
annotated_orphans = pd.read_csv(
    DATADIR / "domain_patterns/pathology_orphans.csv", index_col=False
)
base_patterns = read_json(BASE_PATTERNS)
# %%
annotated_orphans["assignment_new"].unique()
extra_cerebos = annotated_orphans[
    annotated_orphans["assignment_new"] == "Cerebrovascular"
]["term"].tolist()
extra_surg = annotated_orphans[
    annotated_orphans["assignment_new"] == "Interventional - Surgery"
]["term"].tolist()
extra_inflam = annotated_orphans[
    annotated_orphans["assignment_new"] == "Inflammatory & Autoimmune"
]["term"].tolist()
extra_malform = annotated_orphans[
    annotated_orphans["assignment_new"] == "Congenital & Developmental"
]["term"].tolist()
extra_traum = annotated_orphans[annotated_orphans["assignment_new"] == "Traumatic"][
    "term"
].tolist()
extra_bone = annotated_orphans[
    annotated_orphans["assignment_new"] == "Musculoskeletal"
]["term"].tolist()
extra_neoplas = annotated_orphans[
    annotated_orphans["assignment_new"] == "Neoplastic & Paraneoplastic"
]["term"].tolist()
# %%

base_patterns["Traumatic"] += extra_traum
base_patterns["Cerebrovascular"] += extra_cerebos
base_patterns["Interventional - Surgery"] += extra_surg
base_patterns["Inflammatory & Autoimmune"] += extra_inflam
base_patterns["Congenital & Developmental"] += extra_malform
base_patterns["Musculoskeletal"] += extra_bone
base_patterns["Neoplastic & paraneoplastic"] += extra_neoplas
# %%
print(base_patterns["Inflammatory & Autoimmune"][-10:])
# %%
NEW_PATTERNS = DATADIR / "domain_patterns/pathology_patterns_v4/patterns.json"

write_json(NEW_PATTERNS, base_patterns)
