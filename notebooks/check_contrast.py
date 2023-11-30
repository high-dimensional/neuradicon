"""Check contrast vs BRC3 files

agreement is between the contrast label and the label has_a_T1CE,
which you can generate from just listing the jsons in
/BRC3/data_for_multimodal_model_uncropped/xnat_registered_affine_2mm and checking for a T1CE field
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATADIR = Path("DATAPATH/Desktop/neuroData/processed_reports")
BRC = Path(
    "DATAPATH/brc_server/data_for_multimodal_model_uncropped/xnat_registered_affine_2mm"
)
# %%
session_df = pd.read_csv(DATADIR / "processed_reports.csv", low_memory=False)
image_matches = pd.read_csv(DATADIR / "image_data_matches.csv")
# %%


def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def get_contrast_from_brc(df):
    output = []
    for i in tqdm(df["uid"]):
        path = BRC / i.replace(".", "")
        if path.exists():
            for j in path.iterdir():
                if j.suffix == ".json":
                    json_data = read_json(j)
                    output.append("T1CE" in json_data.keys())
                    break
        else:
            output.append(np.nan)
    return output


# %%
image_matches = image_matches.assign(json_contrast_label=get_contrast_from_brc)
# %%

# %%
to_keep = ["sessionindex", "Narrative", "uses_contrast", "Modality", "Procedure"]
expanded_df = image_matches.join(
    session_df[to_keep].set_index("sessionindex"), on="session_idx"
)
# %%
from sklearn.metrics import confusion_matrix


def uses_contrast_(df):
    contrast_condition = pd.concat(
        [
            df["Procedure"].str.contains("+c", case=False, regex=False),
            df["Procedure"].str.contains("contrast", case=False, regex=False),
            df["Procedure"].str.contains("Post Gad", case=False, regex=False),
            df["Narrative"].str.contains("Post Gad", case=False, regex=False),
            df["Narrative"].str.contains("MR+c", case=False, regex=False),
            df["Narrative"].str.contains("+ Gd", case=False, regex=False),
            df["Narrative"].str.contains("post gadolinium", case=False, regex=False),
        ],
        axis=1,
    ).any(axis=1)
    return contrast_condition


has_label = expanded_df.dropna(subset=["json_contrast_label"])
contrast = uses_contrast_(has_label)
has_label = has_label.assign(new_contrast=uses_contrast_)
# %%
has_label["new_contrast"] = contrast
# %%
true, pred = has_label["json_contrast_label"].astype(int), has_label[
    "new_contrast"
].astype(int)
confmat = confusion_matrix(true, pred)
tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
print((tn, fp, fn, tp))
# %%
false_negatives = has_label[
    has_label["json_contrast_label"] & (~has_label["new_contrast"])
]
false_positives = has_label[
    (~has_label["json_contrast_label"]) & (has_label["new_contrast"])
]
# %%
false_negatives[["Modality", "Procedure", "Narrative"]].sample(5).values
# %%
false_positives[["Modality", "Procedure", "Narrative"]].sample(5).values
