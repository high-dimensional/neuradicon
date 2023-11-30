"""Stroke vs SVD separation script

This script is to extract the subset of reports of patients with DWI imaging, but specifically
those who have a stroke rather than mild ischaemic changes such as small vessel disease
"""
from pathlib import Path

import pandas as pd

ROOT = Path("DATAPATH/Desktop")
DATA = ROOT / "neuroNLP/runs/path_model_030123"
DWIMRNS = ROOT / "db_queries/stroke_reports/patientIDs_list.csv"
OUTDIR = ROOT / "db_queries/stroke_reports"

# %%
dwimrns = pd.read_csv(DWIMRNS, names=["MRN"]).astype(str)
df = pd.read_csv(DATA / "report_data.csv", low_memory=False)
# %%
dwi_patients = df[df["MRN"].isin(dwimrns["MRN"])]
print(len(dwi_patients))
# %%
ischaemic_dwis = dwi_patients[
    (dwi_patients["asserted-pathology-ischaemic"] > 0)
    | (dwi_patients["asserted-pathology-cerebrovascular"] > 0)
]
print(len(ischaemic_dwis))
# %%
ischaemic_dwis["mentions_stroke"] = ischaemic_dwis["report_body"].str.contains(
    "cva|stroke", case=False, regex=True
)
ischaemic_dwis["mentions_svd"] = ischaemic_dwis["report_body"].str.contains(
    "small vessel disease|svd", case=False, regex=True
)

# %%
import matplotlib.pyplot as plt

SIZE = 10
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df["_X"], df["_Y"], c="tab:gray", alpha=0.5, label="all reports", s=SIZE)
ax.scatter(
    ischaemic_dwis["_X"], ischaemic_dwis["_Y"], c="r", label="other dwi reports", s=SIZE
)
stroke = ischaemic_dwis[ischaemic_dwis["has_stroke"]]
svds = ischaemic_dwis[ischaemic_dwis["mentions_svd"]]
ax.scatter(svds["_X"], svds["_Y"], c="g", label="dwi mentioning svd", s=SIZE)
ax.scatter(stroke["_X"], stroke["_Y"], c="b", label="dwi mentioning stroke", s=SIZE)
ax.legend()
fig.tight_layout()
fig.savefig("DATAPATH/Dropbox/stroke_vs_svd.png")
# %%
from sklearn.manifold import TSNE

tsne = TSNE()
tsne_xy = tsne.fit_transform(ischaemic_dwis[["_X", "_Y"]].to_numpy())
ischaemic_dwis["tsne_X"] = tsne_xy[:, 0]
ischaemic_dwis["tsne_Y"] = tsne_xy[:, 1]
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(
    ischaemic_dwis["tsne_X"],
    ischaemic_dwis["tsne_Y"],
    c="r",
    label="other dwi reports",
    s=SIZE,
)
stroke = ischaemic_dwis[ischaemic_dwis["mentions_stroke"]]
svds = ischaemic_dwis[ischaemic_dwis["mentions_svd"]]
ax.scatter(svds["tsne_X"], svds["tsne_Y"], c="g", label="dwi mentioning svd", s=SIZE)
ax.scatter(
    stroke["tsne_X"], stroke["tsne_Y"], c="b", label="dwi mentioning stroke", s=SIZE
)
ax.legend()
fig.tight_layout()
fig.savefig("DATAPATH/Dropbox/stroke_vs_svd_tsne.png")
# %%
import spacy
from neuroNLP.custom_pipes import *
from spacy.tokens import DocBin

nlp = spacy.load(
    ROOT
    / "neuroNLP/packages/en_full_neuro_model-1.8/en_full_neuro_model/en_full_neuro_model-1.8"
)
bin = DocBin().from_disk(DATA / "data.spacy")
docss = list(bin.get_docs(nlp.vocab))

# %%
len(docss)


# %%
samp = ischaemic_dwis.sample()
doc = docss[samp.index[0]]
print(samp["report_body"].values, "\n", doc, list(doc.ents))
# %%
doc.ents[3].label_
# %%
from tqdm import tqdm

asserted_stroke = []


def has_stroke(doc):
    terms = ["stroke", "cva"]
    for e in doc.ents:
        if e.label_ in ["pathology-ichaemic", "pathology-cerebrovascular"]:
            if not e._.is_negated:
                if any([t in e.text.lower() for t in terms]):
                    return True
    return False


for i in tqdm(ischaemic_dwis.index):
    doc = docss[i]
    asserted_stroke.append(has_stroke(doc))
# %%
ischaemic_dwis["has_stroke"] = asserted_stroke
ischaemic_dwis["has_stroke"].value_counts()
# %%
ischaemic_dwis[ischaemic_dwis["has_stroke"]]["report_body"].sample(5).values
# %%
reports_asserted_strokes = ischaemic_dwis[ischaemic_dwis["has_stroke"]]
reports_asserted_strokes.to_csv(OUTDIR / "reports_assered_strokes.csv")
