"""Find stroke nearest neighbours

Use patients from pdf/ssnap datasets who are known to have strokes,
then find their locations in the latent space, and find similar reports
to those patients.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# %%
ROOT = Path("DATAPATH/Desktop")
DATA = ROOT / "neuroNLP/runs/path_model_030123"
STROKEMRNS = (
    ROOT
    / "db_queries/stroke_reports/patientIDs_list_confirmed_stroke_with_pdf_ssnap.csv"
)
OUTDIR = ROOT / "db_queries/stroke_reports"


strokemrns = pd.read_csv(STROKEMRNS, names=["MRN"]).astype(str)
df = pd.read_csv(DATA / "report_data.csv", low_memory=False)

# %%
stroke_reports = df[df["MRN"].isin(strokemrns["MRN"])]
print(len(stroke_reports))
# %%
stroke_points = stroke_reports[["_X", "_Y"]].to_numpy()
X = df[["_X", "_Y"]].to_numpy()

nbrs = NearestNeighbors(n_neighbors=2).fit(X)
# %%
dists, idxs = nbrs.kneighbors(stroke_points)

all_idxs = list({neighbor for stroke in idxs for neighbor in stroke})
print(len(all_idxs))
extra_strokes = df.iloc[all_idxs]

# %%
plt.figure(figsize=(20, 20))
plt.scatter(df["_X"], df["_Y"], c="b")
plt.scatter(extra_strokes["_X"], extra_strokes["_Y"], c="r")
plt.scatter(stroke_reports["_X"], stroke_reports["_Y"], c="k")
plt.show()

# %%
extra_strokes["report_body"].sample(5).values
# %%
has_ischaemia = extra_strokes[extra_strokes["asserted-pathology-ischaemic"] > 0]

# %%
len(has_ischaemia)
has_ischaemia["MRN"].value_counts()
# %%
has_ischaemia.loc[
    ~has_ischaemia["MRN"].isin(strokemrns["MRN"]), "MRN"
].value_counts().index.to_series().reset_index(drop=True).to_csv(
    OUTDIR / "extra_stroke_mrns.csv", index=False
)
