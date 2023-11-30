#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

codes_of_interest = ["C71", "I63", "Q03", "G35"]
code_columns = ["has_" + i for i in codes_of_interest]
use_cols = ["_X", "_Y"] + code_columns

ROOT = Path("DATAPATH/Desktop/")
INPUT = ROOT / "prior-datasets/processed_reports/reports_with_icd10.csv"
OUTDIR = Path("DATAPATH/Dropbox/nlp_project/pipeline_paper/paper_images_v4")
df = pd.read_csv(INPUT, usecols=use_cols).rename(
    columns={
        "has_C71": "has C71",
        "has_I63": "has I63",
        "has_Q03": "has Q03",
        "has_G35": "has G35",
    }
)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))


nans = df[df["has C71"].isna()]
not_na = df[df["has C71"].notna()]
has = not_na[not_na["has C71"]]
# has_not = not_na[~(not_na[c])]
ax1.scatter(nans["_X"], nans["_Y"], alpha=0.1, s=1, c="gray", label="no icd data")
ax1.scatter(has["_X"], has["_Y"], c="r", s=1, label="has C71")
ax1.set_title("has C71", fontsize=15)
# plt.scatter(has_not['_X'], has_not['_Y'],c='b', s=0.1)
ax1.legend(title="has C71", fontsize=15, title_fontsize=15)
ax1.grid(True)

nans = df[df["has I63"].isna()]
not_na = df[df["has I63"].notna()]
has = not_na[not_na["has I63"]]
# has_not = not_na[~(not_na[c])]
ax2.scatter(nans["_X"], nans["_Y"], alpha=0.1, s=1, c="gray", label="no icd data")
ax2.scatter(has["_X"], has["_Y"], c="r", s=1, label="has I63")
ax2.set_title("has I63", fontsize=15)
# plt.scatter(has_not['_X'], has_not['_Y'],c='b', s=0.1)
ax2.legend(title="has I63", fontsize=15, title_fontsize=15)
ax2.grid(True)

nans = df[df["has Q03"].isna()]
not_na = df[df["has Q03"].notna()]
has = not_na[not_na["has Q03"]]
# has_not = not_na[~(not_na[c])]
ax3.scatter(nans["_X"], nans["_Y"], alpha=0.1, s=1, c="gray", label="no icd data")
ax3.scatter(has["_X"], has["_Y"], c="r", s=1, label="has Q03")
ax3.set_title("has Q03", fontsize=15)
# plt.scatter(has_not['_X'], has_not['_Y'],c='b', s=0.1)
ax3.legend(title="has Q03", fontsize=15, title_fontsize=15)
ax3.grid(True)

nans = df[df["has G35"].isna()]
not_na = df[df["has G35"].notna()]
has = not_na[not_na["has G35"]]
# has_not = not_na[~(not_na[c])]
ax4.scatter(nans["_X"], nans["_Y"], alpha=0.1, s=1, c="gray", label="no icd data")
ax4.scatter(has["_X"], has["_Y"], c="r", s=1, label="has G35")
ax4.set_title("has G35", fontsize=15)
# plt.scatter(has_not['_X'], has_not['_Y'],c='b', s=0.1)
ax4.legend(title="has G35", fontsize=15, title_fontsize=15)
ax4.grid(True)

plt.tight_layout()
fig.savefig(OUTDIR / "icd10_codes.png")
plt.close()
