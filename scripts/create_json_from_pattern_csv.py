from pathlib import Path

import numpy as np
import pandas as pd
import srsly

ROOT = Path("DATAPATH/Desktop/neuroData/domain_patterns/terminology_data")
from_corrected = pd.read_csv(ROOT / "patterns_from_labels_v1.7.csv")
# %%
# from_corrected.dropna(inplace=True)
print(from_corrected.head())
print(from_corrected["label_names"].unique())
pattern_dict = {}
for i in from_corrected["label_names"].unique():
    patterns = from_corrected[from_corrected["label_names"] == i]["name"].tolist()
    pattern_dict[i] = patterns
# %%
srsly.write_json(ROOT / "patterns.json", pattern_dict)
