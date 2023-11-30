from pathlib import Path

import pandas as pd
import srsly

ROOT = Path("DATAPATH/Desktop/neuroData/domain_patterns/terminology_data")

labels = pd.read_csv(ROOT / "lasso_entity_data_final.csv")
label_map = srsly.read_json(ROOT / "label_map.json")
labels["label"] = labels["label"].fillna("x")
# %%
readable_names = [label_map[i] for i in labels["label"]]
labels["label_names"] = readable_names
# %%
to_remove = [
    "normal-term",
    "comparative-term",
    "equivocation",
    "query-term",
    "nullclass",
    "artefact-term",
    "location-muscles",
]
terms_of_interest = labels[~labels["label_names"].isin(to_remove)]
terms_of_interest = terms_of_interest[
    ~terms_of_interest.duplicated(keep="first", subset=["name"])
]
# %%
to_save = terms_of_interest[["label_names", "name"]].sort_values(by="label_names")
# %%
to_save.to_csv(ROOT / "patterns_from_labels.csv")
# %%
