"""
Look at the reports that are classified as normal, but contain positive pathological classes
"""
import pandas as pd

df = pd.read_csv("DATAPATH/Desktop/neuroData/processed_reports/processed_reports.csv")

# %%
normal_label = df[~df["IS_NORMAL"].isna()]

# %%
path_columns = df.columns[df.columns.str.contains("asserted-pat")]
# %%
has_pathology = (normal_label[path_columns] > 0).any(axis=1)
normals_with_path = normal_label[has_pathology & normal_label["IS_NORMAL"]]
# %%
normals_with_path.to_csv("normals_with_pathology.csv")
