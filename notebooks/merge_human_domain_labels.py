"""add human labels to dataframe

purpose of this script is to add extra columns to the report data to apply
gold-standard domain labels, if available
"""

import numpy as np
import pandas as pd
import srsly

REPORTS = "DATAPATH/Desktop/neuroData/processed_reports/reports_with_embedding.csv"
LABELS = "DATAPATH/Desktop/neuroData/domains_data/ucl-evaluation-data/domain-review-full-alignopth.jsonl"
reports = pd.read_csv(REPORTS, low_memory=False)
labels = list(srsly.read_jsonl(LABELS))

# %%
reports.sample(5)
all_label_texts = [i["text"] for i in labels]
all_categories = {j for t in labels for j in t["accept"]}
# %%
in_labels = reports[
    (reports["report_body"].isin(all_label_texts))
    | (reports["report_body_masked"].isin(all_label_texts))
]
# %%
import numpy as np

output_df = reports
for cat in all_categories:
    col_name = "human-" + cat
    labelled_rows = {col_name: np.nan}
    output_df = output_df.assign(**labelled_rows)
    texts_with_cat = [i["text"] for i in labels if cat in i["accept"]]
    false_condition = (output_df["report_body"].isin(all_label_texts)) | (
        output_df["report_body_masked"].isin(all_label_texts)
    )
    output_df.loc[false_condition, col_name] = False
    true_condition = (output_df["report_body"].isin(texts_with_cat)) | (
        output_df["report_body_masked"].isin(texts_with_cat)
    )
    output_df.loc[true_condition, col_name] = True

# %%
output_df["human-pathology-neoplastic-paraneoplastic"].value_counts()
output_df[~output_df["human-pathology-neoplastic-paraneoplastic"].isna()]

# %%
output_df.to_csv(
    "DATAPATH/Desktop/neuroData/processed_reports/reports_with_gold_labels.csv",
    index=False,
)
