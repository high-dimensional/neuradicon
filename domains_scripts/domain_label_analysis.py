from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn

seaborn.set()

# %%
data = "DATAPATH/Desktop/neuroData/cleaned_report_data/process_reports_joined_v4.csv"

df = pd.read_csv(data)

# %%
descriptors = df["descriptor_domains_v2"].fillna("no label")
all_domain_types = descriptors.str.split("|").tolist()
all_domains = pd.DataFrame(
    data=[j for i in all_domain_types for j in i], columns=["domain label"]
)

# %%
counts = all_domains["domain label"].value_counts()
labels = counts.index.tolist()
fig, ax = plt.subplots()
ax = seaborn.countplot(data=all_domains, x="domain label")
ax.set_xticklabels(labels, rotation=90)
fig.tight_layout()
fig.savefig("descriptor_domain_counts.png")
plt.show()
# %%
# %%
pathologies = df["pathological_domains_v2"].fillna("no label")
all_domain_types = pathologies.str.split("|").tolist()
all_domains = pd.DataFrame(
    data=[j for i in all_domain_types for j in i], columns=["domain label"]
)

# %%
counts = all_domains["domain label"].value_counts()
labels = counts.index.tolist()
fig, ax = plt.subplots()
ax = seaborn.countplot(data=all_domains, x="domain label")
ax.set_xticklabels(labels, rotation=90)
fig.tight_layout()
fig.savefig("pathology_domain_counts.png")
plt.show()
