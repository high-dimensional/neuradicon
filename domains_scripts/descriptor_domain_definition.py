import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from neuroNLP.custom_pipes import *
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from spacy.tokens import DocBin
from tqdm import tqdm
from umap import UMAP

# %%
ROOT = Path("DATAPATH/Desktop")

data = ROOT / "neuroData/170k_spacy_docs_bodies_new_format"
model = ROOT / "neuroNLP/training/rel_neg_detection_tok2vec_v2"
extra_labels = ROOT / "neuroData/domain_patterns/pathology_orphans.csv"
labelled_terms_df = pd.read_csv(extra_labels, index_col=False)
# %%
nlp = spacy.load(model)

doc_bin = DocBin().from_disk(data / "neuronlp_docs.spacy")
docs = doc_bin.get_docs(nlp.vocab)
# %%
all_descriptors = [
    e.text.lower() for doc in docs for e in doc.ents if e.label_ == "DESCRIPTOR"
]
all_descriptors = list(all_descriptors)
counts = Counter(all_descriptors)


# %%
def percentile(counter, percentile):
    n_for_percentile = percentile * sum(counts.values())
    n = 0
    while sum(dict(counter.most_common(n)).values()) <= n_for_percentile:
        n += 1
    return dict(counter.most_common(n))


# %%
nintieth_percentile = percentile(counts, 0.95)
# %%
labelled_terms = labelled_terms_df[labelled_terms_df["assignment_new"] == "descriptor"][
    "term"
].tolist()
# %%
archetypes = {
    "signal": [
        "signal change",
        "hyperintensity",
        "signal abnormality",
        "dwi",
        "flair",
        "confluence",
        "t2",
        "t1",
        "signal",
        "t2 hyperintense foci",
        "white matter signal change",
    ],
    "enhancement": [
        "enhancement",
        "lesion",
        "cyst",
        "enhancing",
        "enhance",
        "contrast enhancement",
        "dural enhancement",
    ],
    "artefact": ["artefact", "susceptibility", "susceptibility artefact"],
    "diffusion": ["restricted diffusion"],
    "treatment": ["radiotherapy", "post-operative", "surgery", "surgical"],
    "intervalchange": ["interval change"],
    "flow/csf": [
        "flow voids",
        "csf",
        "flow",
    ],
    "morphology": [
        "volume loss",
        "atrophy",
        "enlargement",
        "dilatation",
        "stenosis",
        "narrowing",
        "swelling",
        "oedema",
        "thickening",
        "ectopia",
        "enlarged",
        "displacement",
    ],
    "mass": ["mass effect", "tumour", "neoplasm", "high-grade tumour", "effacement"],
}

# %%
all_terms = list(nintieth_percentile.keys())
all_terms.extend(labelled_terms)

is_archetype = [False for i in range(len(all_terms))]
domains = ["" for i in range(len(all_terms))]

archetype_terms, archetype_domain = zip(
    *[(t, k) for k, v in archetypes.items() for t in v]
)
all_terms.extend(list(archetype_terms))
domains.extend(list(archetype_domain))
is_archetype.extend([True for i in range(len(archetype_terms))])

labelled_counts = labelled_terms_df[
    labelled_terms_df["assignment_new"] == "descriptor"
]["frequency"].tolist()
all_counts = list(nintieth_percentile.values())
all_counts.extend(labelled_counts)
all_counts.extend([1 for i in range(len(archetype_terms))])

df = pd.DataFrame(
    data={
        "terms": all_terms,
        "frequency": all_counts,
        "domain": domains,
        "is_archetype": is_archetype,
    }
)

# %%
all_vectors = [nlp.make_doc(term).vector for term in all_terms]
all_vectors = np.stack(all_vectors)
features = all_vectors


# %%
NCOMPONENTS = 50
# reducer1 = SparsePCA(n_components=NCOMPONENTS)
# reduction1 = reducer1.fit_transform(features)
reducer3 = SpectralEmbedding(n_components=NCOMPONENTS)
reduction3 = reducer3.fit_transform(features)
# reducer5 = UMAP(n_components=NCOMPONENTS)
# reduction5 = reducer5.fit_transform(features)
# %%
fig, ax = plt.subplots(1, 3, figsize=(30, 10))
ax[0].scatter(reduction1[:, 10], reduction1[:, 1])
ax[1].scatter(reduction3[:, 10], reduction3[:, 1])
ax[2].scatter(reduction5[:, 10], reduction5[:, 1])
plt.show()

# %%
N_CLUSTERS = 50
cluster1 = SpectralClustering(n_clusters=N_CLUSTERS)

# %%
# clusters1s = cluster1.fit_predict(reduction1)
clusters2s = cluster1.fit_predict(reduction3)
# clusters4s = cluster1.fit_predict(reduction5)
# %%
fig, ax = plt.subplots(1, 3, figsize=(30, 10))
ax[0].scatter(reduction1[:, 10], reduction1[:, 11], c=clusters1s)
ax[1].scatter(reduction3[:, 10], reduction3[:, 11], c=clusters2s)
ax[2].scatter(reduction5[:, 10], reduction5[:, 11], c=clusters4s)
plt.show()
# %%

# df["pcacluster"] = clusters1s
df["speccluster"] = clusters2s
# df["umapcluster"] = clusters4s

# %%
method = "speccluster"
for clus in df[method].unique():
    print(f"samples from cluster: {clus}")
    show_terms = df[df[method] == clus].sort_values(by="frequency", ascending=False)
    if len(show_terms) > 20:
        print(show_terms["terms"][:20])
    else:
        print(show_terms["terms"].sample(20, replace=True))


# %%
fig, ax = plt.subplots(1, 3, figsize=(30, 10))
ax[0].scatter(reduction1[:, 0], reduction1[:, 1], c=df["is_archetype"])
ax[1].scatter(reduction3[:, 0], reduction3[:, 1], c=df["is_archetype"])
ax[2].scatter(reduction5[:, 0], reduction5[:, 1], c=df["is_archetype"])
plt.show()
# %%

k = 5
r = 0.002
nbrs = NearestNeighbors(radius=r).fit(reduction3)
archetype_vecs = reduction3[df["is_archetype"]]
dists, inds = nbrs.radius_neighbors(archetype_vecs)
# %%
for i, idx in zip(df[df["is_archetype"]]["terms"], inds):
    print("\narchetype:", i)
    for j in idx:
        print(df["terms"][j])
# %%
df2 = df.copy()
for d, idxs in zip(df[df["is_archetype"]]["domain"], inds):
    for i in idxs:
        df2.iloc[i, 2] = d
df2["proportion"] = df2["frequency"] / df2["frequency"].sum()
# %%
total_counts = df2["frequency"].sum()
coverage = df2[df2["domain"] != ""]["frequency"].sum() / total_counts
print(coverage)
unassigned = df2[df2["domain"] == ""][["terms", "proportion"]][:50]
print(f"top 10 un assigned terms\n {unassigned}")
# %%
print(unassigned["proportion"].sum())
# %%
with open(ROOT / "neuroData/domain_patterns/descriptor_archetype.json", "w") as file:
    json.dump(archetypes, file)
# %%
df2.to_csv(ROOT / "neuroData/domain_patterns/descriptor_domain_allocations.csv")
# %%
all_patterns = {}
for d in archetypes.keys():
    terms = df2[df2["domain"] == d]["terms"].tolist()
    all_patterns[d] = terms
# %%
with open(
    ROOT / "neuroData/domain_patterns/descriptor_patterns_v2/patterns.json", "w"
) as file:
    json.dump(all_patterns, file)
# %%
dict2save = {}
for name, clusters in allocations.items():
    cluster_patterns = []
    for cls in clusters:
        terms = df[(df["cluster"] == cls) & (df["frequency"] > 5)]["terms"].tolist()
        cluster_patterns.extend(terms)
    dict2save[name] = cluster_patterns

with open(
    ROOT / "neuroData/domain_patterns/descriptor_patterns/patterns.json", "w"
) as file:
    json.dump(dict2save, file)
