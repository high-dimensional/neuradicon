from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

matplotlib.rcParams["font.family"] = "DIN Pro"
sns.set_theme(style="white")
# %%

ROOT = Path("DATAPATH/Desktop/neuroData")
DATA = (
    ROOT / "processed_reports" / "processed_reports_with_hac_101022.csv"
)  # "processed_reports_101022.csv"
out_path = Path("DATAPATH/Dropbox")
df = pd.read_csv(DATA, low_memory=False)
# %%
df["report_length"] = df["report_body"].str.len()
# %%

POINTSIZE = 11
ALPHA = 0.5
DPI = 200
# %%

N = 250000
print(len(df))
sample = df.sample(N)

# %%
kde_sample = 100000
x, y = sample["X"].to_numpy(), sample["Y"].to_numpy()
xy = np.vstack([x, y])
kde = gaussian_kde(xy)
z = kde(xy)
sample["density_colour"] = z

# %%
pathology_columns = [
    "asserted-pathology-cerebrovascular",
    "asserted-pathology-congenital-developmental",
    "asserted-pathology-csf-disorders",
    "asserted-pathology-endocrine",
    "asserted-pathology-haemorrhagic",
    "asserted-pathology-infectious",
    "asserted-pathology-inflammatory-autoimmune",
    "asserted-pathology-ischaemic",
    "asserted-pathology-metabolic-nutritional-toxic",
    "asserted-pathology-musculoskeletal",
    "asserted-pathology-neoplastic-paraneoplastic",
    "asserted-pathology-neurodegenerative-dementia",
    "asserted-pathology-opthalmological",
    "asserted-pathology-traumatic",
    "asserted-pathology-treatment",
    "asserted-pathology-vascular",
]
# %%
# max_dens = sample["density_colour"].max()
# clamp_val = 0.01*max_dens
# sample["density_colour_clamped"] = sample["density_colour"]
# sample.loc[sample["density_colour"]>clamp_val, "density_colour_clamped"] = clamp_val
sample["density_colour_clamped"] = np.log10(sample["density_colour"])

# %%
for i in pathology_columns:
    sample[i] = sample[i].astype(bool)
    fig, ax = plt.subplots(figsize=(20, 20))
    has_col = sample[sample[i]]
    not_col = sample[~sample[i]]
    sns.scatterplot(
        data=has_col,
        x="X",
        y="Y",
        hue="density_colour_clamped",
        ax=ax,
        alpha=ALPHA,
        palette="Oranges",
        s=POINTSIZE,
        legend=False,
    )
    sns.scatterplot(
        data=not_col,
        x="X",
        y="Y",
        hue="density_colour_clamped",
        ax=ax,
        alpha=ALPHA,
        palette="Blues",
        s=POINTSIZE,
        legend=False,
    )
    ax.legend(
        handles=[
            mpatches.Circle((0, 0), color="tab:orange", radius=1),
            mpatches.Circle((0, 0), color="tab:blue", radius=1),
        ],
        labels=["With", "Without"],
        title=f"Report has {i}",
    )
    ax.set_title(f"2d pathology embedding of radiological reports - {i}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path / f"embedding-{i}.png", dpi=DPI)
    plt.close("all")
# %%

# %%
location_columns = [
    "asserted-location-arteries",
    "asserted-location-brain-stem",
    "asserted-location-diencephalon",
    "asserted-location-eye",
    "asserted-location-ganglia",
    "asserted-location-grey-matter",
    "asserted-location-limbic-system",
    "asserted-location-meninges",
    "asserted-location-nerves",
    "asserted-location-neurosecretory-system",
    "asserted-location-other",
    "asserted-location-qualifier",
    "asserted-location-skull",
    "asserted-location-spine",
    "asserted-location-telencephalon",
    "asserted-location-veins",
    "asserted-location-ventricles",
    "asserted-location-white-matter",
]
descriptor_columns = [
    "asserted-descriptor-cyst",
    "asserted-descriptor-damage",
    "asserted-descriptor-diffusion",
    "asserted-descriptor-signal-change",
    "asserted-descriptor-enhancement",
    "asserted-descriptor-flow",
    "asserted-descriptor-interval-change",
    "asserted-descriptor-mass-effect",
    "asserted-descriptor-morphology",
    "asserted-descriptor-collection",
]
# %%

x, y = sample["loc_X"].to_numpy(), sample["loc_Y"].to_numpy()
xy = np.vstack([x, y])
kde = gaussian_kde(xy)
z = kde(xy)
sample["density_colour_loc"] = z

# %%
"""
max_dens = sample["density_colour_loc"].max()
clamp_val = 0.01*max_dens
sample["density_colour_clamped_loc"] = sample["density_colour_loc"]
sample.loc[sample["density_colour_loc"]>clamp_val, "density_colour_clamped_loc"] = clamp_val
"""
sample["density_colour_clamped_loc"] = np.log10(sample["density_colour_loc"])

# %%
for i in location_columns:
    sample[i] = sample[i].astype(bool)
    fig, ax = plt.subplots(figsize=(20, 20))
    has_col = sample[sample[i]]
    not_col = sample[~sample[i]]
    sns.scatterplot(
        data=has_col,
        x="loc_X",
        y="loc_Y",
        hue="density_colour_clamped_loc",
        ax=ax,
        alpha=ALPHA,
        palette="Oranges",
        s=POINTSIZE,
        legend=False,
    )
    sns.scatterplot(
        data=not_col,
        x="loc_X",
        y="loc_Y",
        hue="density_colour_clamped_loc",
        ax=ax,
        alpha=ALPHA,
        palette="Blues",
        s=POINTSIZE,
        legend=False,
    )
    ax.legend(
        handles=[
            mpatches.Circle((0, 0), color="tab:orange", radius=1),
            mpatches.Circle((0, 0), color="tab:blue", radius=1),
        ],
        labels=["With", "Without"],
        title=f"Report has {i}",
    )
    ax.set_title(f"2d location embedding of radiological reports - {i}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path / f"embedding-{i}.png", dpi=DPI)
    plt.close("all")
# %%
for i in descriptor_columns:
    sample[i] = sample[i].astype(bool)
    fig, ax = plt.subplots(figsize=(20, 20))
    has_col = sample[sample[i]]
    not_col = sample[~sample[i]]
    sns.scatterplot(
        data=has_col,
        x="X",
        y="Y",
        hue="density_colour_clamped",
        ax=ax,
        alpha=ALPHA,
        palette="Oranges",
        s=POINTSIZE,
        legend=False,
    )
    sns.scatterplot(
        data=not_col,
        x="X",
        y="Y",
        hue="density_colour_clamped",
        ax=ax,
        alpha=ALPHA,
        palette="Blues",
        s=POINTSIZE,
        legend=False,
    )
    ax.legend(
        handles=[
            mpatches.Circle((0, 0), color="tab:orange", radius=1),
            mpatches.Circle((0, 0), color="tab:blue", radius=1),
        ],
        labels=["With", "Without"],
        title=f"Report has {i}",
    )
    ax.set_title(f"2d descriptor embedding of radiological reports - {i}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path / f"embedding-{i}.png", dpi=DPI)
    plt.close("all")
# %%
to_plot = ["age_at_study", "report_length"]
fig, ax = plt.subplots(figsize=(20, 20))
age_sample = sample[(sample["age_at_study"] > 18) & (sample["age_at_study"] < 100)]
sns.scatterplot(
    data=age_sample, x="X", y="Y", hue="age_at_study", ax=ax, s=POINTSIZE, alpha=ALPHA
)
ax.set_title(f"2d embedding of radiological reports - age_at_study")
ax.axis("off")
fig.tight_layout()
fig.savefig(out_path / f"embedding-age_at_study.png", dpi=DPI)
# %%

fig, ax = plt.subplots(figsize=(20, 20))
sample["log_report_length"] = np.log10(sample["report_length"])
sns.scatterplot(
    data=sample, x="X", y="Y", hue="log_report_length", ax=ax, s=POINTSIZE, alpha=ALPHA
)
ax.set_title(f"2d embedding of radiological reports - log_report_length")
ax.axis("off")
fig.tight_layout()
fig.savefig(out_path / f"embedding-report_length.png", dpi=DPI)
# %%
to_plot = [
    "Patient Sex",
    "Modality",
    "Procedure",
    "uses_contrast",
    "IS_NORMAL",
    "IS_COMPARATIVE",
]
for j in to_plot:
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(data=sample, x="X", y="Y", hue=j, ax=ax, s=POINTSIZE, alpha=ALPHA)
    ax.set_title(f"2d embedding of radiological reports - {j}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path / f"embedding-{j}.png", dpi=DPI)
# %%
rename_dict = {
    i: i.replace("asserted-pathology-", "").replace("asserted-location", "")
    for i in pathology_columns + location_columns
}
tocorr = (
    sample[pathology_columns + location_columns]
    .astype(bool)
    .astype(int)
    .rename(columns=rename_dict)
)
# %%
corr = tocorr.corr()
corrtoplot = corr.iloc[: len(pathology_columns), len(pathology_columns) :]
# %%
fig, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corrtoplot,
    cmap=cmap,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)
ax.set(xlabel="Locations", ylabel="Pathological domains", title="Correlation")
fig.tight_layout()
fig.savefig(out_path / "correlation.png", dpi=DPI)


# %%
M = 50000

n_clusters = 100
# clus = AgglomerativeClustering(n_clusters=55)
point_sample = sample[["X", "Y"]].sample(M).to_numpy()
points = sample[["X", "Y"]].to_numpy()
Z = hierarchy.linkage(point_sample, method="ward")
cut_tree = hierarchy.cut_tree(Z, n_clusters=[10, n_clusters])
# labels_sample = clus.fit_predict(point_sample)
# %%
high_knn = KNeighborsClassifier()
low_knn = KNeighborsClassifier()

high_knn.fit(point_sample, cut_tree[:, 1])
low_knn.fit(point_sample, cut_tree[:, 0])
high_labels = high_knn.predict(points)
low_labels = low_knn.predict(points)

sample["hac_cluster_high"] = high_labels
sample["hac_cluster_low"] = low_labels

# %%
n_pixels = 2048
xmin, xmax = points[:, 0].min(), points[:, 0].max()
ymin, ymax = points[:, 1].min(), points[:, 1].max()
x_field = np.linspace(xmin, xmax, num=n_pixels)
y_field = np.linspace(ymin, ymax, num=n_pixels)
XX, YY = np.meshgrid(x_field, y_field)
coords = np.array([XX, YY]).transpose([1, 2, 0])

background = low_knn.predict(coords.reshape(n_pixels**2, 2))
background = background.reshape(n_pixels, n_pixels)

# %%
sample["cluster_color_val"] = 0.0
for i in sample["hac_cluster_high"].unique():
    lx = sample["X"].max() - sample["X"].min()
    ly = sample["Y"].max() - sample["Y"].min()
    cluster_rows = sample[sample["hac_cluster_high"] == i]
    # value = cluster_rows[["X","Y"]].to_numpy().mean()
    xandy = cluster_rows[["X", "Y"]].to_numpy()
    # z = xandy[:,0].mean() + 1j*xandy[:,1].mean()
    k = 2.0
    z = np.sin(k * 2 * 3.14 * xandy[:, 0].mean() / lx) + np.cos(
        k * 2 * 3.14 * xandy[:, 1].mean() / ly
    )
    # value = np.angle(z)**2 + np.abs(z)**2
    # value = xandy[:,0].mean() + xandy[:,1].mean()**3
    sample.loc[sample["hac_cluster_high"] == i, "cluster_color_val"] = z
# %%
sample["cluster_color_val"].value_counts()
# %%
new_background = np.zeros(shape=background.shape)
for p in np.unique(background):
    meanx = XX[background == p].mean()
    meany = YY[background == p].mean()
    new_background[background == p] = np.angle(meanx + 1j * meany) ** 2
# %%
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title(f"pathology clustering of radiological reports")
ax.axis("off")
ax.contourf(XX, YY, background, levels=10, alpha=0.2, cmap="Pastel1")

sns.scatterplot(
    data=sample,
    x="X",
    y="Y",
    hue="cluster_color_val",
    ax=ax,
    palette="turbo",
    s=POINTSIZE,
    alpha=0.9,
    legend=False,
)
for k in sample["hac_cluster_high"].unique():
    cluster_rows = sample[sample["hac_cluster_high"] == k]
    most_freq_low_clus = cluster_rows["hac_cluster_low"].mode().tolist()[0]
    l = len(cluster_rows)
    mean_x, mean_y = cluster_rows["X"].mean(), cluster_rows["Y"].mean()
    path_sums = cluster_rows[pathology_columns].sum().T
    factors = path_sums.sort_values().index[-3:].tolist()
    factors = [f[19:] for f in factors]
    ax.text(mean_x, mean_y, factors[0], fontsize="small")
    ax.text(mean_x, mean_y - 0.017, factors[1], fontsize="x-small")
    ax.text(mean_x, mean_y - 0.029, factors[2], fontsize="xx-small")
fig.tight_layout()
fig.savefig(out_path / f"pathology-clustering-hac.png", dpi=DPI)


# %%

# %%
sample.to_csv(ROOT / "processed_reports" / "processed_reports_with_hac_v2_101022.csv")
