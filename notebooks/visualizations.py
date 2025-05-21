#!/usr/bin/env python
"""Visualization Plots.

Plots of the embeddings for visualizations for paper

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def load_input(location):
    """load in data"""
    return pd.read_csv(location, low_memory=False, index_col=0)


def reduce_multiclass(df, column):
    top_k_classes = df[column].value_counts().nlargest(9).index
    df[column + "_reduced"] = df.loc[:, column]
    df.loc[~df[column + "_reduced"].isin(top_k_classes), column + "_reduced"] = "Other"
    return df


def transform_data(data):
    """perform the necessary transformation on the input data"""
    cat_cols = ["Modality", "Procedure", "Specialty", "Ordering Clinician"]
    to_plot = (
        data[((data["age_at_study"] > 18.0) & (data["age_at_study"] < 99.0))]
        .pipe(reduce_multiclass, "Modality")
        .pipe(reduce_multiclass, "Specialty")
        .pipe(reduce_multiclass, "Ordering Clinician")
        .pipe(reduce_multiclass, "Procedure")
        .rename(columns={"age_at_study": "Age", "uses_contrast": "Uses Contrast"})
    )
    return to_plot


def plot_categorical(data_to_plot, column, output_loc, grey_value="", filetype="png"):
    """plot the results for a categorical column"""
    fig, ax = plt.subplots(figsize=(10, 10))
    if grey_value:
        non_grey = data_to_plot[data_to_plot[column] != grey_value]
        grey = data_to_plot[data_to_plot[column] == grey_value]
        sns.scatterplot(
            data=grey,
            x="_X",
            y="_Y",
            hue=column,
            ax=ax,
            alpha=0.5,
            s=10,
            palette="Greys",
        )
        scat = sns.scatterplot(
            data=non_grey,
            x="_X",
            y="_Y",
            hue=column,
            ax=ax,
            alpha=0.5,
            s=10,
            palette="tab10",
        )

    else:
        scat = sns.scatterplot(
            data=data_to_plot,
            x="_X",
            y="_Y",
            hue=column,
            ax=ax,
            alpha=0.5,
            s=10,
            palette="tab10",
        )
    scat.set(xlabel="latent dimension 1")
    scat.set(ylabel="latent dimension 2")
    plt.tight_layout()
    fig.savefig(
        output_loc / f"embedding_scatter_{column}.{filetype}",
        bbox_inches="tight",
        dpi=500,
    )
    plt.close()


def plot_continuous(data_to_plot, column, output_loc, filetype="png"):
    """plot data for a continuous valued column"""
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = sns.scatterplot(
        data=data_to_plot, x="_X", y="_Y", hue=column, ax=ax, alpha=0.5, s=10
    )
    scat.set(xlabel="latent dimension 1")
    scat.set(ylabel="latent dimension 2")
    plt.tight_layout()
    fig.savefig(
        output_loc / f"embedding_scatter_{column}.{filetype}", bbox_inches="tight"
    )
    plt.close()


# %% load
ROOT = Path("/home/hwatkins/Desktop/")
INPUT = ROOT / "prior-datasets/processed_reports/reports_with_embedding.csv"
OUTDIR = Path("/home/hwatkins/Dropbox/nlp_project/pipeline_paper/paper_images_v5")
FILETYPE = "png"
data = load_input(INPUT)
transformed_data = transform_data(data)
cols_to_view = [
    "Patient Sex",
    "uses_contrast",
    "IS_NORMAL",
    "IS_NORMAL_FOR_AGE",
    "Modality_reduced",
    "Specialty_reduced",
    "Ordering Clinician_reduced",
    "Age",
    "Procedure_reduced",
]
pathology_cols = [i for i in data.columns if ("asserted-pathology" in i)]
plot_categorical(transformed_data, "Patient Sex", OUTDIR, filetype=FILETYPE)
plot_continuous(transformed_data, "Age", OUTDIR, filetype=FILETYPE)


def define_progression(df, col_a, col_b):
    name_a = col_a.replace("asserted-pathology-", "")
    name_b = col_b.replace("asserted-pathology-", "")
    prog = pd.Series("other", index=df.index)
    prog[df[col_a] > 0] = f"has {name_a}"
    prog[df[col_b] > 0] = f"has {name_b}"
    prog[(df[col_a] > 0) & (df[col_b] > 0)] = f"both {name_b} and {name_a}"
    return prog


# %% intersections
with_prog = data.assign(
    ischaemia_haemorrhage_progression=lambda x: define_progression(
        x, "asserted-pathology-ischaemic", "asserted-pathology-haemorrhagic"
    )
)
plot_categorical(
    with_prog,
    "ischaemia_haemorrhage_progression",
    OUTDIR,
    grey_value="other",
    filetype=FILETYPE,
)

with_prog_2 = data.assign(
    neoplasia_treatment_progression=lambda x: define_progression(
        x,
        "asserted-pathology-treatment",
        "asserted-pathology-neoplastic-paraneoplastic",
    )
)
plot_categorical(
    with_prog_2,
    "neoplasia_treatment_progression",
    OUTDIR,
    grey_value="other",
    filetype=FILETYPE,
)

with_prog_3 = data.assign(
    constrast_treatment_intersection=lambda x: define_progression(
        x, "uses_contrast", "asserted-pathology-treatment"
    )
)
plot_categorical(
    with_prog_3,
    "constrast_treatment_intersection",
    OUTDIR,
    grey_value="other",
    filetype=FILETYPE,
)

with_prog_4 = transformed_data.assign(
    surgical=lambda x: x["Specialty_reduced"] == "NEUROSURGERY"
).assign(
    constrast_surgery_intersection=lambda x: define_progression(
        x, "uses_contrast", "surgical"
    )
)
plot_categorical(
    with_prog_4,
    "constrast_surgery_intersection",
    OUTDIR,
    grey_value="other",
    filetype=FILETYPE,
)
with_prog_5 = transformed_data.assign(
    contrast_neoplasia_intersection=lambda x: define_progression(
        x, "uses_contrast", "asserted-pathology-neoplastic-paraneoplastic"
    )
)
plot_categorical(
    with_prog_5,
    "contrast_neoplasia_intersection",
    OUTDIR,
    grey_value="other",
    filetype=FILETYPE,
)

# %% embeddings
column = "asserted-pathology-neurodegenerative-dementia"  # "asserted-pathology-inflammatory-autoimmune" #"asserted-pathology-treatment"
fig, ax = plt.subplots(figsize=(10, 10))
greys = transformed_data[transformed_data[column] < 1]
colours = transformed_data[transformed_data[column] > 0]
scat = sns.scatterplot(
    data=greys,
    x="_X",
    y="_Y",
    ax=ax,
    hue=column,
    alpha=0.2,
    s=10,
    palette="Greys",
    legend=False,
)

sns.scatterplot(
    data=colours, x="_X", y="_Y", hue="age_at_study", ax=ax, alpha=0.5, s=10
)
ax.set_title(column)
scat.set(xlabel="latent dimension 1")
scat.set(ylabel="latent dimension 2")
plt.tight_layout()
fig.savefig(OUTDIR / f"embedding_scatter_{column}.{FILETYPE}")
plt.close()

column = "asserted-pathology-ischaemic"
fig, ax = plt.subplots(figsize=(10, 10))
scat = sns.scatterplot(
    data=transformed_data,
    x="_X",
    y="_Y",
    ax=ax,
    hue=column,
    alpha=0.5,
    s=10,
)
scat.set(xlabel="latent dimension 1")
scat.set(ylabel="latent dimension 2")
plt.tight_layout()
fig.savefig(OUTDIR / f"embedding_scatter_{column}.{FILETYPE}")
plt.close()

column = "asserted-pathology-treatment"
fig, ax = plt.subplots(figsize=(10, 10))
greys = transformed_data[transformed_data[column] < 1]
colours = transformed_data[transformed_data[column] > 0]
scat = sns.scatterplot(
    data=greys,
    x="_X",
    y="_Y",
    ax=ax,
    hue=column,
    alpha=0.2,
    s=10,
    palette="Greys",
    legend=False,
)

sns.scatterplot(
    data=colours, x="_X", y="_Y", hue="age_at_study", ax=ax, alpha=0.5, s=10
)
ax.set_title(column)
scat.set(xlabel="latent dimension 1")
scat.set(ylabel="latent dimension 2")
plt.tight_layout()
fig.savefig(OUTDIR / f"embedding_scatter_{column}.{FILETYPE}")
plt.close()

# %% pairplots

column1 = "asserted-pathology-ischaemic"
column2 = "asserted-pathology-treatment"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
scat1 = sns.scatterplot(
    data=transformed_data,
    x="_X",
    y="_Y",
    ax=ax1,
    hue=column1,
    alpha=0.5,
    s=10,
)
ax1.set_xlabel("latent dimension 1", fontsize=15)
ax1.set_ylabel("latent dimension 2", fontsize=15)
ax1.legend(title="Ischaemia mentions", fontsize=15, title_fontsize=15)

greys = transformed_data[transformed_data[column2] < 1]
colours = transformed_data[transformed_data[column2] > 0]
scat2 = sns.scatterplot(
    data=greys,
    x="_X",
    y="_Y",
    ax=ax2,
    hue=column2,
    alpha=0.2,
    s=10,
    palette="Greys",
    legend=False,
)

sns.scatterplot(data=colours, x="_X", y="_Y", hue="Age", ax=ax2, alpha=0.5, s=10)
ax2.set_title(column2, fontsize=15)
ax2.set_xlabel("latent dimension 1", fontsize=15)
ax2.set_ylabel("latent dimension 2", fontsize=15)
ax2.legend(title="Age", fontsize=15, title_fontsize=15)

plt.tight_layout()
fig.savefig(OUTDIR / f"embedding_scatter_{column1}_{column2}.{FILETYPE}")
plt.close()


# %% classes
def neoplasm_classes(df):
    classes = pd.Series("not neoplastic", index=df.index)
    classes[df["asserted-pathology-neoplastic-paraneoplastic"] > 0] = "other neoplastic"
    has_schwanoma = (
        df["report_body_masked"].str.contains("glioblastoma", case=False)
    ) & (df["asserted-pathology-neoplastic-paraneoplastic"] > 0)
    has_glioma = (df["report_body_masked"].str.contains("astrocytoma", case=False)) & (
        df["asserted-pathology-neoplastic-paraneoplastic"] > 0
    )
    has_meningioma = (
        df["report_body_masked"].str.contains("oligodendroglioma", case=False)
    ) & (df["asserted-pathology-neoplastic-paraneoplastic"] > 0)
    classes[has_schwanoma] = "glioblastoma"
    classes[has_glioma] = "astrocytoma"
    classes[has_meningioma] = "oligodendroglioma"
    return classes


neo_type = transformed_data.assign(neoplastic_types=neoplasm_classes)
print(neo_type["neoplastic_types"].value_counts())
plot_categorical(
    neo_type, "neoplastic_types", OUTDIR, grey_value="not neoplastic", filetype=FILETYPE
)


def vascular_classes(df):
    classes = pd.Series("not vascular", index=df.index)
    classes[df["asserted-pathology-vascular"] > 0] = "other vascular"
    has_aneurysm = (df["asserted-pathology-vascular"] > 0) & (
        df["report_body_masked"].str.contains("aneurysm", case=False)
    )
    has_cavernoma = (df["asserted-pathology-vascular"] > 0) & (
        df["report_body_masked"].str.contains("cavernoma", case=False)
    )
    classes[has_aneurysm] = "aneurysm"
    classes[has_cavernoma] = "cavernoma"
    return classes


vasc_type = transformed_data.assign(vascular_types=vascular_classes)
plot_categorical(
    vasc_type, "vascular_types", OUTDIR, grey_value="not vascular", filetype=FILETYPE
)


# %% cat-class-pairs


def plot_pair_categorical(
    data_to_plot,
    column1,
    column2,
    output_loc,
    grey_value_1,
    grey_value_2,
    filetype="png",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    non_grey = data_to_plot[data_to_plot[column1] != grey_value_1]
    grey = data_to_plot[data_to_plot[column1] == grey_value_1]
    sns.scatterplot(
        data=grey,
        x="_X",
        y="_Y",
        hue=column1,
        ax=ax1,
        alpha=0.1,
        s=10,
        palette="Greys",
    )
    sns.scatterplot(
        data=non_grey,
        x="_X",
        y="_Y",
        hue=column1,
        ax=ax1,
        alpha=0.5,
        s=10,
        palette="tab10",
    )
    ax1.set_xlabel("latent dimension 1", fontsize=15)
    ax1.set_ylabel("latent dimension 2", fontsize=15)
    ax1.legend(title=column1.replace("_", " "), fontsize=15, title_fontsize=15)

    non_grey = data_to_plot[data_to_plot[column2] != grey_value_2]
    grey = data_to_plot[data_to_plot[column2] == grey_value_2]

    sns.scatterplot(
        data=grey,
        x="_X",
        y="_Y",
        hue=column2,
        ax=ax2,
        alpha=0.1,
        s=10,
        palette="Greys",
    )
    sns.scatterplot(
        data=non_grey,
        x="_X",
        y="_Y",
        hue=column2,
        ax=ax2,
        alpha=0.5,
        s=10,
        palette="tab10",
    )
    ax2.set_xlabel("latent dimension 1", fontsize=15)
    ax2.set_ylabel("latent dimension 2", fontsize=15)
    ax2.legend(title=column2.replace("_", " "), fontsize=15, title_fontsize=15)
    plt.tight_layout()
    fig.savefig(
        output_loc / f"embedding_scatter_{column1}_{column2}.{filetype}",
        bbox_inches="tight",
    )
    plt.close()


vasc_type = transformed_data.assign(
    ischaemia_haemorrhage_progression=lambda x: define_progression(
        x, "asserted-pathology-ischaemic", "asserted-pathology-haemorrhagic"
    )
).assign(vascular_types=vascular_classes)


plot_pair_categorical(
    vasc_type,
    "ischaemia_haemorrhage_progression",
    "vascular_types",
    OUTDIR,
    "other",
    "not vascular",
    filetype=FILETYPE,
)

# %% cat-class-pair-2


neo_type = transformed_data.assign(
    neoplasia_treatment_progression=lambda x: define_progression(
        x,
        "asserted-pathology-treatment",
        "asserted-pathology-neoplastic-paraneoplastic",
    )
).assign(neoplastic_types=neoplasm_classes)

plot_pair_categorical(
    neo_type,
    "neoplasia_treatment_progression",
    "neoplastic_types",
    OUTDIR,
    "other",
    "not neoplastic",
    filetype=FILETYPE,
)

# %% contrastpair

contrast_pair = transformed_data.assign(
    contrast_neoplasia_intersection=lambda x: define_progression(
        x, "Uses Contrast", "asserted-pathology-neoplastic-paraneoplastic"
    )
).assign(
    constrast_treatment_intersection=lambda x: define_progression(
        x, "Uses Contrast", "asserted-pathology-treatment"
    )
)


plot_pair_categorical(
    contrast_pair,
    "contrast_neoplasia_intersection",
    "constrast_treatment_intersection",
    OUTDIR,
    "other",
    "other",
    filetype=FILETYPE,
)
