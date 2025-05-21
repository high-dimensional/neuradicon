#!/usr/bin/env python
"""Visualization Plots.

Plots of the embeddings for visualizations for paper.
Specifically the nifti SPM outputs from geoSPM

"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
from matplotlib.colors import CenteredNorm


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
    )
    return to_plot


ROOT = Path("DATAPATH/Desktop/")
INPUT = ROOT / "prior-datasets/processed_reports/reports_with_embedding.csv"
SPMROOT = ROOT / "geospm_neuro/runs/2023_02_14_17_39_10"
SPMINPUT = SPMROOT / "spm_output"
OUTDIR = Path(
    "DATAPATH/Dropbox/nlp_project/pipeline_paper/paper_images_v4"
)  # ROOT / "paper_images"

FILETYPE = "png"

data = load_input(INPUT)
transformed_data = transform_data(data)

name_mapping = {
    "0001": "age at study",
    "0002": "haemorrhagic",
    "0003": "ischaemic",
    "0006": "treatment",
    "0007": "inflammatory-autoimmune",
    "0011": "neoplastic-paraneoplastic",
    "0013": "neurodegenerative-dementia",
}
# %% plot1
variable = "0001"
variable_name = name_mapping[variable]
imtype = "beta"
level = 7
smoothing_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
level_name = smoothing_levels[level]

name = imtype + "_" + variable
img = nibabel.load(SPMINPUT / f"{name}.nii")
spm_map = img.get_fdata()
spm_mask = np.rot90(spm_map[:, :, level])
data_extent = (-2.0, 2.5, -3.6, 2.2)
plt.figure(figsize=(12, 10))
plt.scatter(data["_X"], data["_Y"], c="tab:gray", alpha=0.5, s=1, zorder=0)
plt.xlabel("latent dimension 1")
plt.ylabel("latent dimension 2")
plt.imshow(
    spm_mask,
    extent=data_extent,
    alpha=0.8,
    zorder=1,
    cmap="RdBu_r",
    norm=CenteredNorm(),
    aspect="auto",
)
plt.colorbar(
    label="regression coefficient", shrink=1.0, pad=0.01, aspect=25, location="right"
)
plt.tight_layout()
plt.title(variable_name + f" - smoothing level - {level_name}")
# plt.show()
plt.savefig(
    OUTDIR / f"{variable_name}_spm_map.{FILETYPE}", dpi=300, bbox_inches="tight"
)
plt.close()
# %% plot2
variable_1 = "0002"
variable_2 = "0003"
variable_name_1 = name_mapping[variable_1]
variable_name_2 = name_mapping[variable_2]
mask1 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_1}_mask.nii")
mask2 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_2}_mask.nii")
mask1_map = mask1.get_fdata()
mask1_map = np.rot90(mask1_map[:, :, level])
mask2_map = mask2.get_fdata()
mask2_map = np.rot90(mask2_map[:, :, level])

intersection = np.logical_and(mask1_map > 0, mask2_map > 0).astype(int)
plt.figure(figsize=(10, 10))
plt.scatter(data["_X"], data["_Y"], c="tab:gray", alpha=0.7, s=1, zorder=0)
plt.xlabel("latent dimension 1")
plt.ylabel("latent dimension 2")
plt.imshow(
    intersection, extent=data_extent, alpha=0.8, zorder=1, cmap="Greys", aspect="auto"
)
plt.tight_layout()
plt.title(
    f"{variable_name_1}, {variable_name_2} intersection - smoothing level {level_name}"
)
# plt.show()
plt.savefig(
    OUTDIR / f"{variable_1}_{variable_2}_intersection_spm_map.{FILETYPE}",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# %% plot3


variable_1 = "0006"
variable_2 = "0011"

variable = variable_1
variable_name = name_mapping[variable]
imtype = "beta"
level = 7
smoothing_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
level_name = smoothing_levels[level]

name = imtype + "_" + variable
img = nibabel.load(SPMINPUT / f"{name}.nii")
spm_map = img.get_fdata()
spm_mask = np.rot90(spm_map[:, :, level])

variable_name_1 = name_mapping[variable_1]
variable_name_2 = name_mapping[variable_2]
mask1 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_1}_mask.nii")
mask2 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_2}_mask.nii")
mask1_map = mask1.get_fdata()
mask1_map = np.rot90(mask1_map[:, :, level])
mask2_map = mask2.get_fdata()
mask2_map = np.rot90(mask2_map[:, :, level])

intersection = np.logical_and(mask1_map > 0, mask2_map > 0).astype(int)
plt.figure(figsize=(10, 10))
plt.scatter(data["_X"], data["_Y"], c="tab:gray", alpha=0.7, s=1, zorder=0)
plt.xlabel("latent dimension 1")
plt.ylabel("latent dimension 2")
plt.imshow(
    spm_mask,
    extent=data_extent,
    alpha=0.8,
    zorder=1,
    cmap="RdBu_r",
    norm=CenteredNorm(),
    aspect="auto",
)
plt.contour(
    np.flip(intersection, axis=0),
    extent=data_extent,
    levels=0,
    aspect="auto",
    linestyles="dashed",
    zorder=2,
)
plt.tight_layout()
plt.title(
    f"{variable_name_1}, {variable_name_2} intersection - smoothing level {level_name}"
)
# plt.show()
plt.savefig(
    OUTDIR / f"{variable_1}_{variable_2}_contour_intersection_spm_map.{FILETYPE}",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# %% jointplot
data_extent = (-2.0, 2.5, -3.6, 2.2)

variable_1 = "0001"
variable_2 = "0003"

variable = variable_1
variable_name = name_mapping[variable]
imtype = "beta"
level = 7
smoothing_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
level_name = smoothing_levels[level]

name = imtype + "_" + variable
img = nibabel.load(SPMINPUT / f"{name}.nii")
spm_map = img.get_fdata()
spm_mask = np.rot90(spm_map[:, :, level])

variable_name_1 = name_mapping[variable_1]
variable_name_2 = name_mapping[variable_2]
mask1 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_1}_mask.nii")
mask2 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_2}_mask.nii")
mask1_map = mask1.get_fdata()
mask1_map = np.rot90(mask1_map[:, :, level])
mask2_map = mask2.get_fdata()
mask2_map = np.rot90(mask2_map[:, :, level])

intersection = np.logical_and(mask1_map > 0, mask2_map > 0).astype(int)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
ax1.scatter(data["_X"], data["_Y"], c="tab:gray", alpha=0.7, s=1, zorder=0)
ax1.grid(True)
ax1.set_xlabel("latent dimension 1", fontsize=15)
ax1.set_ylabel("latent dimension 2", fontsize=15)
ax1.imshow(
    spm_mask,
    extent=data_extent,
    alpha=0.8,
    zorder=1,
    cmap="RdBu_r",
    norm=CenteredNorm(),
    aspect="auto",
)
ax1.contour(
    np.flip(intersection, axis=0),
    extent=data_extent,
    levels=0,
    aspect="auto",
    linestyles="dashed",
    zorder=2,
)

ax1.set_title(
    f"{variable_name_1}, {variable_name_2} intersection - smoothing level {level_name}",
    fontsize=15,
)


variable_1 = "0006"
variable_2 = "0011"

variable = variable_1
variable_name = name_mapping[variable]
imtype = "beta"
level = 7
smoothing_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
level_name = smoothing_levels[level]

name = imtype + "_" + variable
img = nibabel.load(SPMINPUT / f"{name}.nii")
spm_map = img.get_fdata()
spm_mask = np.rot90(spm_map[:, :, level])

variable_name_1 = name_mapping[variable_1]
variable_name_2 = name_mapping[variable_2]
mask1 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_1}_mask.nii")
mask2 = nibabel.load(SPMROOT / f"th_1/spmT_{variable_2}_mask.nii")
mask1_map = mask1.get_fdata()
mask1_map = np.rot90(mask1_map[:, :, level])
mask2_map = mask2.get_fdata()
mask2_map = np.rot90(mask2_map[:, :, level])

intersection = np.logical_and(mask1_map > 0, mask2_map > 0).astype(int)

ax2.scatter(data["_X"], data["_Y"], c="tab:gray", alpha=0.7, s=1, zorder=0)
ax2.grid(True)
ax2.set_xlabel("latent dimension 1", fontsize=15)
ax2.set_ylabel("latent dimension 2", fontsize=15)
im = ax2.imshow(
    spm_mask,
    extent=data_extent,
    alpha=0.8,
    zorder=1,
    cmap="RdBu_r",
    norm=CenteredNorm(),
    aspect="auto",
)
ax2.contour(
    np.flip(intersection, axis=0),
    extent=data_extent,
    levels=0,
    aspect="auto",
    linestyles="dashed",
    zorder=2,
)

ax2.set_title(
    f"{variable_name_1}, {variable_name_2} intersection - smoothing level {level_name}",
    fontsize=15,
)

# plt.colorbar(
#    im,ax=ax2,label="regression coefficient", shrink=1.0, pad=0.01, aspect=25, location="right"
# ).set_label(label="regression coefficient",size=15)

fig.colorbar(
    im,
    ax=[ax1, ax2],
    label="regression coefficient",
    shrink=1.0,
    pad=0.01,
    aspect=25,
    location="right",
).set_label(label="regression coefficient", size=15)

# plt.tight_layout()
# plt.show()
fig.savefig(
    OUTDIR / f"joint_contour_intersection_spm_map.{FILETYPE}",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
