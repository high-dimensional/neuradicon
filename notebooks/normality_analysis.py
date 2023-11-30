#!/usr/bin/env python
"""Normality analysis script.

This script produces histograms for the normality class counts. It also produces histograms for the
number of longitudinal scans per patient, stratified by pathology class
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import nan

sns.set()


def load_input(location):
    """load in data or take input from stdin"""
    df = pd.read_csv(location, low_memory=False, parse_dates=["End Exam Date"])
    return df


def merge_norm_classes(df):
    df["norm_class"] = nan
    df.loc[df["IS_MISSING"].fillna(False), "norm_class"] = "IS_MISSING"
    df.loc[df["IS_COMPARATIVE"].fillna(False), "norm_class"] = "IS_COMPARATIVE"
    df.loc[df["IS_NORMAL"].fillna(False), "norm_class"] = "IS_NORMAL"
    df.loc[df["IS_NORMAL_FOR_AGE"].fillna(False), "norm_class"] = "IS_NORMAL_FOR_AGE"
    return df


def transform_data(data):
    """perform the necessary transformation on the input data"""
    # binarize pathologies
    p_cols = [p for p in data.columns if "asserted-path" in p]
    for i in p_cols:
        data.loc[:, "has-" + i[18:]] = data.loc[:, i].astype(bool)

    # merge norm classes
    data = merge_norm_classes(data)
    return data


def aggregate_patient_data(df):
    """create patient-level df"""
    aggregation_dict = {
        "Patient Sex": lambda x: x.mode(),
        "Patient Firstname": lambda x: x.mode(),
        "Patient Surname": lambda x: x.mode(),
        "Patient DOB": lambda x: x.mode(),
        "Ethnic Origin": lambda x: x.mode(),
        "Patient Postcode": lambda x: x.mode(),
    }
    p_cols = [p for p in df.columns if "has-" in p]
    for p in p_cols:
        aggregation_dict[p] = lambda x: x.any()

    patient_df = df.groupby("MRN").agg(aggregation_dict)
    patient_df["n_reports"] = df.groupby("MRN").size()
    patient_df["first_report_date"] = df.groupby("MRN")["End Exam Date"].min()
    patient_df["last_report_date"] = df.groupby("MRN")["End Exam Date"].max()
    patient_df["reporting_period"] = (
        patient_df["last_report_date"] - patient_df["first_report_date"]
    )
    return patient_df


def plot_result(df, patient_df):
    """plot the results"""
    figs = {}
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="norm_class", ax=ax)
    figs["norm_class"] = fig
    fig2, ax2 = plt.subplots()
    sns.histplot(data=patient_df, x="n_reports", ax=ax2, binwidth=1, binrange=(0, 20))
    figs["n_reports"] = fig2
    # fig3, ax3 = plt.subplots()
    # int_bins = patient_df['reporting_period'].max()// pd.Timedelta("1s")
    # sns.histplot(data=patient_df, x='reporting_period', ax=ax3, bins=int_bins)
    # ax3.set_xticks(int_bins, labels=[b.strftime("%Y-%m-%d") for b in date_bins])
    # figs['reporting_period'] = fig3
    for i in [p for p in patient_df.columns if "has-" in p]:
        fig4, ax4 = plt.subplots()
        subset = patient_df[patient_df[i]]
        sns.histplot(data=subset, x="n_reports", ax=ax4, binwidth=1, binrange=(0, 20))
        figs[i] = fig4
    return figs


def output_results(plots_figs, outdir):
    """output analysis, save to file or send to stdout"""
    for name, f in plots_figs.items():
        f.savefig(outdir / f"{name}-plot.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report CSV", type=Path)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()
    data = load_input(args.input)
    transformed_data = transform_data(data)
    patient_data = aggregate_patient_data(transformed_data)
    plots = plot_result(transformed_data, patient_data)
    output_results(plots, args.outdir)


if __name__ == "__main__":
    main()
