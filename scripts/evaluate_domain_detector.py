#!/usr/bin/env python
"""Evaluate domain detector pipe.

Calculate the multilabel performance metrics for the domain detector model.
"""

import argparse
from pathlib import Path

import numpy as np
import spacy
import srsly
from neuroNLP.custom_pipes import *
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def multilabel_specificity(y_true, y_pred, classes):
    mlcm = multilabel_confusion_matrix(y_true, y_pred)
    tn, fn, tp, fp = mlcm[:, 0, 0], mlcm[:, 1, 0], mlcm[:, 1, 1], mlcm[:, 0, 1]
    specifics = tn / (tn + fp)
    output = {i: j for i, j in zip(classes, specifics)}
    output["micro avg"] = tn.sum() / (tn.sum() + fp.sum())
    output["macro avg"] = specifics.mean()
    support = y_true.sum(0)
    output["weighted avg"] = (specifics * support).sum() / support.sum()
    return output


def output_to_file(
    file, conf_mat, model, binarizer, gold_labels, true_labels, pred_labels
):
    file.write("NER-based performance\n")

    classes_to_keep = [
        "pathology-cerebrovascular",
        "pathology-congenital-developmental",
        "pathology-csf-disorders",
        "pathology-endocrine",
        "pathology-haemorrhagic",
        "pathology-infectious",
        "pathology-inflammatory-autoimmune",
        "pathology-ischaemic",
        "pathology-metabolic-nutritional-toxic",
        # "pathology-musculoskeletal",
        "pathology-neoplastic-paraneoplastic",
        "pathology-neurodegenerative-dementia",
        "pathology-opthalmological",
        "pathology-traumatic",
        "pathology-treatment",
        "pathology-vascular",
    ]

    for i, name in enumerate(binarizer.classes_):
        file.write(name + "\n")
        file.write(str(conf_mat[i]) + "\n")

    for i, v in enumerate(binarizer.classes_):
        idx_fp = np.logical_and(true_labels[:, i] == 0, pred_labels[:, i] == 1)
        idx_fn = np.logical_and(true_labels[:, i] == 1, pred_labels[:, i] == 0)

        fp_docs = [d for d, i in zip(gold_labels, idx_fp) if i]
        file.write(f"\nFalse positives for {v}\n")
        for t in fp_docs:
            file.write(t["text"])
            file.write("\n")
            doc = model(t["text"])
            file.write(
                "ents: "
                + str(
                    [
                        (e.text, e.label_)
                        for e in doc.ents
                        if e.label_ in classes_to_keep
                    ]
                )
            )
            file.write("\n")
            file.write("\n")

        fn_docs = [d for d, i in zip(gold_labels, idx_fn) if i]
        file.write(f"\nFalse negatives for {v}\n")
        for t in fn_docs:
            file.write(t["text"])
            file.write("\n")
            doc = model(t["text"])
            file.write(
                "ents: "
                + str(
                    [
                        (e.text, e.label_)
                        for e in doc.ents
                        if e.label_ in classes_to_keep
                    ]
                )
            )
            file.write("\n")
            file.write("\n")


def load_input(arguments):
    """load in data or take input from stdin"""
    full_nlp = spacy.load(arguments.model)
    gold_path_labels = [
        i for i in srsly.read_jsonl(arguments.data) if i["answer"] == "accept"
    ]
    return gold_path_labels, full_nlp


def transform_data(data, model):
    """perform the necessary transformation on the input data"""
    gold_path_domains = [sorted(list(set(d["accept"]))) for d in data]
    full_neuro_pred_path_domains = [
        extract_domains(model(d["text"])) for d in tqdm(data)
    ]
    path_bina = MultiLabelBinarizer()
    path_true = path_bina.fit_transform(gold_path_domains)
    path_neuro_pred = path_bina.transform(full_neuro_pred_path_domains)
    return path_bina, path_true, path_neuro_pred


def analysis(binarizer, true_label, pred_label):
    """perform analysis on data"""
    mtlconf = multilabel_confusion_matrix(true_label, pred_label)
    output = classification_report(
        true_label,
        pred_label,
        target_names=binarizer.classes_,
        zero_division=0,
        output_dict=True,
    )
    specifics = multilabel_specificity(true_label, pred_label, binarizer.classes_)
    for i, label in enumerate(output.keys()):
        if label in specifics.keys():
            output[label]["specificity"] = specifics[label]
            output[label]["sensitivity"] = output[label]["recall"]
    return mtlconf, output


def output_results(
    args,
    confusion_matrix,
    model,
    output,
    binarizer,
    gold_labels,
    true_labels,
    pred_labels,
):
    """output analysis, save to file or send to stdout"""
    model_name = args.model.stem
    with open(args.outdir / f"domain-perf-{model_name}.txt", "w") as f:
        output_to_file(
            f, confusion_matrix, model, binarizer, gold_labels, true_labels, pred_labels
        )
    srsly.write_json(args.outdir / f"domain-perf-{model_name}.json", output)


def main():
    parser = argparse.ArgumentParser(prog=__file__, description=__doc__)
    parser.add_argument("model", help="path to domain detector model", type=Path)
    parser.add_argument("data", help="path to labelled evaluation data", type=Path)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    data, model = load_input(args)
    binarizer, true_labels, pred_labels = transform_data(data, model)
    conf_mat, output = analysis(binarizer, true_labels, pred_labels)
    output_results(
        args, conf_mat, model, output, binarizer, data, true_labels, pred_labels
    )


if __name__ == "__main__":
    main()
