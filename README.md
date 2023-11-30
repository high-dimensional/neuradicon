# neuradicon

A natural language processing pipeline for neuroradiological reports
-----

neuradicon is a set of models for extracting clinical information from neuroradiological reports, and deriving clinically-meaning representations of the text of the report as a whole.
It currently provides models for the following tasks:
- Artefact detection
- Section segmentation
- Clinical entity extraction
- Radiological signal detection
- Report type classification
- Clinical text representation

## Installation

The neuradicon project depends on two other reporting projects, `neurollm` and `neurocluster`. These must be installed before installing neuradicon. Their repositories can be found in the reporting project group.
To install neuradicon, clone this repository and run pip from the repository root.
```console
pip install -e .
```
Alternatively, an easier means of using the software is to pull the accompanying docker image, containing both models and the environment required to run neuradicon.

## Usage
You can find an example script for using neuradicon in the `examples` directory.


