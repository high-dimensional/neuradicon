"""Factories is a set of wrappers around the core classes of neuradicon.

These also import and wrap functions and classes from neurollm and neurocluster
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import spacy
import torch
from negspacy.negation import Negex
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from neuradicon.custom_pipes import ArtefactChecker, RelationExtractor
from neurocluster.predictors import AE
from neurocluster.vectorizers import EntityCountVectorizer, FeatureClusterer
from neurollm.utils import DocConverter, NeuroPipeline, Sectioner, minibatch

BATCH_SIZE = 32


class Model(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class ArtefactModel(Model):
    def __init__(self, model_loc):
        self.model = self.load_model(model_loc)
        self.artefact_checker = ArtefactChecker(self.model)

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        spacy_model = spacy.load(
            model_loc / "en_core_web_sm-3.0.0",
            exclude=["parser", "attribute_ruler", "lemmatizer", "ner"],
        )
        return spacy_model

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        nlp_docs = self.model.pipe(text_iter, batch_size=batch_size, n_process=n_procs)
        artefacts = [
            self.artefact_checker(d)._.has_artefact
            for d in tqdm(nlp_docs, total=len(text_iter))
        ]
        return artefacts


class SectionModel(Model):
    def __init__(self, model_loc):
        self.model = self.load_model(model_loc)

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        tokenizer_name = model_loc / "basemodel"
        model_name = model_loc / "segmentation"
        sectioner = Sectioner(model_name, tokenizer_name, device="cuda:0")
        return sectioner

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        self.model.batch_size = batch_size
        return [s for s in tqdm(self.model(text_iter), total=len(text_iter))]


class NormalityModel(Model):
    def __init__(self, model_loc):
        self.model = self.load_model(model_loc)

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        tokenizer_name = model_loc / "basemodel"
        cls_name = model_loc / "cls"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(cls_name)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cuda:0"),
        )
        return classifier

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
        output = [
            i["label"]
            for batch in tqdm(
                minibatch(text_iter, n=BATCH_SIZE), total=len(text_iter) // BATCH_SIZE
            )
            for i in self.model(batch, **tokenizer_kwargs)
        ]
        return output


DOMAIN_LABELS = [
    "location_arteries",
    "location_brain_stem",
    "location_diencephalon",
    "location_ent",
    "location_eye",
    "location_ganglia",
    "location_grey_matter",
    "location_limbic_system",
    "location_meninges",
    "location_nerves",
    "location_neurosecretory_system",
    "location_other",
    "location_skull",
    "location_spine",
    "location_telencephalon",
    "location_veins",
    "location_ventricles",
    "location_white_matter",
    "location_qualifier",
    "descriptor_cyst",
    "descriptor_damage",
    "descriptor_diffusion",
    "descriptor_signal_change",
    "descriptor_enhancement",
    "descriptor_flow",
    "descriptor_interval_change",
    "descriptor_mass_effect",
    "descriptor_morphology",
    "descriptor_collection",
    "pathology_haemorrhagic",
    "pathology_ischaemic",
    "pathology_vascular",
    "pathology_cerebrovascular",
    "pathology_treatment",
    "pathology_inflammatory_autoimmune",
    "pathology_congenital_developmental",
    "pathology_csf_disorders",
    "pathology_musculoskeletal",
    "pathology_neoplastic_paraneoplastic",
    "pathology_infectious",
    "pathology_neurodegenerative_dementia",
    "pathology_metabolic_nutritional_toxic",
    "pathology_endocrine",
    "pathology_opthalmological",
    "pathology_traumatic",
    "descriptor_necrosis",
]

DESCRIPTOR_LABELS = [
    "descriptor_cyst",
    "descriptor_damage",
    "descriptor_diffusion",
    "descriptor_signal_change",
    "descriptor_enhancement",
    "descriptor_flow",
    "descriptor_interval_change",
    "descriptor_mass_effect",
    "descriptor_morphology",
    "descriptor_collection",
    "descriptor_necrosis",
]

PATHOLOGY_LABELS = [
    "pathology_haemorrhagic",
    "pathology_ischaemic",
    "pathology_vascular",
    "pathology_cerebrovascular",
    "pathology_treatment",
    "pathology_inflammatory_autoimmune",
    "pathology_congenital_developmental",
    "pathology_csf_disorders",
    "pathology_musculoskeletal",
    "pathology_neoplastic_paraneoplastic",
    "pathology_infectious",
    "pathology_neurodegenerative_dementia",
    "pathology_metabolic_nutritional_toxic",
    "pathology_endocrine",
    "pathology_opthalmological",
    "pathology_traumatic",
]

ENTITY_LABELS = PATHOLOGY_LABELS + DESCRIPTOR_LABELS


class ClinicalEntityModel:
    def __init__(self, model_loc, link_locations=False):
        self.model = self.load_model(model_loc)
        self.link_locations = link_locations
        if link_locations:
            self.converter, self.relex = self.load_relex(model_loc)

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        tokenizer_name = model_loc / "basemodel"
        ner_name = model_loc / "ner-fine-tuned"
        neg_name = model_loc / "negation"
        pipeline = NeuroPipeline(
            ner_name,
            neg_name,
            tokenizer_name,
            aggregation_strategy="first",
            device="cuda:0",
            batch_size=BATCH_SIZE,
        )
        return pipeline

    def load_relex(self, model_loc):
        model_loc = Path(model_loc)
        converter = DocConverter(model_loc / "en_core_web_sm")
        relex = RelationExtractor(converter.nlp)
        relex.from_disk(model_loc / "relex_small")
        return converter, relex

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        doc_domains = list(
            tqdm(self.model(text_iter), total=len(text_iter) // BATCH_SIZE)
        )
        if self.link_locations:
            doc_domains = self.get_locations(doc_domains)
        return doc_domains

    def get_locations(self, dict_iter):
        new_text_iter = []
        for t in tqdm(dict_iter, total=len(dict_iter)):
            doc = self.converter(t)
            new_doc = self.relex(doc)
            links = []
            for e in new_doc.ents:
                if e.label_ in ENTITY_LABELS:
                    for r in e._.relation:
                        links.append((e.text, r.text))
            with_links = t
            t["links"] = links
            new_text_iter.append(with_links)
        return new_text_iter


class EmbeddingModel(Model):
    def __init__(self, model_loc):
        self.model, self.vectorizer, self.clusterer = self.load_model(model_loc)
        self.nlp = spacy.load(
            Path(model_loc)
            / "en_full_neuro_model-1.8/en_full_neuro_model/en_full_neuro_model-1.8"
        )

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        embed_model = AE([1954, 256, 64, 2], sigmoid_output=True)
        embed_model.load_state_dict(torch.load(model_loc / "ae_model/model.bin"))
        embed_model.eval()
        vectorizer = EntityCountVectorizer()
        vectorizer.load_from_pickle(model_loc / "ae_model")
        clusterer = FeatureClusterer(2)
        clusterer.load_from_pickle(model_loc / "ae_model")
        return embed_model, vectorizer, clusterer

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        docs = list(
            tqdm(
                self.nlp.pipe(text_iter, batch_size=batch_size, n_process=20),
                total=len(text_iter),
            )
        )
        vectors = self.vectorize_data(docs, merge=True)
        with torch.no_grad():
            test_input = torch.from_numpy(vectors).float()
            embedded_vectors = self.model.encode(test_input).detach().numpy()
        return embedded_vectors

    def vectorize_data(self, docs, merge=False):
        vecs = np.array(self.vectorizer.transform(docs))
        agglo_vecs = self.clusterer.transform(vecs)
        if merge:
            agglo_vecs = np.concatenate((vecs, agglo_vecs), axis=1)
        return agglo_vecs


class SignalModel:
    def __init__(self, model_loc):
        self.model = self.load_model(model_loc)

    def __call__(self, text_iter, batch_size=128, n_procs=1):
        docs = self.model.pipe(text_iter, n_process=n_procs, batch_size=batch_size)
        transformed_data = [
            self.has_signals(doc) for doc in tqdm(docs, total=len(text_iter))
        ]
        return transformed_data

    def has_signals(self, doc):
        signals = ["high-signal", "low-signal", "unk-signal"]
        modalities = ["T1", "T2", "DWI", "FLAIR", "unk-modality"]
        assertion = ["asserted", "denied"]
        classes = {
            j + "-" + k + "-" + i: 0
            for i in signals
            for k in assertion
            for j in modalities
        }

        for s in doc.sents:
            sent_signals = [
                "denied-" + e.label_ if e._.is_negated else "asserted-" + e.label_
                for e in s.ents
                if e.label_ in signals
            ]
            sent_modalities = [e.label_ for e in s.ents if e.label_ in modalities]
            for a in sent_signals:
                if not sent_modalities:
                    classes["unk-modality-" + a] += 1
                for b in sent_modalities:
                    key = b + "-" + a
                    classes[key] += 1
        return classes

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        model = spacy.load(model_loc / "en_core_web_sm", exclude=["ner"])
        ruler = model.add_pipe(
            "entity_ruler", config={"phrase_matcher_attr": "LOWER"}
        ).from_disk(model_loc / "signal_patterns.jsonl")
        negex = model.add_pipe("negation").from_disk(
            model_loc / "negation_patterns_signal"
        )
        return model
