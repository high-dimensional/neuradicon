"""Custom spacy pipes and pipeline utilties for neuradicon."""

from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import spacy
import srsly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from spacy.language import Language
from spacy.matcher import DependencyMatcher, Matcher, PhraseMatcher
from spacy.scorer import PRFScore, Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import ensure_path, from_disk, to_disk
from tqdm import tqdm


@Language.factory(
    "relex",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def create_relex_component(nlp: Language, name: str):
    """relation extraction pipeline component factory"""
    return RelationExtractor(nlp, name)


class RelationExtractor:
    """Pipe assigning relations between concepts and their corresponding locations.

    PATHOLOGY and DESCRIPTOR concepts tagged by an entity recogniser
    pipe are connected to their corresponding LOCATIONS by using a
    grammatical tree-traversal algorithm.

    The relations for each entity are stored in the Span.relation attribute,
    where the Span entity objects can be accessed via the Doc.ents generator.
    """

    def __init__(self, nlp, name="relex"):
        self.name = name
        self.patterns = {}
        self.matcher = DependencyMatcher(nlp.vocab)
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default=[])
        if not Span.has_extension("relation"):
            Span.set_extension("relation", getter=self.relation_getter)

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        doc._.rel.extend(
            list(set([(token_ids[0], token_ids[-1]) for _, token_ids in matches]))
        )
        return doc

    def relation_getter(self, span):
        """given an entity span, find all relations attached to it"""
        span_rel_ids = set(
            [rels[-1] for w in span for rels in span.doc._.rel if w.i == rels[0]]
        )
        spans = []
        for e in span.doc.ents:
            for w in e:
                if w.i in span_rel_ids:
                    spans.append(e)
                    break
        return spans

    def add_patterns(self, pattern_dict):
        for name, pattern in pattern_dict.items():
            self.matcher.add(name, [pattern])

    def to_disk(self, path, exclude=tuple()):
        serializers = {
            "patterns": lambda p: srsly.write_json(
                p.with_suffix(".json"), self.patterns
            )
        }
        to_disk(path, serializers, {})

    def from_disk(self, path, exclude=tuple()):
        self.patterns = srsly.read_json(path / "patterns.json")

        deserializers_patterns = {
            "patterns": lambda p: self.add_patterns(
                srsly.read_json(p.with_suffix(".json"))
            )
        }
        from_disk(path, deserializers_patterns, {})
        return self

    def initialize(self, get_examples=None, nlp=None, data={}):
        self.patterns = data

    def score(self, examples: Iterable[Example]) -> Dict[str, Any]:
        prf = PRFScore()
        for example in examples:
            # gold = set(example.reference._.rel)
            # pred = set(example.predicted._.rel)

            # n_tp = len(gold.intersection(pred))
            # n_fp = len(pred.difference(gold))
            # n_fn = len(gold.difference(pred))

            # prf.tp += n_tp
            # prf.fp += n_fp
            # prf.fn += n_fn

            for ref_e, pred_e in zip(example.reference.ents, example.predicted.ents):
                if ref_e.label_ in ["PATHOLOGY", "DESCRIPTOR"]:
                    gold_relations = set(
                        [(grel.start, grel.end) for grel in ref_e._.relation]
                    )
                    pred_relations = set(
                        [(prel.start, prel.end) for prel in pred_e._.relation]
                    )
                    n_tp = len(gold_relations.intersection(pred_relations))
                    n_fp = len(pred_relations.difference(gold_relations))
                    n_fn = len(gold_relations.difference(pred_relations))

                    prf.tp += n_tp
                    prf.fp += n_fp
                    prf.fn += n_fn

        return {
            "rel_micro_p": prf.precision,
            "rel_micro_r": prf.recall,
            "rel_micro_f": prf.fscore,
        }


@Language.factory(
    "negation",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.neg"],
    default_score_weights={
        "neg_micro_p": None,
        "neg_micro_r": None,
        "neg_micro_f": None,
    },
)
def create_negation_component(nlp: Language, name: str):
    return NegationDetector(nlp, name)


class NegationDetector:
    """Detects negation of entities.

    A spaCy pipeline component which identifies negated tokens in text.
    Based on: NegEx - A Simple Algorithm for Identifying Negated
    Findings and Diseases in Discharge Summaries Chapman, Bridewell,
    Hanbury, Cooper, Buchanan. This pipe uses a set of negation patterns.
    """

    def __init__(self, nlp, name="negation"):
        self.name = name
        self.patterns = {}
        self.matcher = DependencyMatcher(nlp.vocab)
        if not Span.has_extension("is_negated"):
            Span.set_extension("is_negated", getter=self.is_negated_getter)
        if not Doc.has_extension("neg"):
            Doc.set_extension("neg", default=[])

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        neg_ids = [token_ids[0] for _, token_ids in matches]
        doc._.neg.extend(list(set(neg_ids)))
        return doc

    def is_negated_getter(self, span):
        return True if any([w.i in span.doc._.neg for w in span]) else False

    def add_patterns(self, pattern_dict):
        for name, pattern in pattern_dict.items():
            self.matcher.add(name, [pattern])

    def to_disk(self, path, exclude=tuple()):
        serializers = {
            "patterns": lambda p: srsly.write_json(
                p.with_suffix(".json"), self.patterns
            )
        }
        to_disk(path, serializers, {})

    def from_disk(self, path, exclude=tuple()):
        self.patterns = srsly.read_json(path / "patterns.json")

        deserializers_patterns = {
            "patterns": lambda p: self.add_patterns(
                srsly.read_json(p.with_suffix(".json"))
            )
        }
        from_disk(path, deserializers_patterns, {})
        return self

    def initialize(self, get_examples=None, nlp=None, data={}):
        self.patterns = data

    def score(self, examples: Iterable[Example]) -> Dict[str, Any]:
        prf = PRFScore()
        for example in examples:
            for ref_e, pred_e in zip(example.reference.ents, example.predicted.ents):
                if ref_e._.is_negated and pred_e._.is_negated:
                    prf.tp += 1
                elif (not ref_e._.is_negated) and pred_e._.is_negated:
                    prf.fp += 1
                elif ref_e._.is_negated and (not pred_e._.is_negated):
                    prf.fn += 1

        return {
            "neg_micro_p": prf.precision,
            "neg_micro_r": prf.recall,
            "neg_micro_f": prf.fscore,
        }


@Language.factory(
    "domain_detector",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.domains"],
)
def create_domain_component(nlp: Language, name: str, method: str):
    return DomainDetector(nlp, name, method)


class DomainDetector:
    """Detects pathological domain of ENT-tagged report"""

    def __init__(
        self,
        nlp,
        name="domainer",
        method="matcher",
        class_types=["PATHOLOGY"],
        use_negated=False,
    ):
        self.name = name
        self.patterns = {}
        self.nlp = nlp
        self.class_types = class_types
        self.use_negated = use_negated

        if method in ["matcher", "char-cls", "vec-cls"]:
            self.method = method
        else:
            raise Exception(
                "Method not recognized, use either matcher or char-cls or vec-cls"
            )
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.vectorizer = TfidfVectorizer(
            lowercase=True, analyzer="char", ngram_range=(2, 3), max_features=1500
        )
        self.knn_cls = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        self.label_encoder = LabelEncoder()
        if not Doc.has_extension("domains"):
            Doc.set_extension("domains", default=[])
        if not Span.has_extension("domain"):
            Span.set_extension("domain", default=None)

    def __call__(self, doc: Doc) -> Doc:
        for e in doc.ents:
            if e.label_ in self.class_types:
                if (not self.use_negated) and e._.is_negated:
                    continue
                else:
                    match_name = None
                    if self.method == "matcher":
                        for match_id, start, end in self.matcher(e.as_doc()):
                            match_name = doc.vocab.strings[match_id]
                    elif self.method == "char-cls":
                        entity_vector = self.vectorizer.transform([e.text])
                        prediction = self.knn_cls.predict(entity_vector)
                        match_name = self.label_encoder.inverse_transform(prediction)
                        match_name = match_name[0]
                    elif self.method == "vec-cls":
                        entity_vector = e.vector.reshape(1, 200)
                        prediction = self.knn_cls.predict(entity_vector)
                        match_name = self.label_encoder.inverse_transform(prediction)
                        match_name = match_name[0]

                    e._.domain = match_name

        doc_matches = list(set([e._.domain for e in doc.ents if e._.domain]))
        doc._.domains = doc_matches
        return doc

    def add_patterns(self, pattern_dict):
        for name, patterns in pattern_dict.items():
            doc_patterns = [self.nlp.make_doc(i) for i in patterns]
            self.matcher.add(name, doc_patterns)

    def to_disk(self, path, exclude=tuple()):
        serializers = {
            "patterns": lambda p: srsly.write_json(
                p.with_suffix(".json"), self.patterns
            )
        }
        to_disk(path, serializers, {})

    def from_disk(self, path, exclude=tuple()):
        path = Path(path)
        self.patterns = srsly.read_json(path / "patterns.json")

        deserializers_patterns = {
            "patterns": lambda p: self.add_patterns(
                srsly.read_json(p.with_suffix(".json"))
            )
        }
        from_disk(path, deserializers_patterns, {})

        if self.method == "char-cls":
            self.initialize_classifer(self.patterns)
        elif self.method == "vec-cls":
            self.initialize_vec_classifier(self.patterns)

        return self

    def initialize_classifer(self, pattern_dict):
        pattern_labels, pattern_strings = zip(
            *[(key, v) for key, val in pattern_dict.items() for v in val]
        )
        pattern_labels, pattern_strings = list(pattern_labels), list(pattern_strings)
        pattern_label_codes = self.label_encoder.fit_transform(pattern_labels)
        pattern_vectors = self.vectorizer.fit_transform(pattern_strings)
        self.knn_cls.fit(pattern_vectors, pattern_label_codes)

    def initialize_vec_classifier(self, pattern_dict):
        pattern_labels, pattern_strings = zip(
            *[(key, v) for key, val in pattern_dict.items() for v in val]
        )
        pattern_labels, pattern_strings = list(pattern_labels), list(pattern_strings)
        pattern_label_codes = self.label_encoder.fit_transform(pattern_labels)
        pattern_vectors = np.array([self.nlp(t).vector for t in pattern_strings])
        self.knn_cls.fit(pattern_vectors, pattern_label_codes)

    def initialize(self, get_examples=None, nlp=None, data={}):
        self.patterns = data


## Custom Entity linker removed 28/07/2022


class ArtefactChecker:
    def __init__(self, nlp):
        self.matcher = Matcher(nlp.vocab)
        if not Doc.has_extension("has_artefact"):
            Doc.set_extension("has_artefact", default=False)
        self.patterns = [
            [{"LOWER": "artefact"}],
            [{"LOWER": "artifactual"}],
            [{"LOWER": "artifact"}],
            [{"LOWER": "artefacts"}],
            [{"LOWER": "artifacts"}],
            [{"LOWER": "aliasing"}],
            [{"LOWER": "degraded"}],
            [{"LOWER": "degradation"}],
            [{"LOWER": "blooming"}],
        ]
        self.matcher.add("artefacts", self.patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        if matches:
            doc._.has_artefact = True
        return doc


class SpacySectioner:
    def __init__(self, parser, section_cls):
        self.parser = spacy.load(parser)  # , exclude=["ner", "tagger", "lemmatizer"])
        self.section_cls = spacy.load(section_cls)

    def __call__(
        self, text_iter, use_tqdm=False, batch_size=512, partition_size=1000, n_procs=1
    ):
        text_sections = []
        length = len(text_iter)

        parser_docs = self.parser.pipe(
            text_iter, batch_size=batch_size, n_process=n_procs
        )
        sentence_generator = [
            (i, s.text)
            for i, doc in tqdm(enumerate(parser_docs), total=length)
            for s in doc.sents
        ]
        _, sentences = zip(*sentence_generator)
        classified_sentence_docs = self.section_cls.pipe(
            sentences, batch_size=batch_size, n_process=n_procs
        )
        classified_sentences = (
            (i, t, max(s.cats, key=s.cats.get))
            for (i, t), s in zip(sentence_generator, classified_sentence_docs)
        )
        for i, g in tqdm(
            groupby(classified_sentences, key=lambda f: f[0]), total=length
        ):
            section_dict = {}
            for a, b, c in g:
                if c not in section_dict.keys():
                    section_dict[c] = [b]
                else:
                    section_dict[c].append(b)
            section_dict = {key: " ".join(vals) for key, vals in section_dict.items()}

            text_sections.append(section_dict)
        return text_sections


## Normality ruler removed 28/07/2022


def extract_domains(doc, use_negated=False):
    """Extract the entity domains present in a spacy doc.
    For use with the updated neuradicon NER model"""
    domains = {e.label_ for e in doc.ents if ((not e._.is_negated) or use_negated)}
    return sorted(list(domains))


def remove_commentary(doc, classes_to_remove=["location-spine", "location-ent"]):
    """Remove sentences that include commentary about particular locations or pathologies.
    Primary application is for the removal of commentary about the spine.
    """
    sentences_to_keep = [
        s.text
        for s in doc.sents
        if not any([e.label_ in classes_to_remove for e in s.ents])
    ]
    return " ".join(sentences_to_keep)
