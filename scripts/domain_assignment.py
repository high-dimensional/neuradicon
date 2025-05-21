import time
from pathlib import Path

import pandas as pd
import spacy
from neuroNLP.custom_pipes import *
from spacy.tokens import DocBin
from tqdm import tqdm

# %%

ontologies = Path("DATAPATH/Desktop/neuroOnto/ontologies")
chosen_onto = ontologies / "mesh"
cons = chosen_onto / "ontology_concepts.csv"
rels = chosen_onto / "ontology_relations.csv"
onto = Ontology(cons, rels)

data_dir = Path("DATAPATH/Desktop/neuroData/170k_spacy_docs_bodies_new_format")
nlp_dir = "DATAPATH/Desktop/neuroNLP/training/rel_neg_detection_tok2vec_v2"

print("loading pipeline")
nlp = spacy.load(nlp_dir)
print("\nFull pipeline: ", nlp.pipe_names)

# %%
CLASS2COUNT = "DESCRIPTOR"
# %%
print("preparing structured docs")
start_time = time.time()
doc_bin = DocBin().from_disk(data_dir / "neuronlp_docs.spacy")
all_docs = list(doc_bin.get_docs(nlp.vocab))
end_time = time.time()
print("structured doc execution time: {} seconds".format(end_time - start_time))

# %%
PATTERN_DIR = Path("DATAPATH/Desktop/neuroData/domain_patterns/descriptor_patterns_v2")

detector = DomainDetector(
    nlp, class_types=[CLASS2COUNT], method="matcher", use_negated=True
)
detector.from_disk(PATTERN_DIR)

# %%

# %%
matched_docs = [detector(doc) for doc in tqdm(all_docs)]
# %%


def tp_condition_ent(e, label):
    if e.label_ == label:
        if e._.domain:
            return True
    return False


def fp_condition_ent(e, label):
    if not (e.label_ == label):
        if e._.domain:
            return True
    return False


def tn_condition_ent(e, label):
    if not (e.label_ == label):
        if not e._.domain:
            return True
    return False


def fn_condition_ent(e, label):
    if e.label_ == label:
        if not e._.domain:
            return True
    return False


# %%
fn = len(
    [e for doc in matched_docs for e in doc.ents if fn_condition_ent(e, CLASS2COUNT)]
)
fp = len(
    [e for doc in matched_docs for e in doc.ents if fp_condition_ent(e, CLASS2COUNT)]
)
tn = len(
    [e for doc in matched_docs for e in doc.ents if tn_condition_ent(e, CLASS2COUNT)]
)
tp = len(
    [e for doc in matched_docs for e in doc.ents if tp_condition_ent(e, CLASS2COUNT)]
)

print("coverage")
print("fn: ", fn)
print("fp: ", fp)
print("tn: ", tn)
print("tp: ", tp)
print("total: ", fn + fp + tn + tp)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
print("precision: ", prec)
print("recall: ", rec)
print("f1: ", 2 * prec * rec / (prec + rec))
# %%
# %%
orphaned_paths = [
    e for doc in matched_docs for e in doc.ents if fn_condition_ent(e, CLASS2COUNT)
]
orphaned_paths_strs = [e.text.lower() for e in orphaned_paths]  # if not e._.is_negated]
asdf = pd.Series(orphaned_paths_strs)
valcounts = asdf.value_counts()
print("most common orphans")
print(valcounts[:50])

valcounts[:200].to_csv("descriptor_orphans.csv")
# %%
example_doc = matched_docs[389]
for e in example_doc.ents:
    print(e.text, e._.domain)
# %%
"""
normal_cls = spacy.load("./training/binary_normal_cls_v2/model-best")
compar_cls = spacy.load("./training/comparative_cls/model-best")
#%%
texts = [d.text for d in all_docs]
#%%
norm_docs = [d.cats["NORMAL"]>0.5 for d in tqdm(normal_cls.pipe(texts, n_process=16, batch_size=128))]
#%%
norm_docv2 = [for d in matched_docs for e in d.ents if]
#%%
print(compar_cls(example_doc.text).cats)
#%%
comp_docs = [d.cats["COMPARITIVE"]>0.5 for d in tqdm(compar_cls.pipe(texts, n_process=16, batch_size=128))]
#%%
abnormal_noncompare = [(not i) and (not j) for i,j in zip(norm_docs, comp_docs)]
#%%
to_count = [d for d, i in zip(matched_docs, abnormal_noncompare) if i]
#%%
len([d for d in to_count if d._.domains])/len(to_count)

#%%
missing_domains = [d for d in to_count if not d._.domains]
#%%
print(missing_domains[9599])
"""
