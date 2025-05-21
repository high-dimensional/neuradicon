#!/usr/bin/env python
# coding: utf-8

# # Anatomical domain assigment

# In[1]:


import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from neuroNLP.custom_pipes import *
from neuroNLP.ontology import (CandidateGenerator, Ontology,
                               find_domain_patterns, get_decendant_cuis)
from spacy.tokens import DocBin
from tqdm import tqdm

# In[2]:


# In[3]:


# ## Load ontology

# In[5]:


ontologies = Path("DATAPATH/Desktop/neuroOnto/ontologies")
chosen_onto = ontologies / "mesh"
cons = chosen_onto / "ontology_concepts.csv"
rels = chosen_onto / "ontology_relations.csv"


# In[6]:


onto = Ontology(cons, rels)


# In[7]:


root = onto.get_root()


# In[8]:


root.children


# In[9]:


top_level_domains = [
    "D005123",
    "D009490",
    "D001158",
    "D014680",
    "D005724",
    "D012886",
    "D008578",
    "D001933",
    "D002552",
    "D066128",
    "D066127",
    "D008032",
    "D013687",
    "D004027",
    "D003391",
]


# ## Load data

# In[10]:


data_dir = Path("DATAPATH/Desktop/neuroData/170k_spacy_docs_bodies_new_format")
nlp_dir = "/training/rel_neg_detection_tok2vec_v2"


# In[11]:


print("loading pipeline")
nlp = spacy.load(nlp_dir)
print("\nFull pipeline: ", nlp.pipe_names)


# In[12]:


print("preparing structured docs")
start_time = time.time()
doc_bin = DocBin().from_disk(data_dir / "neuronlp_docs.spacy")
all_docs = list(doc_bin.get_docs(nlp.vocab))
end_time = time.time()
print("structured doc execution time: {} seconds".format(end_time - start_time))


# ## Get domain patterns

# In[13]:


custom_domain_concepts = [onto[cui] for cui in top_level_domains]


# In[15]:


domain_cuis = [c.cui for c in custom_domain_concepts]


# In[16]:


domain_dict = find_domain_patterns(domain_cuis, onto)


# In[18]:


PATTERN_DIR = Path("DATAPATH/Desktop/neuroData/domain_patterns/mesh_anat_patterns")


# In[20]:


with open(PATTERN_DIR / "patterns.json", "w") as file:
    json.dump(domain_dict, file, indent=4)


# ## Test pipe

# In[78]:


del matchpipe


# In[79]:


matchpipe = DomainDetector(
    nlp, class_types=["LOCATION"], method="matcher", use_negated=True
)


# In[80]:


matchpipe.from_disk(PATTERN_DIR)


# In[81]:


test_doc = all_docs[random.randint(0, 10000)]
print(test_doc)
labelled_doc = matchpipe(test_doc)
for e in labelled_doc.ents:
    print(e.text, e.label_, e._.is_negated, e._.domain)


# ## Investigate entitiy coverage

# In[82]:


matched_docs = [matchpipe(doc) for doc in tqdm(all_docs)]


# In[83]:


def tp_condition_ent(e):
    if e.label_ == "LOCATION":
        if e._.domain:
            return True
    return False


def fp_condition_ent(e):
    if not (e.label_ == "LOCATION"):
        if e._.domain:
            return True
    return False


def tn_condition_ent(e):
    if not (e.label_ == "LOCATION"):
        if not e._.domain:
            return True
    return False


def fn_condition_ent(e):
    if e.label_ == "LOCATION":
        if not e._.domain:
            return True
    return False


# In[84]:


fn = len([e for doc in matched_docs for e in doc.ents if fn_condition_ent(e)])
fp = len([e for doc in matched_docs for e in doc.ents if fp_condition_ent(e)])
tn = len([e for doc in matched_docs for e in doc.ents if tn_condition_ent(e)])
tp = len([e for doc in matched_docs for e in doc.ents if tp_condition_ent(e)])


# In[85]:


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


# ## Orphaned entities

# In[86]:


orphaned_paths = [e for doc in matched_docs for e in doc.ents if fn_condition_ent(e)]


# In[87]:


orphaned_paths_strs = [e.text.lower() for e in orphaned_paths]


# In[88]:


asdf = pd.Series(orphaned_paths_strs)


# In[89]:


valcounts = asdf.value_counts()


# In[90]:


valcounts[:50]


# In[91]:


tp_array = tp + valcounts.cumsum() - valcounts
fn_array = valcounts
recall_array = tp_array / (tp_array + fn_array)


# In[92]:


valcounts = valcounts.to_frame("counts")


# In[93]:


valcounts["recall"] = recall_array


# In[64]:


valcounts[:200].to_csv("extra_terms_vs_recall_200_mesh.csv")


# In[49]:


valcounts["n_extra_terms"] = np.arange(1, len(valcounts) + 1)


# In[50]:


ax = valcounts.plot(x="n_extra_terms", y="recall", xlim=(0, 100), figsize=(10, 8))


# In[94]:


print(valcounts[:50])


# ## Labelled orphans

# In[66]:


labelledorphans = pd.read_csv(
    "DATAPATH/Desktop/neuroData/domain_patterns/mesh_anat_patterns/extra_terms_vs_recall_200_mesh.csv"
)


# In[68]:


relevant_orphans = labelledorphans[~labelledorphans["label"].isna()]


# In[69]:


domain_dict.keys()


# In[70]:


new_domain_dict = {key: val for key, val in domain_dict.items()}


# In[71]:


new_domain_dict["brain"] = ["brain"]


# In[75]:


for name, _, _, _, cui in relevant_orphans.to_records(index=False):
    domain = onto[cui].name
    patterns = new_domain_dict[domain]
    patterns.append(name)
    new_domain_dict[domain] = patterns


# In[76]:


PATTERN_DIR = Path("DATAPATH/Desktop/neuroData/domain_patterns/mesh_anat_patterns")


# In[77]:


with open(PATTERN_DIR / "patterns.json", "w") as file:
    json.dump(new_domain_dict, file, indent=4)


# In[ ]:
