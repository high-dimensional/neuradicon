#!/usr/bin/env python
# coding: utf-8

# # Pathological domain assignment
# - Create a classifier on top of entities to extract pathological domain labels
# - ideally would use word vectors but for now try just with tfidf char ngrams
# - start with knn classifier

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

# In[4]:


ontologies = Path("DATAPATH/Desktop/neuroOnto/ontologies")
chosen_onto = ontologies / "mesh"
cons = chosen_onto / "ontology_concepts.csv"
rels = chosen_onto / "ontology_relations.csv"


# In[5]:


onto = Ontology(cons, rels)


# In[6]:


root = onto.get_root()


# In[9]:


radlex_pathological_domain_types = [
    "RID5230",
    "RID7473",
    "RID4738",
    "RID39206",
    "RID34615",
    "RID3381",
    "RID49479",
    "RID5043",
    "RID34717",
    "RID34783",
    "RID35738",
    "RID4590",
    "RID3730",
    "RID34413",
    "RID4564",
    "RID50261",
    "RID39546",
    "RID34658",
    "RID34648",
    "RID34784",
    "RID4628",
    "RID29041",
    "RID5238",
]


# In[8]:


snomed_pathological_domain_types = [
    "49601007",
    "75478009",
    "7895008",
    "283025007",
    "66091009",
    "362969004",
    "371405004",
    "34093004",
    "362971004",
    "41266007",
    "105969002",
    "128117002",
    "126950007",
    "118940003",
    "75934005",
    "2492009",
    "16545005",
]


# In[134]:


mesh_pathological_domain_types = [
    "D002318",
    "D064419",
    "D006259",
    "D009358",
    "D004700",
    "D005128",
    "D006425",
    "D007154",
    "D003240",
    "D002494",
    "D009423",
    "D009422",
    "D009750",
    "D019635",
    "D013568",
    "D013812",
    "D001847",
]


# In[165]:


pathological_domain_types = mesh_pathological_domain_types


# ## Load data

# In[166]:


data_dir = Path("DATAPATH/Desktop/neuroData/170k_spacy_docs_bodies_new_format")
nlp_dir = "/training/rel_neg_detection_tok2vec_v2"


# In[167]:


print("loading pipeline")
nlp = spacy.load(nlp_dir)
print("\nFull pipeline: ", nlp.pipe_names)


# In[168]:


print("preparing structured docs")
start_time = time.time()
doc_bin = DocBin().from_disk(data_dir / "neuronlp_docs.spacy")
all_docs = list(doc_bin.get_docs(nlp.vocab))
end_time = time.time()
print("structured doc execution time: {} seconds".format(end_time - start_time))


# ## Get domain patterns

# In[169]:


custom_pdomain_concepts = [onto[cui] for cui in pathological_domain_types]


# In[136]:


domain_cuis = [c.cui for c in custom_pdomain_concepts]


# In[137]:


domain_dict = find_domain_patterns(domain_cuis, onto)


# In[10]:


PATTERN_DIR = "DATAPATH/Desktop/neuroData/meshplus_path_patterns"


# In[11]:


with open(PATTERN_DIR + "/patterns.json", "w") as file:
    json.dump(domain_dict, file, indent=4)


# ## Test pipe

# In[179]:


del matchpipe


# In[180]:


matchpipe = DomainDetector(nlp, method="matcher", use_negated=True)


# In[181]:


domainer_PATTERN_DIR = "DATAPATH/Desktop/neuroData/meshplus_path_patterns_v2"


# In[182]:


matchpipe.from_disk(Path(domainer_PATTERN_DIR))


# In[173]:


test_doc = all_docs[random.randint(0, 10000)]
print(test_doc)
labelled_doc = matchpipe(test_doc)
for e in labelled_doc.ents:
    print(e.text, e.label_, e._.is_negated, e._.domain)


# In[174]:


print(labelled_doc._.domains)


# In[15]:


print([e._.domain for e in labelled_doc.ents])


# ## cui classifier

# In[12]:


cgen = CandidateGenerator(onto)


# In[13]:


ent_strings = [e.text.lower() for e in test_doc.ents]
root_ent_strings = [e.root.text.lower() for e in test_doc.ents]


# In[14]:


cui_classifications = cgen.get_cls_candidate(ent_strings)
root_cui_classifications = cgen.get_cls_candidate(root_ent_strings)

for string, cui in zip(ent_strings, cui_classifications):
    print(string, ": ", onto[cui])

for string, cui in zip(root_ent_strings, root_cui_classifications):
    print(string, ": ", onto[cui])


# ## Investigate report coverage
# - how many of the reports with asserted pathologies are assigned to a domain?

# In[183]:


matched_docs = [matchpipe(doc) for doc in tqdm(all_docs)]


# In[45]:


def tp_condition(d):
    return bool(len(d._.domains) > 0) and bool(
        len([e for e in d.ents if (e.label_ == "PATHOLOGY") and (not e._.is_negated)])
        > 0
    )


def fp_condition(d):
    return bool(len(d._.domains) > 0) and bool(
        len([e for e in d.ents if (e.label_ == "PATHOLOGY") and (not e._.is_negated)])
        == 0
    )


def tn_condition(d):
    return bool(len(d._.domains) == 0) and bool(
        len([e for e in d.ents if (e.label_ == "PATHOLOGY") and (not e._.is_negated)])
        == 0
    )


def fn_condition(d):
    return bool(len(d._.domains) == 0) and bool(
        len([e for e in d.ents if (e.label_ == "PATHOLOGY") and (not e._.is_negated)])
        > 0
    )


# In[26]:


fn = len([doc for doc in matched_docs if fn_condition(doc)])
fp = len([doc for doc in matched_docs if fp_condition(doc)])
tn = len([doc for doc in matched_docs if tn_condition(doc)])
tp = len([doc for doc in matched_docs if tp_condition(doc)])


# In[12]:


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


# ## Investigate pathological entitiy coverage

# In[184]:


def tp_condition_ent(e):
    if e.label_ == "PATHOLOGY":
        if e._.domain:
            return True
    return False


def fp_condition_ent(e):
    if not (e.label_ == "PATHOLOGY"):
        if e._.domain:
            return True
    return False


def tn_condition_ent(e):
    if not (e.label_ == "PATHOLOGY"):
        if not e._.domain:
            return True
    return False


def fn_condition_ent(e):
    if e.label_ == "PATHOLOGY":
        if not e._.domain:
            return True
    return False


# In[185]:


fn = len([e for doc in matched_docs for e in doc.ents if fn_condition_ent(e)])
fp = len([e for doc in matched_docs for e in doc.ents if fp_condition_ent(e)])
tn = len([e for doc in matched_docs for e in doc.ents if tn_condition_ent(e)])
tp = len([e for doc in matched_docs for e in doc.ents if tp_condition_ent(e)])


# In[186]:


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

# In[187]:


orphaned_paths = [e for doc in matched_docs for e in doc.ents if fn_condition_ent(e)]


# In[188]:


orphaned_paths_strs = [e.text.lower() for e in orphaned_paths]


# In[189]:


asdf = pd.Series(orphaned_paths_strs)


# In[190]:


valcounts = asdf.value_counts()


# In[191]:


valcounts[:50]


# In[34]:


onto.name2cui("glioma")


# In[34]:


tp_array = tp + valcounts.cumsum() - valcounts
fn_array = valcounts
recall_array = tp_array / (tp_array + fn_array)


# In[35]:


valcounts = valcounts.to_frame("counts")


# In[36]:


valcounts["recall"] = recall_array


# In[38]:


valcounts[:200].to_csv("extra_terms_vs_recall_200_meshplus.csv")


# In[39]:


valcounts["n_extra_terms"] = np.arange(1, len(valcounts) + 1)


# In[40]:


ax = valcounts.plot(x="n_extra_terms", y="recall", xlim=(0, 100), figsize=(10, 8))


# In[33]:


fig = ax.get_figure()
fig.savefig("recallvsextraterms.png")


# ## Labelled orphans

# In[141]:


PN_ORPHAN_LABELS = Path(
    "DATAPATH/Desktop/neuroData/extra_terms_vs_recall_200_mesh.PN[2877].csv"
)
PN_ORPHAN_LABELS2 = Path(
    "DATAPATH/Desktop/neuroData/extra_terms_vs_recall_200_meshplusPN.csv"
)


# In[142]:


labelled_orphans1 = pd.read_csv(PN_ORPHAN_LABELS)
labelled_orphans2 = pd.read_csv(PN_ORPHAN_LABELS2)


# In[143]:


labelled_orphans2 = labelled_orphans2.drop(columns=["counts"])


# In[144]:


labelled_orphans2.tail()


# In[145]:


labelled_orphans = pd.concat(
    [labelled_orphans1, labelled_orphans2], axis=0, ignore_index=True
)


# In[146]:


labelled_orphans.tail()


# In[147]:


labelled_orphans["SUBID"] = labelled_orphans["SUBID"].fillna(labelled_orphans["ID"])


# In[148]:


irrelevant_codes = [
    "descriptor",
    "normal",
    "spine",
    "comparative",
    "anatomy",
    "stable",
    "instrumental",
    "ent",
    "ENT",
    "anatomy & D013568",
    "artefact",
]


# In[149]:


relevant_orphans = labelled_orphans[~labelled_orphans["ID"].isin(irrelevant_codes)]


# In[150]:


relevant_orphans["SUBID"].unique()


# In[151]:


for i in relevant_orphans["SUBID"].unique():
    if i not in mesh_pathological_domain_types:
        print(onto[i])


# In[152]:


relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D002561", "D002318", regex=False
)
relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D020300", "D002318", regex=False
)
relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D009421", "D009422", regex=False
)
relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D020196", "D006259", regex=False
)
relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D006849", "D009422", regex=False
)
relevant_orphans["SUBID"] = relevant_orphans["SUBID"].str.replace(
    "D020274", "D009422", regex=False
)


# In[153]:


for i in relevant_orphans["SUBID"].unique():
    print(onto[i])


# In[154]:


PATTERN_DIR = "DATAPATH/Desktop/neuroData/mesh_pathology_patterns_v3"
with open(PATTERN_DIR + "/patterns.json", "r") as file:
    mesh_patterns = json.load(file)


# In[155]:


orphan_records = relevant_orphans.to_records(index=False)


# In[156]:


orphan_domains = {string: onto[SUBID].name for string, _, _, SUBID, _ in orphan_records}


# In[157]:


for string, domain in orphan_domains.items():
    pattern_list = mesh_patterns[domain]
    pattern_list.append(string)
    mesh_patterns[domain] = pattern_list


# In[158]:


mesh_patterns.keys()


# In[163]:


PATTERN_DIR = "DATAPATH/Desktop/neuroData/meshplus_path_patterns_v2"


# In[164]:


with open(PATTERN_DIR + "/patterns.json", "w") as file:
    json.dump(mesh_patterns, file, indent=4)


# ## Domain counts & whole report domains

# In[192]:


all_domains = [d for doc in matched_docs for d in doc._.domains]
domain_counts = {d: all_domains.count(d) for d in set(all_domains)}


# In[194]:


len(all_domains)


# In[193]:


print(domain_counts)


# In[195]:


n_domains = [len(doc._.domains) for doc in matched_docs]


# In[196]:


n_domains = pd.Series(n_domains)


# In[197]:


n_domains.value_counts()


# In[198]:


unmatched_reports = [doc for doc in matched_docs if len(doc._.domains) == 0]


# In[204]:


unmatched_reports[random.randint(0, len(unmatched_reports))]


# In[ ]:
