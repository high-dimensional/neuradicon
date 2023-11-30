from pathlib import Path

import numpy as np
import spacy
from neuroNLP.custom_pipes import *
from neuroonto.onto import Ontology, find_domain_patterns
from sklearn.neighbors import NearestNeighbors

# %%
ROOT = Path("DATAPATH/Desktop")
VECTORMODEL = (
    ROOT / "neuroNLP/training/floret_custom_vectors/floret_custom_vectors.floret_model"
)
MESH = ROOT / "neuroOnto/ontologies/mesh"
SNOMED = ROOT / "neuroOnto/ontologies/snomed"
QS = ROOT / "neuroOnto/ontologies/qs"
CUSTOM = ROOT / "neuroOnto/ontologies/custom"
BASEMODEL = ROOT / "neuroNLP/training/rel_neg_detection_tok2vec_v2"

report_sample = [
    "Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies dated 16/02 17 and 09/01/2018. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses. WM/ Dr Sachit Shah Consultant Neuroradiologist neurorad@uclh.nhs.uk",
    "There is moderate dilatation of the third and lateral ventricles, distension of the third ventricular recesses and mild enlargement of the pituitary fossa with depression of the glandular tissue. Appearances are in keeping with hydrocephalus and the mild associated frontal and peritrigonal transependymal oedema indicates an ongoing active element. No cause for hydrocephalus is demonstrated and the fourth ventricle and aqueduct are normal in appearance. Note is made of T1 shortening within the floor of the third ventricle. On the T2 gradient echo there is a bulbous focus of reduced signal in this region that is inseparable from the terminal basilar artery but likely to be lying anterosuperiorly. The combination of features is most likely to reflect a small incidental dermoid within the third ventricle with fat and calcified components. This could be confirmed by examination of the CT mentioned on the request form. If the presence of fat and calcium is not corroborated on the plain CT, a time-of-flight MR sequence or CTA would be prudent to exclude the small possibility of a vascular abnormality in this region. Dr. Matthew Adams",
    "Clinical details: L1 conus SOL. MRI brain/whole spine to rule out other lesions. Findings: There is bony asymmetry of the occipital bones and a small lipoma within the nuchal soft tissues. The intracranial appearances are otherwise normal. Unchanged appearances of the previously demonstrated enhancing lumbar mass lesion. Note is again made of mild spinal cord compression at T12-L1 secondary to degenerative disc changes. No other spinal or intracranial lesions are identified. KW",
    "Clinical Indications for MRI - White matter lesions on earlier scan has phospholipuid syndrome but ? MS Will need contrast Findings: Comparison is made with the previous scan performed 6 April 2016. Stable appearances of the diffuse and confluent bilateral white matter high T2/FLAIR signal changes. A few more conspicuous focal lesions in the periventricular and juxta-cortical white matter are again demonstrated and unchanged. There is no evidence of signal changes in the posterior fossa structures. The imaged cervical cord returns normal signal. There is no evidence of pathological enhancement. Summary: Stable appearances of the supratentorial white matter signal changes. Although some lesions appear more conspicuous in the periventricular and juxtacortical regions, there is no significant lesion burden to characterise dissemination in space at this time point. Dr Kelly Pegoretti Consultant Neuroradiologist email: neurorad@uclh.nhs.uk",
]

nlp = spacy.load(BASEMODEL)
vecnlp = spacy.load(VECTORMODEL)
# %%
onto = Ontology(MESH / "ontology_concepts.csv", MESH / "ontology_relations.csv")

# %%
root_con = onto.get_root()
root_con.children[0].children
# %%
# domains_cuis = ['D009422',  'D002943',  'D007239', 'D007154', 'D009369', 'D064419', 'D006425', 'D002319', 'D009141', 'D004700', 'D013812', 'D017437', 'D006424', 'D009140', 'D005128','D002318', 'D001829', 'D013514', 'D009358', 'D013568', 'D009420', 'D004703', 'D009750']
# domains_cuis = ['404684003', '363787002', '272379006', '123037004']
domains_cuis = [root_con.cui]
all_aliases = find_domain_patterns(domains_cuis, onto)
# %%
docs = [nlp(t) for t in report_sample]
# %%
all_ents = [e.text.lower() for d in docs for e in d.ents]
# %%
list_aliases = [i for k, v in all_aliases.items() for i in v]
all_vectors = np.array([vecnlp(i).vector for i in list_aliases])
# %%
nn = NearestNeighbors(radius=6.0)
nn.fit(all_vectors)
# %%
all_ent_vecs = np.array([vecnlp(e).vector for e in all_ents])
# %%
dist, idx = nn.radius_neighbors(all_ent_vecs)  # kneighbors(all_ent_vecs)
# %%

for i, val in enumerate(all_ents):
    print("\n")
    print(val)
    print([list_aliases[j] for j in idx[i]])
