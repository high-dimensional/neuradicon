import pytest

from neuradicon.models import *

report_sample = [
    "MRI Head & Neck Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies dated 16/02 17 and 09/01/2018. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. There is sign of multiple sclerosis in the frontal lobe. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses. WM/ Dr Sachit Shah Consultant Neuroradiologist neurorad@uclh.nhs.uk",
    "Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies dated 16/02 17 and 09/01/2018. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses. WM/ Dr Sachit Shah Consultant Neuroradiologist neurorad@uclh.nhs.uk",
    "There is moderate dilatation of the third and lateral ventricles, distension of the third ventricular recesses and mild enlargement of the pituitary fossa with depression of the glandular tissue. Appearances are in keeping with hydrocephalus and the mild associated frontal and peritrigonal transependymal oedema indicates an ongoing active element. No cause for hydrocephalus is demonstrated and the fourth ventricle and aqueduct are normal in appearance. Note is made of T1 shortening within the floor of the third ventricle. On the T2 gradient echo there is a bulbous focus of reduced signal in this region that is inseparable from the terminal basilar artery but likely to be lying anterosuperiorly. The combination of features is most likely to reflect a small incidental dermoid within the third ventricle with fat and calcified components. This could be confirmed by examination of the CT mentioned on the request form. If the presence of fat and calcium is not corroborated on the plain CT, a time-of-flight MR sequence or CTA would be prudent to exclude the small possibility of a vascular abnormality in this region. Dr. Matthew Adams",
    "Clinical details: L1 conus SOL. MRI brain/whole spine to rule out other lesions. Findings: There is bony asymmetry of the occipital bones and a small lipoma within the nuchal soft tissues. The intracranial appearances are otherwise normal. Unchanged appearances of the previously demonstrated enhancing lumbar mass lesion. Note is again made of mild spinal cord compression at T12-L1 secondary to degenerative disc changes. No other spinal or intracranial lesions are identified. KW",
    "Clinical Indications for MRI - White matter lesions on earlier scan has phospholipuid syndrome but ? MS Will need contrast Findings: Comparison is made with the previous scan performed 6 April 2016. Stable appearances of the diffuse and confluent bilateral white matter high T2/FLAIR signal changes. A few more conspicuous focal lesions in the periventricular and juxta-cortical white matter are again demonstrated and unchanged. There is no evidence of signal changes in the posterior fossa structures. The imaged cervical cord returns normal signal. There is no evidence of pathological enhancement. Summary: Stable appearances of the supratentorial white matter signal changes. Although some lesions appear more conspicuous in the periventricular and juxtacortical regions, there is no significant lesion burden to characterise dissemination in space at this time point. Dr Kelly Pegoretti Consultant Neuroradiologist email: neurorad@uclh.nhs.uk",
]


def test_artefact_model():
    model_loc = "./model_dir/artefact_model"
    model = ArtefactModel(model_loc)
    output = model(report_sample)
    assert not output[0]


def test_section_model():
    model_loc = "./model_dir/section_model"
    model = SectionModel(model_loc)
    output = model(report_sample)
    assert output[0]["report_body"][:14] == "There has been"


def test_normality_model():
    model_loc = "./model_dir/normality_model"
    model = NormalityModel(model_loc)
    output = model(report_sample)
    assert output[0] == "IS_COMPARATIVE"


def test_embedding_model():
    model_loc = "./model_dir/embedding_model"
    model = EmbeddingModel(model_loc)
    output = model(report_sample)
    assert output.shape == (5, 2)


def test_signal_model():
    model_loc = "./model_dir/signal_model"
    model = SignalModel(model_loc)
    output = model(report_sample)
    assert output[0]["T1-asserted-high-signal"] == 1


@pytest.fixture
def clinical_model():
    model_loc = "./model_dir/clinical_entity_model"
    model = ClinicalEntityModel(model_loc, link_locations=True)
    return model


def test_clinical_entity_model(clinical_model):
    output = clinical_model(report_sample)
    assert output[0]["ents"][0]["entity_group"] == "pathology_treatment"


def test_relex(clinical_model):
    output = clinical_model(report_sample)
    first_links = output[0]["links"]
    print(first_links)
    print(output)
    assert first_links[0][1] == "superior"
