import pytest

from neuradicon.models import *

report_sample = ["7210825 20/02/2018 MR Head 7210825 20/02/2018 Imaging under GA 7210825 20/02/2018 MR MRA Clinical Indication: Stroke patient thrombolysed 19/02/18. Low GCS I+V to ICU NOTE: Patient likely I+V On ICU Findings: Reference made to CT scan dated 19/02/18. No foci of restricted diffusion to suggest infarction. No evidence of haemorrhage and no focal mass lesion. The irregularity of M1 segment of the left middle cerebral artery is not well demonstrated on TOF-MRI. Normal flow-related signal is seen in the imaged portions of ICA, vertebral arteries and intracranial arteries. Conclusion: No evidence of infarction. GC Dr P S Rangi Consultant Neuroradiologist GMC NO: 4189686 Email: neurorad.uclh.nhs.uk",
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
    assert output[0]["report_body"][:14] == "No foci of res"


def test_normality_model():
    model_loc = "./model_dir/normality_model"
    model = NormalityModel(model_loc)
    output = model(report_sample)
    assert output[0] == "IS_NORMAL"


def test_embedding_model():
    model_loc = "./model_dir/embedding_model"
    model = EmbeddingModel(model_loc)
    output = model(report_sample)
    assert output.shape == (1, 2)


def test_signal_model():
    model_loc = "./model_dir/signal_model"
    model = SignalModel(model_loc)
    output = model(report_sample)
    assert output[0]["T1-asserted-high-signal"] == 0


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
    assert first_links[0][1] == "superior"
