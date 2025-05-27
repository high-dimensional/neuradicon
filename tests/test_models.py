import pytest

from neuradicon.models import *

report_sample = [
    "placeholder"
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
