import pytest
from spacy.tokens import Doc

from neurollm.utils import *

texts = [
   "words and more words",
    "   ",
]


@pytest.fixture
def model():
    model_loc = "./model_dir/clinical_entity_model"
    tokenizer_name = model_loc + "/basemodel"
    ner_name = model_loc + "/ner-fine-tuned"
    neg_name = model_loc + "/negation"
    neuro_pipeline = NeuroPipeline(
        ner_name,
        neg_name,
        tokenizer_name,
        aggregation_strategy="first",
        device="cuda:0",
    )
    return neuro_pipeline


@pytest.fixture
def docs(model):
    output = list(model(texts))
    MODEL = "./model_dir/clinical_entity_model/en_core_web_sm"
    converter = DocConverter(MODEL)
    doc1 = converter(output[0])
    doc2 = converter(output[1])
    return [doc1, doc2]


def test_neuro_pipeline(model):
    output = list(model(texts))
    assert output[0]["ents"][0]["entity_group"] == "pathology_cerebrovascular"


def test_sectioner():
    model = "./model_dir/section_model"
    tokenizer = model + "/basemodel"
    model = model + "/segmentation"
    sectioner = Sectioner(model, tokenizer)
    big_texts = [texts[0] for i in range(100)]
    output = list(sectioner(big_texts))
    true_body = "No foci of restricted diffusion to suggest infarction. No evidence of haemorrhage and no focal mass lesion. The irregularity of M1 segment of the left middle cerebral artery is not well demonstrated on TOF - MRI. Normal flow - related signal is seen in the imaged portions of ICA, vertebral arteries and intracranial arteries. Conclusion : No evidence of infarction."
    assert output[0]["report_body"] == true_body


def test_pipeline2doc(docs):
    doc = docs[0]
    assert isinstance(doc, Doc)
    assert doc.ents[0].text == "Stroke"
    assert doc.ents[0].label_ == "pathology_cerebrovascular"
    assert not doc.ents[0]._.negex


def test_relex(docs):
    doc2 = docs[1]
    relex = RelationExtractor(doc2.vocab)
    relex.from_disk("./model_dir/clinical_entity_model/relex_small")
    new_doc = relex(doc2)
    print(new_doc._.rel)
    print([(e, e._.relation) for e in new_doc.ents])
    assert new_doc.ents[13]._.relation[0].text == "supratentorial"


def test_doc2array(docs):
    relex = RelationExtractor(docs[0].vocab)
    relex.from_disk("./models-llm/relex_small")
    vctr = DocVectorizer()
    array = vctr.doc2array(docs[0])
    assert array[1, 0, 1] == 2
