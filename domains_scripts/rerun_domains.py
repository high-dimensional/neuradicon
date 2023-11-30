from pathlib import Path

import pandas as pd
import spacy
from neuroNLP.custom_pipes import *
from tqdm import tqdm

data_path = Path("DATAPATH/Desktop/neuroData")
nlp_models_path = Path("DATAPATH/Desktop/neuroNLP/training")

PATH_PATTERN_DIR = Path(
    "DATAPATH/Desktop/neuroData/domain_patterns/descriptor_patterns_v3"
)

REPORTS = data_path / "cleaned_report_data" / "process_reports_joined_v3.csv"

df = pd.read_csv(REPORTS, low_memory=False, index_col=0)

spacy.prefer_gpu()


nlp = spacy.load(nlp_models_path / "rel_neg_detection_tok2vec_v2")
nlp_docs = nlp.pipe(df["report_body"], batch_size=512)
domainer = DomainDetector(nlp)
domainer.from_disk(PATH_PATTERN_DIR)
# %%
domains = ["|".join(domainer(doc)._.domains) for doc in tqdm(nlp_docs, total=len(df))]
df["descriptor_domains_v2"] = domains
# %%
df.to_csv(
    data_path / "cleaned_report_data/process_reports_joined_v4.csv",
)
