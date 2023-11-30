from pathlib import Path

import spacy
from sklearn.metrics import classification_report

package_path = Path("../packages")

report_sample = []

normal_labels = ["NONIS_NORMAL", "IS_NORMAL", "IS_NORMAL", "IS_NORMAL"]
comparative_labels = [
    "IS_COMPARATIVE",
    "IS_COMPARATIVE",
    "NONIS_COMPARATIVE",
    "IS_COMPARATIVE",
]
normal_cls_path = package_path / "en_norm_cls-1.0" / "en_norm_cls" / "en_norm_cls-1.0"
comparative_cls_path = (
    package_path / "en_comp_cls-1.0" / "en_comp_cls" / "en_comp_cls-1.0"
)

norm_cls = spacy.load(normal_cls_path)
comp_cls = spacy.load(comparative_cls_path)

docs = [norm_cls(i) for i in report_sample]
normal_predictions = [max(d.cats, key=d.cats.get) for d in docs]

docs = [comp_cls(i) for i in report_sample]
comparative_predictions = [max(d.cats, key=d.cats.get) for d in docs]

print(classification_report(comparative_labels, comparative_predictions))
print(classification_report(normal_labels, normal_predictions))
