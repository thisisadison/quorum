"""Compare LLM labels against the human labels."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from quorum.labelling import LABELS

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def add_human_labels(results, df):
    """attach the human label to each result, matched on item_id"""
    compare = results.merge(df[["item_id", "label"]], on="item_id")
    compare["human_label"] = compare["label"].map(LABEL_MAP)
    compare["confidence"] = compare[LABELS].max(axis=1)
    compare["correct"] = compare["llm_label"] == compare["human_label"]
    return compare


def evaluate(compare, model_name):
    """print the detailed report and return the key numbers for the comparison table"""
    print(f"\n===== {model_name} =====")
    print(classification_report(compare["human_label"], compare["llm_label"]))
    print(confusion_matrix(compare["human_label"], compare["llm_label"], labels=LABELS))

    return {
        "model": model_name,
        "accuracy": accuracy_score(compare["human_label"], compare["llm_label"]),
        "macro_f1": f1_score(compare["human_label"], compare["llm_label"], average="macro"),
        "below_0.90": (compare["confidence"] < 0.90).mean(),
        "conf_when_right": compare.loc[compare["correct"], "confidence"].mean(),
        "conf_when_wrong": compare.loc[~compare["correct"], "confidence"].mean(),
    }
