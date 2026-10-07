import hashlib
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_fpb_data(config_name):
    ds = load_dataset("takala/financial_phrasebank", config_name, trust_remote_code=True)
    df = ds['train'].to_pandas()
    print(df.head())
    print(df["label"].unique())
    return df

def make_item_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# 85/15 train/test split
# stratified by label to ensure similar distribution of labels
def split_pool_test(df, test_size, seed):
    pool, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    assert len(pool) + len(test) == len(df)
    assert set(pool["item_id"]).isdisjoint(set(test["item_id"])), "Pool and Test leakage"
    return pool, test

def main():
    df = load_fpb_data("sentences_50agree")
    df = df.drop_duplicates(subset="sentence", keep="first")
    df["item_id"] = df["sentence"].apply(make_item_id) 
    df = df[["item_id", "sentence", "label"]]

    print(f"Total rows: {len(df)}")
    print(df["label"].value_counts(normalize=True).round(3))

    pool, test = split_pool_test(df, test_size=0.15, seed=42)
    print(f"\nPool: {len(pool)} rows")
    print(pool["label"].value_counts(normalize=True).round(3))

    print(f"\nTest: {len(test)} rows")
    print(test["label"].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    main()