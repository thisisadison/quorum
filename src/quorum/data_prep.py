import hashlib
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_fpb_data(config_name):
    print("Loading Financial PhraseBank...")
    ds = load_dataset("takala/financial_phrasebank", config_name, trust_remote_code=True)
    df = ds['train'].to_pandas()
    print(df.head())
    print(f"\n Unique Labels: {df["label"].unique()}")
    return df

def make_item_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# 85/15 train/test split
# stratified by label to ensure similar distribution of labels
def split_pool_test(df, test_size, seed):
    print("\nSplitting into pool and test...")
    pool, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    assert len(pool) + len(test) == len(df)
    assert set(pool["item_id"]).isdisjoint(set(test["item_id"])), "Pool and Test leakage"
    return pool, test

def prepare_splits(test_size, seed):
    df = load_fpb_data("sentences_50agree")
    before = len(df)
    df = df.drop_duplicates(subset="sentence", keep="first")
    n = before - len(df)
    print(f"\nRemoved {n} duplicates") 
    df["item_id"] = df["sentence"].apply(make_item_id) 
    df = df[["item_id", "sentence", "label"]]

    pool, test = split_pool_test(df, test_size=test_size, seed=seed)

    return pool, test