import pandas as pd

from quorum.data_prep import prepare_splits
from pathlib import Path

def main():
    pool, test = prepare_splits(test_size=0.15, seed=42)

    print(f"\nPool: {len(pool)} rows")
    print(pool["label"].value_counts(normalize=True).round(3))

    print(f"\nTest: {len(test)} rows")
    print(test["label"].value_counts(normalize=True).round(3))

    df_to_csv(pool, "pool")
    df_to_csv(test, "test")
    
    return pool, test

def df_to_csv(df, filename):
    df.to_csv(f"data/processed/{filename}.csv", index=False)
    print(f"\nSave {filename} Pandas Dataframe to CSV...")

if __name__ == "__main__":
    main()