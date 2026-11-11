"""Label the pool with each LLM and save one CSV per model.

Needs a GPU. On Kaggle, run from the repo root:
    PYTHONPATH=src python scripts/run_labelling.py --pool /kaggle/input/.../pool.csv --out-dir /kaggle/working
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from quorum.evaluation import add_human_labels, evaluate
from quorum.labelling import (check_label_mass, check_label_tokens, free_gpu,
                              label_dataset, load_model)

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "google/gemma-2-9b-it",              # largest, so it loads onto empty GPUs
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]


def run_model(model_name, pool, prompt_template, out_dir):
    """label the pool with one model, save labels_<model>.csv, return its summary"""
    out_path = out_dir / f"labels_{model_name.split('/')[-1]}.csv"
    model, tokenizer = None, None
    try:
        model, tokenizer = load_model(model_name)
        check_label_tokens(tokenizer)
        print(label_dataset(pool.head(5), prompt_template, model, tokenizer))   # smoke test

        start = time.time()
        results = label_dataset(pool, prompt_template, model, tokenizer, out_path=out_path)
        elapsed = time.time() - start
        print(f"\n{elapsed / 60:.1f} min for {len(pool)} sentences")

        check_label_mass(results)
        compare = add_human_labels(results, pool)
        compare.to_csv(out_path, index=False)

        summary = evaluate(compare, model_name)
        summary["minutes"] = elapsed / 60
        return summary
    finally:
        # runs whether the run succeeded or a check failed
        del model, tokenizer
        free_gpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=ROOT / "data/processed/pool.csv")
    parser.add_argument("--prompt", default=ROOT / "configs/prompts/prompt_v1.txt")
    parser.add_argument("--out-dir", default=ROOT / "data/labels")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N sentences (for testing)")
    args = parser.parse_args()

    pool = pd.read_csv(args.pool)
    if args.limit:
        pool = pool.head(args.limit)
    prompt_template = Path(args.prompt).read_text()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool: {len(pool)} sentences, prompt: {args.prompt}")

    summaries = []
    for model_name in args.models:
        try:
            summaries.append(run_model(model_name, pool, prompt_template, out_dir))
        except Exception as e:
            print(f"\n!!! {model_name} failed: {e}")

    table = pd.DataFrame(summaries)
    table.to_csv(out_dir / "labelling_summary.csv", index=False)
    print("\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
