"""
Stage 1 - Define labels + sample GoEmotions.

Pulls the official GoEmotions test split (via HuggingFace `datasets`),
takes a stratified sample so every emotion is represented, and writes
it to goemotions_sample.csv in the same schema as custom_challenge_set.csv
so both files can be fed through the same scoring pipeline.

Run:  python 01_sample_goemotions.py --n_per_label 4
(28 labels x 4 = ~112 rows -- big enough to be meaningful, small enough
to score in the time you have.)
"""
import argparse
import pandas as pd
from datasets import load_dataset

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def main(n_per_label: int, seed: int):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    id2label = {i: l for i, l in enumerate(ds.features["labels"].feature.names)}

    rows = []
    for ex in ds:
        gold = [id2label[i] for i in ex["labels"]]
        rows.append({"text": ex["text"], "gold_labels": ",".join(gold)})
    df = pd.DataFrame(rows)

    # stratified sample: for each label, grab n_per_label examples that
    # include that label as one of their (possibly multiple) gold labels
    picked = []
    seen_idx = set()
    rng_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle once
    for label in LABELS:
        mask = rng_df["gold_labels"].apply(lambda g: label in g.split(","))
        candidates = rng_df[mask]
        candidates = candidates[~candidates.index.isin(seen_idx)]
        take = candidates.head(n_per_label)
        seen_idx.update(take.index.tolist())
        picked.append(take)

    sample = pd.concat(picked).drop_duplicates(subset="text").reset_index(drop=True)
    sample.insert(0, "id", [f"G{idx+1:03d}" for idx in range(len(sample))])
    sample["feature_type"] = "goemotions"

    sample.to_csv("goemotions_sample.csv", index=False)
    print(f"Wrote {len(sample)} rows to goemotions_sample.csv")
    print(sample["gold_labels"].apply(lambda g: g.split(",")[0]).value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_label", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.n_per_label, args.seed)
