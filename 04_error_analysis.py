"""
Stage 4 - Analyze errors.

Reads predictions.csv and, per model, breaks the ERROR RATE down by:
  - feature_type (sarcasm / slang / emoji / figurative / indirect / mixed / goemotions)
  - text length bucket (short <8 words, medium 8-20, long >20)

This tells you whether mistakes cluster around specific language
features (systematic) or look roughly even across categories (random).

Run:  python 04_error_analysis.py
"""
import pandas as pd

MODELS = ["gemini", "llama", "gemma"]


def primary_label(gold_labels: str) -> str:
    return gold_labels.split(",")[0].strip()


def length_bucket(text: str) -> str:
    n = len(str(text).split())
    if n < 8:
        return "short (<8 words)"
    if n <= 20:
        return "medium (8-20 words)"
    return "long (>20 words)"


def main():
    df = pd.read_csv("predictions.csv")
    df["primary"] = df["gold_labels"].apply(primary_label)
    df["length_bucket"] = df["text"].apply(length_bucket)

    rows = []
    for model in MODELS:
        df[f"{model}_correct"] = df["primary"] == df[f"{model}_pred"]

        by_feature = (
            1 - df.groupby("feature_type")[f"{model}_correct"].mean()
        ).rename("error_rate").reset_index()
        by_feature["model"] = model
        by_feature["breakdown"] = "feature_type"
        by_feature = by_feature.rename(columns={"feature_type": "group"})

        by_length = (
            1 - df.groupby("length_bucket")[f"{model}_correct"].mean()
        ).rename("error_rate").reset_index()
        by_length["model"] = model
        by_length["breakdown"] = "length_bucket"
        by_length = by_length.rename(columns={"length_bucket": "group"})

        rows.append(by_feature)
        rows.append(by_length)

    out = pd.concat(rows, ignore_index=True)
    out = out[["model", "breakdown", "group", "error_rate"]].sort_values(
        ["model", "breakdown", "error_rate"], ascending=[True, True, False]
    )
    out.to_csv("error_analysis.csv", index=False)
    print(out.to_string(index=False))
    print("\nWrote error_analysis.csv")

    # quick worst-offenders list per model: misclassified custom-challenge rows
    misses = df[
        (df["feature_type"] != "goemotions")
        & ~(df[[f"{m}_correct" for m in MODELS]].all(axis=1))
    ][["id", "text", "gold_labels", "feature_type"] + [f"{m}_pred" for m in MODELS]]
    misses.to_csv("misclassified_examples.csv", index=False)
    print(f"Wrote misclassified_examples.csv ({len(misses)} rows for manual review)")


if __name__ == "__main__":
    main()
