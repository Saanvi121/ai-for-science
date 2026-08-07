"""
Stage 3 - Score answers + compute metrics.

Reads predictions.csv and, for each model, computes:
  - lenient accuracy  (correct if prediction is ANY of the gold labels)
  - strict accuracy   (correct only if prediction == the PRIMARY/first gold label)
  - precision / recall / macro-F1 (using strict/primary-label scoring,
    since sklearn's classification_report needs one gold label per row)
  - per-emotion F1
  - confusion matrix (primary label vs prediction)

Metrics are reported overall, and split by source (goemotions vs the
custom challenge set) so you can see the generalization gap directly.

Run:  python 03_score_and_metrics.py
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

MODELS = ["gemini", "llama", "gemma"]


def primary_label(gold_labels: str) -> str:
    return gold_labels.split(",")[0].strip()


def lenient_correct(pred: str, gold_labels: str) -> bool:
    gold_set = {g.strip() for g in gold_labels.split(",")}
    return pred in gold_set


def report_for(df: pd.DataFrame, model: str, tag: str):
    pred_col = f"{model}_pred"
    df = df.copy()
    df["primary"] = df["gold_labels"].apply(primary_label)
    df["lenient_ok"] = df.apply(lambda r: lenient_correct(r[pred_col], r["gold_labels"]), axis=1)

    lenient_acc = df["lenient_ok"].mean()
    strict_acc = accuracy_score(df["primary"], df[pred_col])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["primary"], df[pred_col], average="macro", zero_division=0
    )

    print(f"\n=== {model.upper()} | {tag} (n={len(df)}) ===")
    print(f"Lenient accuracy (any gold label): {lenient_acc:.3f}")
    print(f"Strict accuracy (primary label):   {strict_acc:.3f}")
    print(f"Macro precision: {precision:.3f}  Macro recall: {recall:.3f}  Macro F1: {f1:.3f}")

    return {
        "model": model, "subset": tag, "n": len(df),
        "lenient_acc": lenient_acc, "strict_acc": strict_acc,
        "macro_precision": precision, "macro_recall": recall, "macro_f1": f1,
    }


def main():
    df = pd.read_csv("predictions.csv")
    df["source"] = df["feature_type"].apply(
        lambda ft: "goemotions" if ft == "goemotions" else "custom_challenge"
    )

    summary_rows = []
    for model in MODELS:
        summary_rows.append(report_for(df, model, "overall"))
        summary_rows.append(report_for(df[df.source == "goemotions"], model, "goemotions"))
        summary_rows.append(report_for(df[df.source == "custom_challenge"], model, "custom_challenge"))

        # per-emotion F1 (strict/primary), full report -> csv
        sub = df.copy()
        sub["primary"] = sub["gold_labels"].apply(primary_label)
        report_dict = classification_report(
            sub["primary"], sub[f"{model}_pred"], zero_division=0, output_dict=True
        )
        pd.DataFrame(report_dict).T.to_csv(f"per_emotion_f1_{model}.csv")

        # confusion matrix -> csv
        labels_sorted = sorted(set(sub["primary"]) | set(sub[f"{model}_pred"]))
        cm = confusion_matrix(sub["primary"], sub[f"{model}_pred"], labels=labels_sorted)
        pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
            f"confusion_matrix_{model}.csv"
        )

    pd.DataFrame(summary_rows).to_csv("metrics_summary.csv", index=False)
    print("\nWrote metrics_summary.csv, per_emotion_f1_<model>.csv, confusion_matrix_<model>.csv")


if __name__ == "__main__":
    main()
