import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)

# ----------------------------
# Config
# ----------------------------
TEST_PATH = "data/test.tsv"
RUNS = {
    "svm": "artifacts/svm/test_pred.tsv",
    "lstm": "artifacts/lstm/test_pred.tsv",
}

OUTDIR = Path("artifacts/eval")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LABEL_MAP = {"NOT": 0, "OFF": 1}
INV_LABEL_MAP = {0: "NOT", 1: "OFF"}

# ----------------------------
# Helpers
# ----------------------------

def load_gold(test_path: str) -> pd.DataFrame:
    df = pd.read_csv(test_path, sep="\t", header=None, names=["text", "label"])
    df["gold"] = df["label"].astype(str).str.strip().map(LABEL_MAP)
    if df["gold"].isna().any():
        bad = df.loc[df["gold"].isna(), "label"].unique()
        raise ValueError(f"Unexpected labels in {test_path}: {bad}")
    return df

def load_preds(pred_path: str) -> pd.Series:
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    p = pd.read_csv(pred_path, sep="\t", header=None, names=["pred_raw"])["pred_raw"].astype(str).str.strip()
    # Try int 0/1, else try map NOT/OFF, else try probabilities -> threshold .5
    try:
        pred = p.astype(int)
    except ValueError:
        if set(p.unique()) <= {"NOT", "OFF"}:
            pred = p.map(LABEL_MAP)
        else:
            # If they are probs, cast to float and threshold at 0.5
            try:
                prob = p.astype(float)
                pred = (prob >= 0.5).astype(int)
            except ValueError:
                raise ValueError("Predictions must be ints {0,1}, labels {NOT,OFF}, or probabilities in [0,1].")
    return pred

def plot_confusion(cm: np.ndarray, labels: list, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate cells
    max_val = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="black" if val < 0.7*max_val else "white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    # Escape underscores in index
    df_latex = df.copy()
    df_latex.index = [str(i).replace("_", r"\_") for i in df_latex.index]
    return df_latex.to_latex(escape=True, caption=caption, label=label, index=True, float_format="%.4f")

# ----------------------------
# Evaluation
# ----------------------------

def evaluate_run(name: str, pred_path: str, gold_df: pd.DataFrame) -> dict:
    pred = load_preds(pred_path)
    if len(pred) != len(gold_df):
        raise ValueError(f"Row count mismatch for {name}: test={len(gold_df)}, preds={len(pred)}")

    df = gold_df.copy()
    df["pred"] = pred.values

    gold = df["gold"].to_numpy()
    yhat = df["pred"].to_numpy()

    # Basic metrics
    acc = accuracy_score(gold, yhat)
    macro_f1 = f1_score(gold, yhat, average="macro")
    micro_f1 = f1_score(gold, yhat, average="micro")
    weighted_f1 = f1_score(gold, yhat, average="weighted")
    kappa = cohen_kappa_score(gold, yhat)

    # Per-class
    p, r, f1, support = precision_recall_fscore_support(gold, yhat, labels=[0,1], zero_division=0)
    per_class = pd.DataFrame({
        "class": [INV_LABEL_MAP[0], INV_LABEL_MAP[1]],
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support
    }).set_index("class")

    # Confusion matrix
    cm = confusion_matrix(gold, yhat, labels=[0,1])
    cm_path = OUTDIR / f"{name}_confusion.png"
    plot_confusion(cm, labels=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]],
                   title=f"Confusion Matrix — {name}", outpath=cm_path)

    # Classification report text (for quick inspection)
    clf_text = classification_report(gold, yhat, target_names=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]], zero_division=0)

    # Save tables
    summary = pd.DataFrame({
        "accuracy": [acc],
        "macro_f1": [macro_f1],
        "micro_f1": [micro_f1],
        "weighted_f1": [weighted_f1],
        "cohen_kappa": [kappa],
    }, index=[name])

    per_class_csv = OUTDIR / f"{name}_per_class.csv"
    per_class_tex = OUTDIR / f"{name}_per_class.tex"
    per_class.to_csv(per_class_csv)
    with open(per_class_tex, "w") as f:
        f.write(to_latex_table(per_class, caption=f"Per-class metrics for {name}.", label=f"tab:{name}_perclass"))

    # Save summary row as CSV and LaTeX (append later across runs)
    summary_csv = OUTDIR / f"{name}_summary.csv"
    summary_tex = OUTDIR / f"{name}_summary.tex"
    summary.to_csv(summary_csv)
    with open(summary_tex, "w") as f:
        f.write(to_latex_table(summary, caption=f"Overall metrics for {name}.", label=f"tab:{name}_summary"))

    # Save a small sample of mismatches (for qualitative error analysis)
    mismatches = df[df["gold"] != df["pred"]]
    sample_size = min(10, len(mismatches))
    sample_path = OUTDIR / f"{name}_mismatch_sample.tsv"
    if sample_size > 0:
        mm = mismatches.sample(sample_size, random_state=RANDOM_SEED).copy()
        mm["gold_label"] = mm["gold"].map(INV_LABEL_MAP)
        mm["pred_label"] = mm["pred"].map(INV_LABEL_MAP)
        mm[["text", "gold_label", "pred_label"]].to_csv(sample_path, sep="\t", index=False)

    # Pretty print
    print("="*72)
    print(f"[{name}]  Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | Micro-F1={micro_f1:.4f} "
          f"| Weighted-F1={weighted_f1:.4f} | Cohen's κ={kappa:.4f}")
    print("- Per-class:\n", per_class.round(4))
    print("- Confusion matrix:\n", cm)
    print("- Classification report:\n", clf_text)
    print(f"- Saved confusion matrix plot -> {cm_path}")
    if sample_size > 0:
        print(f"- Saved mismatch sample ({sample_size} rows) -> {sample_path}")

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "per_class": per_class,
        "cm": cm,
        "cm_path": str(cm_path)
    }

def main():
    gold_df = load_gold(TEST_PATH)

    # Evaluate each run
    all_summaries = []
    for name, pred_path in RUNS.items():
        res = evaluate_run(name, pred_path, gold_df)
        all_summaries.append(res)

    # Combine overall summary into one table for LaTeX/CSV
    summary_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k in {"name","accuracy","macro_f1","micro_f1","weighted_f1","kappa"}}
         for r in all_summaries]
    ).set_index("name").sort_index()

    summary_csv = OUTDIR / "combined_summary.csv"
    summary_tex = OUTDIR / "combined_summary.tex"
    summary_df.to_csv(summary_csv)
    with open(summary_tex, "w") as f:
        f.write(to_latex_table(
            summary_df,
            caption="Overall test metrics across runs (unmasked vs. masked).",
            label="tab:combined_summary"
        ))

    print("="*72)
    print("Combined summary:\n", summary_df.round(4))
    print(f"- Saved overall CSV -> {summary_csv}")
    print(f"- Saved LaTeX table -> {summary_tex}")
    print("Done.")

if __name__ == "__main__":
    main()




