import pandas as pd
import random

# Load data without headers
test_df = pd.read_csv("data/test.tsv", sep="\t",
                      header=None, names=["text", "label"])
pred_df = pd.read_csv("artifacts/transformer/test_pred.tsv",
                      sep="\t", header=None, names=["pred"])

# Map textual gold labels (NOT/OFF) to numeric values
label_map = {"NOT": 0, "OFF": 1}
test_df["gold"] = test_df["label"].astype(str).str.strip().map(label_map)

# Strip and convert predictions to int
pred_df["pred"] = pred_df["pred"].astype(str).str.strip().astype(int)

# Check for missing label mappings
if test_df["gold"].isna().any():
    invalid_labels = test_df.loc[test_df["gold"].isna(), "label"].unique()
    raise ValueError(f"Unexpected label values in test.tsv: {invalid_labels}")

# Align data lengths
if len(test_df) != len(pred_df):
    raise ValueError(
        f"Row count mismatch: test={len(test_df)}, preds={len(pred_df)}")

# Merge predictions
test_df["pred"] = pred_df["pred"]

# Identify mismatches
mismatches = test_df[test_df["gold"] != test_df["pred"]]

# Display random sample of mismatches for error analysis
sample_size = min(10, len(mismatches))
if sample_size == 0:
    print("No mismatches found.")
else:
    sample_rows = mismatches.sample(
        sample_size, random_state=random.randint(0, 9999))
    for i, row in sample_rows.iterrows():
        gold_label = "OFF" if row["gold"] == 1 else "NOT"
        pred_label = "OFF" if row["pred"] == 1 else "NOT"
        print(f"Index: {i}")
        print(f"Text: {row['text']}")
        print(f"Gold: {gold_label}, Pred: {pred_label}\n")
