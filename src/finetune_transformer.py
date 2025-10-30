import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from data_utils import load_splits, to_xy
import evaluate
from common import ensure_dir, set_seed


def load_dfs(train_path: str, dev_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TSV splits without headers using shared util functions."""
    tr, dv, te = load_splits(train_path, dev_path, test_path)

    X_tr, y_tr = to_xy(tr)
    X_dv, y_dv = to_xy(dv)
    X_te, y_te = to_xy(te)

    df_tr = pd.DataFrame({"text": X_tr, "label": y_tr})
    df_dv = pd.DataFrame({"text": X_dv, "label": y_dv})
    df_te = pd.DataFrame({"text": X_te, "label": y_te})

    return df_tr, df_dv, df_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model-name", default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="artifacts/transformer")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    df_tr, df_dv, df_te = load_dfs(args.train, args.dev, args.test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    ds_tr = Dataset.from_pandas(df_tr).map(tok_fn, batched=True)
    ds_dv = Dataset.from_pandas(df_dv).map(tok_fn, batched=True)
    ds_te = Dataset.from_pandas(df_te).map(tok_fn, batched=True)

    for ds in [ds_tr, ds_dv, ds_te]:
        if "text" in ds.column_names:
            ds = ds.remove_columns(["text"])
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])
        ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        score = metric_f1.compute(
            predictions=preds, references=labels, average="macro")
        return {"macro_f1": score["f1"]}

    args_tr = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        do_eval=True,
        logging_steps=50,
        save_steps=500,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tr,
        eval_dataset=ds_dv,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    dev_metrics = trainer.evaluate(eval_dataset=ds_dv)
    Path(f"{args.outdir}/dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2))

    test_metrics = trainer.evaluate(eval_dataset=ds_te)
    Path(f"{args.outdir}/test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    preds = trainer.predict(ds_te)
    hard = np.argmax(preds.predictions, axis=1)
    np.savetxt(f"{args.outdir}/test_pred.tsv", hard, fmt="%d", delimiter="\t")


if __name__ == "__main__":
    main()
