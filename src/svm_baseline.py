import argparse
import json
from pathlib import Path
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_utils import load_splits, to_xy
from metrics import full_report
from common import ensure_dir, set_seed


def build_pipeline(ngram_max: int, min_df: int, use_prob: bool) -> Pipeline:
    """Create a TF-IDF + linear SVM pipeline; optional probability via calibration."""
    vec = TfidfVectorizer(ngram_range=(1, ngram_max),
                          min_df=min_df, lowercase=True, strip_accents="unicode")
    # Balanced handles label skew robustly
    base = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base) if use_prob else base
    return Pipeline([("tfidf", vec), ("clf", clf)])


def fit_eval(pipe: Pipeline, X_tr, y_tr, X_dv, y_dv) -> Tuple[dict, np.ndarray]:
    """Train pipeline and return dev metrics and predictions."""
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_dv)
    metrics = full_report(y_dv, y_pred)
    return metrics, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="artifacts/svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--use-prob", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    tr, dv, te = load_splits(args.train, args.dev,
                             args.test)
    X_tr, y_tr = to_xy(tr)
    X_dv, y_dv = to_xy(dv)
    X_te, y_te = to_xy(te)

    pipe = build_pipeline(args.ngram_max, args.min_df, args.use_prob)

    dev_metrics, _ = fit_eval(pipe, X_tr, y_tr, X_dv, y_dv)
    test_pred = pipe.predict(X_te)
    test_metrics = full_report(y_te, test_pred)

    Path(f"{args.outdir}/dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2))
    Path(f"{args.outdir}/test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.savetxt(f"{args.outdir}/test_pred.tsv",
               test_pred, fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
