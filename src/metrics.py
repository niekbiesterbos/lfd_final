from typing import Sequence, Tuple, Dict
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro-averaged F1 for binary classification."""
    return f1_score(y_true, y_pred, average="macro")


def full_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    """Return a structured dict with macro-F1, confusion matrix and a textual classification report."""
    f1 = macro_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report_txt = classification_report(y_true, y_pred, digits=4)
    return {"macro_f1": float(f1), "confusion_matrix": cm, "report": report_txt}
