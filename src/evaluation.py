import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)


def evaluate_predictions(y_true, y_pred, scores=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            results["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            results["roc_auc"] = np.nan

        try:
            results["pr_auc"] = average_precision_score(y_true, scores)
        except Exception:
            results["pr_auc"] = np.nan
    else:
        results["roc_auc"] = np.nan
        results["pr_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    return results, cm


def metrics_to_dataframe(metrics_dict):
    return pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "model"})