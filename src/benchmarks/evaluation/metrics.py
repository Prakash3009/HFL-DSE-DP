"""Multiclass E1 metrics with explicit averaging conventions and CM checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_multiclass(y_true, probabilities, class_names):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n_classes = len(class_names)
    labels = np.arange(n_classes)
    if probabilities.shape != (len(y_true), n_classes):
        raise ValueError(f"Expected probability shape {(len(y_true), n_classes)}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite class probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Softmax probabilities do not sum to one")
    if set(np.unique(y_true)) != set(labels.tolist()):
        raise ValueError("Every mapped class must be present in the evaluated split")

    predictions = probabilities.argmax(axis=1)
    cm = confusion_matrix(y_true, predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average=None, zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average="weighted", zero_division=0,
    )
    accuracy = float(np.trace(cm) / cm.sum())
    macro_f1 = float(np.mean(f1))
    mcc = float(matthews_corrcoef(y_true, predictions))

    # Independently reconstruct confusion-matrix-based metrics.
    cm_support = cm.sum(axis=1)
    predicted_support = cm.sum(axis=0)
    cm_tp = np.diag(cm).astype(np.float64)
    cm_precision = np.divide(cm_tp, predicted_support, out=np.zeros_like(cm_tp), where=predicted_support != 0)
    cm_recall = np.divide(cm_tp, cm_support, out=np.zeros_like(cm_tp), where=cm_support != 0)
    cm_f1 = np.divide(2 * cm_precision * cm_recall, cm_precision + cm_recall,
                      out=np.zeros_like(cm_tp), where=(cm_precision + cm_recall) != 0)
    cm_weight = cm_support / cm_support.sum()
    cm_accuracy = float(cm_tp.sum() / cm.sum())
    cm_weighted_precision = float(np.dot(cm_weight, cm_precision))
    cm_weighted_recall = float(np.dot(cm_weight, cm_recall))
    cm_weighted_f1 = float(np.dot(cm_weight, cm_f1))
    cm_macro_f1 = float(cm_f1.mean())
    c = cm.sum()
    numerator = c * cm_tp.sum() - float(np.dot(cm_support, predicted_support))
    denominator = np.sqrt(
        (c * c - float(np.dot(predicted_support, predicted_support)))
        * (c * c - float(np.dot(cm_support, cm_support)))
    )
    cm_mcc = float(numerator / denominator) if denominator else 0.0
    checks = [
        (accuracy, cm_accuracy, "accuracy"),
        (float(weighted_precision), cm_weighted_precision, "weighted precision"),
        (float(weighted_recall), cm_weighted_recall, "weighted recall"),
        (float(weighted_f1), cm_weighted_f1, "weighted F1"),
        (macro_f1, cm_macro_f1, "macro F1"),
        (mcc, cm_mcc, "multiclass MCC"),
    ]
    for observed, reconstructed, metric in checks:
        if not np.isclose(observed, reconstructed, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"{metric} disagrees with confusion-matrix reconstruction: {observed} vs {reconstructed}")

    per_class_roc, per_class_pr = [], []
    for class_id in labels:
        binary_true = (y_true == class_id).astype(np.int8)
        if len(np.unique(binary_true)) != 2:
            raise ValueError(f"Class {class_names[class_id]} is absent or universal; one-vs-rest AUC undefined")
        per_class_roc.append(float(roc_auc_score(binary_true, probabilities[:, class_id])))
        pr_precision, pr_recall, _ = precision_recall_curve(binary_true, probabilities[:, class_id])
        per_class_pr.append(float(auc(pr_recall, pr_precision)))

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "macro_f1": macro_f1,
        "mcc_multiclass": mcc,
        "roc_auc_ovr_macro": float(np.mean(per_class_roc)),
        "pr_auc_ovr_macro_trapezoidal": float(np.mean(per_class_pr)),
    }
    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class_id": i, "class_name": name, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            "roc_auc_ovr": per_class_roc[i], "pr_auc_ovr_trapezoidal": per_class_pr[i],
        })
    return metrics, cm.astype(np.int64), per_class, predictions
