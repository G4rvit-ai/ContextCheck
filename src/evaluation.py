"""Metric calculation and JSON persistence for measured model evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.labels import LABEL_NAMES


def compute_classification_results(y_true, y_pred) -> dict:
    """Return standard metrics and matrices for labels in canonical order."""
    labels = list(range(len(LABEL_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(LABEL_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_results(path: str | Path, results: dict) -> None:
    """Write JSON-safe metrics, converting NumPy values if present."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    target.write_text(json.dumps(convert(results), indent=2), encoding="utf-8")
