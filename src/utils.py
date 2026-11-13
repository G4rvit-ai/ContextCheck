"""Shared paths and data helpers for the Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "contextcheck_distilbert"
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.json"


def batch_predict_dataframe(dataframe: pd.DataFrame, predictor) -> pd.DataFrame:
    """Predict valid CSV rows and mark invalid records without stopping the batch."""
    required = {"premise", "hypothesis"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}.")

    result = dataframe.copy()
    valid_positions = []
    valid_pairs = []
    status = [""] * len(result)
    for position, row in enumerate(result[["premise", "hypothesis"]].itertuples(index=False)):
        premise, hypothesis = row
        if (
            not isinstance(premise, str)
            or not premise.strip()
            or not isinstance(hypothesis, str)
            or not hypothesis.strip()
        ):
            status[position] = "Invalid: premise and hypothesis are required."
        else:
            valid_positions.append(position)
            valid_pairs.append((premise, hypothesis))

    result["predicted_label"] = "INVALID"
    result["model_score"] = float("nan")
    for label in ("entailment", "neutral", "contradiction"):
        result[f"score_{label}"] = float("nan")
    result["status"] = status

    predictions = predictor.predict_many(valid_pairs)
    for position, prediction in zip(valid_positions, predictions):
        result.iloc[position, result.columns.get_loc("predicted_label")] = prediction["label"]
        result.iloc[position, result.columns.get_loc("model_score")] = prediction["score"]
        for label, score in prediction["scores"].items():
            result.iloc[position, result.columns.get_loc(f"score_{label}")] = score
        result.iloc[position, result.columns.get_loc("status")] = "ok"
    return result
