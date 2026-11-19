"""Shared paths and data helpers for the Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "contextcheck_distilbert"
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.json"


def is_model_ready(model_dir: str | Path = MODEL_DIR) -> bool:
    """Return whether a directory contains the minimum saved-model artifacts."""
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return False
    has_weights = (model_path / "pytorch_model.bin").is_file() or any(
        model_path.glob("*.safetensors")
    )
    has_tokenizer = (model_path / "tokenizer.json").is_file() or (
        (model_path / "vocab.txt").is_file()
        and (model_path / "tokenizer_config.json").is_file()
    )
    return (
        (model_path / "config.json").is_file()
        and has_weights
        and has_tokenizer
    )


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

    if valid_pairs:
        predictions = predictor.predict_many(valid_pairs)
        if len(predictions) != len(valid_pairs):
            raise RuntimeError(
                "The predictor returned an unexpected number of results "
                f"({len(predictions)} for {len(valid_pairs)} valid rows)."
            )
        expected_scores = {"entailment", "neutral", "contradiction"}
        for position, prediction in zip(valid_positions, predictions):
            required_prediction_keys = {"label", "score", "scores"}
            if not required_prediction_keys.issubset(prediction):
                raise RuntimeError("The predictor returned an incomplete prediction result.")
            if set(prediction["scores"]) != expected_scores:
                raise RuntimeError("The predictor returned an invalid score mapping.")
            result.iloc[position, result.columns.get_loc("predicted_label")] = prediction["label"]
            result.iloc[position, result.columns.get_loc("model_score")] = prediction["score"]
            for label, score in prediction["scores"].items():
                result.iloc[position, result.columns.get_loc(f"score_{label}")] = score
            result.iloc[position, result.columns.get_loc("status")] = "ok"
    return result
