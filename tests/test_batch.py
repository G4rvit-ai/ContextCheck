import pandas as pd
import pytest

from src.utils import batch_predict_dataframe


class FakePredictor:
    def predict_many(self, pairs):
        return [
            {
                "label": "entailment",
                "score": 0.8,
                "scores": {
                    "entailment": 0.8,
                    "neutral": 0.1,
                    "contradiction": 0.1,
                },
            }
            for _ in pairs
        ]


def test_batch_predicts_valid_rows_and_marks_invalid_rows():
    frame = pd.DataFrame(
        {
            "premise": ["A premise.", None],
            "hypothesis": ["A statement.", "Missing premise."],
        }
    )
    result = batch_predict_dataframe(frame, FakePredictor())
    assert result.loc[0, "predicted_label"] == "entailment"
    assert result.loc[0, "status"] == "ok"
    assert result.loc[0, "score_entailment"] == pytest.approx(0.8)
    assert result.loc[1, "predicted_label"] == "INVALID"
    assert result.loc[1, "status"].startswith("Invalid:")


def test_batch_requires_both_text_columns():
    with pytest.raises(ValueError, match="premise"):
        batch_predict_dataframe(pd.DataFrame({"premise": ["text"]}), FakePredictor())


def test_batch_rejects_incomplete_predictor_output():
    class IncompletePredictor:
        def predict_many(self, pairs):
            return []

    frame = pd.DataFrame({"premise": ["text"], "hypothesis": ["statement"]})
    with pytest.raises(RuntimeError, match="unexpected number"):
        batch_predict_dataframe(frame, IncompletePredictor())
