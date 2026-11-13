import pytest
import torch

import src.inference as inference
from src.inference import NliPredictor


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_transformer"):
        NliPredictor(tmp_path / "not-trained")


def test_prediction_returns_one_valid_class_and_normalized_scores(tmp_path, monkeypatch):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, premises, hypotheses, **kwargs):
            return {"input_ids": torch.ones((len(premises), 2), dtype=torch.long)}

    class FakeModel:
        config = type("Config", (), {"num_labels": 3})()

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return type(
                "Output",
                (),
                {"logits": torch.tensor([[2.0, 1.0, 0.0]] * len(inputs["input_ids"]))},
            )()

    monkeypatch.setattr(inference, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(inference, "AutoModelForSequenceClassification", FakeModel)
    predictor = NliPredictor(tmp_path)
    prediction = predictor.predict("Context.", "Statement.")

    assert prediction["label"] in {"entailment", "neutral", "contradiction"}
    assert set(prediction["scores"]) == {"entailment", "neutral", "contradiction"}
    assert sum(prediction["scores"].values()) == pytest.approx(1.0)
    assert prediction["label"] == "entailment"
