"""Load a saved fine-tuned model and predict relationships between texts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.labels import ID_TO_LABEL, validate_label
from src.preprocessing import validate_text_pair


class NliPredictor:
    """Small inference wrapper supporting CPU, CUDA, and Apple Silicon MPS."""

    def __init__(self, model_dir: str | Path, max_length: int = 128):
        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. "
                "Run `python -m src.train_transformer` first."
            )
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        if int(self.model.config.num_labels) != 3:
            raise ValueError("The saved model must have exactly three output classes.")
        self.model.to(self.device)
        self.model.eval()

    def predict_many(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Return label and model-derived softmax scores for each text pair."""
        if not pairs:
            return []
        normalized = [validate_text_pair(*pair) for pair in pairs]
        premises, hypotheses = zip(*normalized)
        inputs = self.tokenizer(
            list(premises),
            list(hypotheses),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            probabilities = softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()

        results = []
        for scores in probabilities:
            prediction = validate_label(max(range(len(scores)), key=scores.__getitem__))
            results.append(
                {
                    "label": ID_TO_LABEL[prediction],
                    "label_id": prediction,
                    "score": float(scores[prediction]),
                    "scores": {
                        ID_TO_LABEL[index]: float(score)
                        for index, score in enumerate(scores)
                    },
                }
            )
        return results

    def predict(self, premise: str, hypothesis: str) -> dict:
        """Predict one premise/hypothesis pair."""
        return self.predict_many([(premise, hypothesis)])[0]
