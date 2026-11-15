"""Load a saved fine-tuned model and predict relationships between texts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.labels import ID_TO_LABEL, validate_label
from src.preprocessing import validate_text_pair
from src.utils import is_model_ready


class NliPredictor:
    """Small inference wrapper supporting CPU, CUDA, and Apple Silicon MPS."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 128,
        max_batch_size: int = 32,
    ):
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive.")
        model_path = Path(model_dir)
        if not is_model_ready(model_path):
            raise FileNotFoundError(
                f"Complete fine-tuned model artifacts were not found at {model_path}. "
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
        self.max_batch_size = max_batch_size
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
        if len(pairs) > self.max_batch_size:
            results = []
            for start in range(0, len(pairs), self.max_batch_size):
                results.extend(self.predict_many(pairs[start : start + self.max_batch_size]))
            return results
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
            logits = self.model(**inputs).logits
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise RuntimeError(
                    "The saved model returned an invalid output shape; expected "
                    f"(batch, 3), received {tuple(logits.shape)}."
                )
            probabilities = softmax(logits, dim=-1).cpu().tolist()

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
