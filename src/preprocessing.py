"""Tokenization and deterministic subset helpers shared by training code."""

from __future__ import annotations

from typing import Any

from src.labels import validate_label


def validate_text_pair(premise: Any, hypothesis: Any) -> tuple[str, str]:
    """Validate and normalize a supplied premise/hypothesis pair."""
    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("Premise must contain non-empty text.")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("Hypothesis must contain non-empty text.")
    return premise.strip(), hypothesis.strip()


def tokenize_pairs(batch: dict[str, list], tokenizer, max_length: int = 128):
    """Tokenize paired sentences; padding is deferred to the batch collator."""
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=max_length,
    )


def validate_dataset_labels(labels) -> None:
    """Raise if any dataset label does not follow the SNLI class mapping."""
    for label in labels:
        validate_label(label)


def select_train_subset(dataset, max_samples: int | None, seed: int = 42):
    """Select a repeatable training-only subset, leaving other splits untouched."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    return dataset.shuffle(seed=seed).select(range(max_samples))
