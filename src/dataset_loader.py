"""Load and clean the official SNLI splits without mixing their roles."""

from __future__ import annotations

from typing import Any


def is_usable_example(example: dict[str, Any]) -> bool:
    """Reject unlabeled rows and rows with missing or blank sentence text."""
    label = example.get("label")
    premise = example.get("premise")
    hypothesis = example.get("hypothesis")
    return (
        label is not None
        and label in (0, 1, 2)
        and isinstance(premise, str)
        and bool(premise.strip())
        and isinstance(hypothesis, str)
        and bool(hypothesis.strip())
    )


def load_snli(clean: bool = True):
    """Load SNLI's official train, validation, and test splits from Hub.

    The Hugging Face ``datasets`` dependency is imported lazily so other
    application functionality (including inference) does not depend on it.
    """
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        raise RuntimeError(
            "The 'datasets' package is required to load SNLI. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset("stanfordnlp/snli")
    required_splits = {"train", "validation", "test"}
    missing = required_splits.difference(dataset.keys())
    if missing:
        raise ValueError(f"SNLI is missing expected splits: {sorted(missing)}")

    print("SNLI feature structure:")
    print(dataset["train"].features)
    print("Rows before cleaning:", {name: len(dataset[name]) for name in required_splits})

    if not clean:
        return dataset, {}

    cleaned = {}
    statistics = {}
    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]
        original_count = len(split)
        cleaned[split_name] = split.filter(is_usable_example)
        retained_count = len(cleaned[split_name])
        statistics[split_name] = {
            "original_rows": original_count,
            "retained_rows": retained_count,
            "removed_rows": original_count - retained_count,
        }

    print("Rows after cleaning:", {
        name: len(cleaned[name]) for name in required_splits
    })
    return cleaned, statistics
