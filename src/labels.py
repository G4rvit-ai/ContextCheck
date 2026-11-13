"""SNLI's label IDs and the user-facing label names."""

LABEL_NAMES = ("entailment", "neutral", "contradiction")
LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = dict(enumerate(LABEL_NAMES))


def validate_label(label: int) -> int:
    """Return a valid SNLI class ID or raise a clear error."""
    label = int(label)
    if label not in ID_TO_LABEL:
        raise ValueError(f"Expected a class ID from 0 to 2; received {label}.")
    return label
