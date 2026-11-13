import pytest

from src.labels import ID_TO_LABEL, LABEL_TO_ID, validate_label
from src.preprocessing import tokenize_pairs, validate_text_pair


def test_snli_label_mapping_is_canonical():
    assert LABEL_TO_ID == {"entailment": 0, "neutral": 1, "contradiction": 2}
    assert ID_TO_LABEL == {0: "entailment", 1: "neutral", 2: "contradiction"}


def test_validate_label_rejects_unlabeled_and_out_of_range_values():
    assert validate_label(2) == 2
    with pytest.raises(ValueError):
        validate_label(-1)
    with pytest.raises(ValueError):
        validate_label(3)


@pytest.mark.parametrize(
    ("premise", "hypothesis"),
    [(None, "statement"), ("context", ""), ("  ", "statement")],
)
def test_text_pair_requires_two_non_empty_strings(premise, hypothesis):
    with pytest.raises(ValueError):
        validate_text_pair(premise, hypothesis)


def test_validate_text_pair_strips_outer_whitespace():
    assert validate_text_pair(" context ", " statement ") == ("context", "statement")


def test_pair_tokenization_uses_truncation_without_static_padding():
    class FakeTokenizer:
        def __call__(self, premise, hypothesis, **kwargs):
            return {"premise": premise, "hypothesis": hypothesis, **kwargs}

    encoded = tokenize_pairs(
        {"premise": ["p"], "hypothesis": ["h"]}, FakeTokenizer(), max_length=64
    )
    assert encoded["truncation"] is True
    assert encoded["max_length"] == 64
    assert "padding" not in encoded
