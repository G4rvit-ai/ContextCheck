from src.dataset_loader import is_usable_example


def test_dataset_filter_removes_unlabeled_and_unusable_rows():
    assert is_usable_example({"premise": "A", "hypothesis": "B", "label": 0})
    assert not is_usable_example({"premise": "A", "hypothesis": "B", "label": -1})
    assert not is_usable_example({"premise": "", "hypothesis": "B", "label": 1})
    assert not is_usable_example({"premise": "A", "hypothesis": None, "label": 2})
    assert not is_usable_example({"premise": "A", "hypothesis": "B", "label": None})
