from src.utils import is_model_ready


def test_model_ready_requires_config_weights_and_tokenizer(tmp_path):
    assert not is_model_ready(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    assert not is_model_ready(tmp_path)
    (tmp_path / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert is_model_ready(tmp_path)
