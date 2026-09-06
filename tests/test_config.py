from pathlib import Path

from capability_capsule.config import Settings


def test_example_configuration_is_valid() -> None:
    settings = Settings.from_toml(Path("config.example.toml"))

    assert str(settings.ollama.base_url) == "http://127.0.0.1:11434/"
    assert settings.ollama.generation_model == "qwen3.5:4b"
    assert settings.ollama.embedding_model == "nomic-embed-text"
    assert settings.http.trust_env is False
    assert settings.capsule.size_budget_bytes == 1_073_741_824
