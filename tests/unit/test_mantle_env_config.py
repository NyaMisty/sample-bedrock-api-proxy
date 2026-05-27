"""Configuration tests for Bedrock Mantle environment variable names."""

from app.core.config import Settings


def _clear_mantle_env(monkeypatch):
    for name in (
        "BEDROCK_API_KEY",
        "MANTLE_ENDPOINT_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_reads_bedrock_mantle_env_names(monkeypatch):
    _clear_mantle_env(monkeypatch)
    monkeypatch.setenv("BEDROCK_API_KEY", "bedrock-key-new")
    monkeypatch.setenv("MANTLE_ENDPOINT_URL", "https://mantle.example/v1")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "bedrock-key-new"
    assert settings.openai_base_url == "https://mantle.example/v1"


def test_settings_keeps_legacy_openai_env_fallback(monkeypatch):
    _clear_mantle_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "bedrock-key-legacy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy-mantle.example/v1")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "bedrock-key-legacy"
    assert settings.openai_base_url == "https://legacy-mantle.example/v1"
