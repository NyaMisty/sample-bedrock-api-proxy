"""Configuration tests for the BACKEND_MODE feature."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _clear_backend_env(monkeypatch):
    for name in (
        "BACKEND_MODE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "BEDROCK_API_KEY",
        "MANTLE_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_backend_mode_defaults_to_bedrock(monkeypatch):
    _clear_backend_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.backend_mode == "bedrock"
    assert settings.anthropic_base_url == "https://api.anthropic.com"


def test_backend_mode_rejects_invalid_value(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_MODE", "gemini")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_anthropic_mode_requires_anthropic_api_key(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_MODE", "anthropic")
    # No ANTHROPIC_API_KEY set → validation must fail.
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_anthropic_mode_accepts_when_key_present(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_MODE", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings(_env_file=None)
    assert settings.backend_mode == "anthropic"
    assert settings.anthropic_api_key == "sk-ant-test"


def test_openai_mode_requires_openai_credentials(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_MODE", "openai")
    # No OPENAI_API_KEY / OPENAI_BASE_URL → validation must fail.
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_mode_accepts_when_credentials_present(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://mantle.example/v1")
    settings = Settings(_env_file=None)
    assert settings.backend_mode == "openai"
