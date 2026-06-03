from app.core.config import settings


def test_default_model_mapping_includes_gpt_5_aliases():
    assert settings.default_model_mapping["gpt-5.5"] == "openai.gpt-5.5"
    assert settings.default_model_mapping["gpt-5.4"] == "openai.gpt-5.4"
