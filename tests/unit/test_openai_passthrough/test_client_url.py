"""Tests for upstream_url — guards against the httpx RFC 3986 path-replacement footgun.

If you ever set ``base_url`` on the AsyncClient and pass a leading-slash path,
httpx will silently drop the ``/v1`` from the base. This test family asserts
that ``upstream_url`` always produces a fully-qualified URL with both the
configured ``OPENAI_BASE_URL`` path AND the request path joined intact.
"""
from app.api.openai_passthrough.client import upstream_url


def test_includes_base_url_path_with_leading_slash(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.openai_base_url",
        "https://bedrock-mantle.us-west-2.api.aws/v1",
    )
    # The bug: with httpx base_url=".../v1", "/chat/completions" would drop "/v1".
    # upstream_url must keep both segments.
    assert (
        upstream_url("/chat/completions")
        == "https://bedrock-mantle.us-west-2.api.aws/v1/chat/completions"
    )


def test_includes_base_url_path_without_leading_slash(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.openai_base_url",
        "https://bedrock-mantle.us-west-2.api.aws/v1",
    )
    assert (
        upstream_url("models")
        == "https://bedrock-mantle.us-west-2.api.aws/v1/models"
    )


def test_strips_trailing_slash_from_base(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.openai_base_url",
        "https://bedrock-mantle.us-west-2.api.aws/v1/",
    )
    assert (
        upstream_url("/responses")
        == "https://bedrock-mantle.us-west-2.api.aws/v1/responses"
    )


def test_works_with_response_id_in_path(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.openai_base_url",
        "https://bedrock-mantle.us-west-2.api.aws/v1",
    )
    assert (
        upstream_url("/responses/resp-123/cancel")
        == "https://bedrock-mantle.us-west-2.api.aws/v1/responses/resp-123/cancel"
    )


def test_works_with_base_url_no_path_segment(monkeypatch):
    """Some clients might point at a domain root; still produce a sensible URL."""
    monkeypatch.setattr(
        "app.core.config.settings.openai_base_url",
        "https://example.com",
    )
    assert upstream_url("/models") == "https://example.com/models"
