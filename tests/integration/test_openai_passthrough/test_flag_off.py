"""Verify that /openai/v1/* paths return 404 when ENABLE_OPENAI_PASSTHROUGH is off."""
import importlib

from fastapi.testclient import TestClient


def test_flag_off_returns_404(monkeypatch):
    """With ENABLE_OPENAI_PASSTHROUGH=False, /openai/v1/* paths must not exist."""
    monkeypatch.setattr("app.core.config.settings.enable_openai_passthrough", False)
    monkeypatch.setattr("app.core.config.settings.require_api_key", False)

    # Reload main so the conditional mount re-evaluates with the flag off.
    import app.main as _main
    importlib.reload(_main)

    client = TestClient(_main.app)
    r = client.post(
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "x", "messages": []},
    )
    assert r.status_code == 404, f"expected 404 with flag off, got {r.status_code}"

    r = client.get("/openai/v1/models")
    assert r.status_code == 404


def test_flag_off_does_not_register_routes(monkeypatch):
    """Programmatic verification: no route paths under /openai/v1 when flag is off."""
    monkeypatch.setattr("app.core.config.settings.enable_openai_passthrough", False)

    import app.main as _main
    importlib.reload(_main)

    extra = [getattr(r, "path", "") for r in _main.app.routes
             if getattr(r, "path", "").startswith("/openai/v1")]
    assert not extra, f"unexpected routes registered: {extra}"
