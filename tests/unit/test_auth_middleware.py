"""Tests for AuthMiddleware transparent-proxy mode.

Verifies that when ``TRANSPARENT_PROXY=true`` the middleware:
- skips DDB / master-key validation,
- extracts the client key from x-api-key (or Authorization: Bearer),
- stashes it on ``request.state.api_key_info`` for the backend to relay.

And that with the flag off (default), the original validation path runs.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from app.core.config import settings
from app.middleware.auth import AuthMiddleware


def _build_app():
    """Minimal Starlette app with AuthMiddleware and an echo endpoint that
    returns request.state.api_key_info as JSON."""
    app = Starlette()

    async def echo(request):
        info = getattr(request.state, "api_key_info", None)
        return JSONResponse({"info": info})

    app.add_route("/v1/messages", echo, methods=["POST"])
    # AuthMiddleware requires a dynamodb_client; pass a MagicMock — in
    # transparent mode the APIKeyManager is never constructed/used.
    app.add_middleware(AuthMiddleware, dynamodb_client=MagicMock())
    return app


def _patch_settings(**overrides):
    """Patch settings attributes for the duration of a test."""
    return patch.multiple(settings, **overrides)


def test_transparent_mode_relays_x_api_key_without_validation():
    with _patch_settings(transparent_proxy=True, require_api_key=True, master_api_key=""):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-client-123"},
            json={},
        )
    assert resp.status_code == 200
    info = resp.json()["info"]
    assert info is not None
    assert info["api_key"] == "sk-client-123"
    assert info["user_id"] == "transparent"
    assert info["is_master"] is False
    assert info["rate_limit"] is None


def test_transparent_mode_accepts_authorization_bearer():
    with _patch_settings(transparent_proxy=True, require_api_key=True, master_api_key=""):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-bearer-456"},
            json={},
        )
    assert resp.status_code == 200
    assert resp.json()["info"]["api_key"] == "sk-bearer-456"


def test_transparent_mode_x_api_key_takes_precedence_over_bearer():
    with _patch_settings(transparent_proxy=True, require_api_key=True, master_api_key=""):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-from-xapikey",
                "Authorization": "Bearer sk-from-bearer",
            },
            json={},
        )
    assert resp.status_code == 200
    assert resp.json()["info"]["api_key"] == "sk-from-xapikey"


def test_transparent_mode_rejects_missing_key():
    with _patch_settings(transparent_proxy=True, require_api_key=True, master_api_key=""):
        app = _build_app()
        client = TestClient(app)
        resp = client.post("/v1/messages", json={})
    assert resp.status_code == 401
    err = resp.json()
    assert err["error"]["type"] == "authentication_error"


def test_transparent_mode_skips_ddb_validation():
    """A key that is NOT in DDB and NOT the master key must still pass —
    proving DDB/master validation is bypassed."""
    with _patch_settings(transparent_proxy=True, require_api_key=True, master_api_key="master-secret"):
        # Even though the APIKeyManager would reject this key, transparent
        # mode never calls it.
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-not-in-ddb"},
            json={},
        )
    assert resp.status_code == 200
    assert resp.json()["info"]["api_key"] == "sk-not-in-ddb"


def test_non_transparent_mode_runs_validation():
    """Regression: with transparent_proxy off, the master key is honoured
    (proving the original validation path still runs and the transparent
    shortcut is skipped)."""
    with _patch_settings(
        transparent_proxy=False,
        require_api_key=True,
        master_api_key="master-secret",
        api_key_header="x-api-key",
    ):
        app = _build_app()
        client = TestClient(app)
        resp_ok = client.post(
            "/v1/messages", headers={"x-api-key": "master-secret"}, json={}
        )
        assert resp_ok.status_code == 200
        assert resp_ok.json()["info"]["is_master"] is True
        assert resp_ok.json()["info"]["user_id"] == "master"

