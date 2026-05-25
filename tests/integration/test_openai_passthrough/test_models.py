"""Integration test for GET /openai/v1/models — pure passthrough."""
import httpx


def test_list_models_forwards(client, respx_mock):
    upstream = {
        "object": "list",
        "data": [
            {"id": "openai.gpt-oss-120b", "object": "model"},
            {"id": "us.anthropic.claude-sonnet-4-6", "object": "model"},
        ],
    }
    respx_mock.get("/models").mock(return_value=httpx.Response(200, json=upstream))

    r = client.get("/openai/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200
    assert r.json() == upstream
