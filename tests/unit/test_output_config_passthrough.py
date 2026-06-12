"""Regression tests: output_config (and context_management on the standalone
path) must be forwarded to Bedrock on the standalone code execution loop and
PTC continuation/finalization paths."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.anthropic import MessageRequest, MessageResponse, Usage
from app.schemas.ptc import PTCExecutionState
from app.services.ptc.sandbox import ExecutionResult


def _end_turn_response(model: str) -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        content=[{"type": "text", "text": "done"}],
        model=model,
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _fake_session():
    return SimpleNamespace(
        session_id="sess-test",
        expires_at=datetime.now() + timedelta(minutes=5),
    )


def test_ptc_execution_state_preserves_original_output_config():
    state = PTCExecutionState(
        session_id="sess-test",
        code_execution_tool_id="srvtoolu_test",
        original_output_config={"effort": "max"},
    )
    assert state.original_output_config == {"effort": "max"}


async def test_standalone_loop_forwards_output_config_and_context_management():
    from app.services.standalone_code_execution_service import (
        StandaloneCodeExecutionService,
    )

    service = StandaloneCodeExecutionService()
    captured = []

    async def fake_invoke(req, *args, **kwargs):
        captured.append(req)
        return _end_turn_response(req.model)

    context_management = {"edits": [{"type": "context_compaction_20260112"}]}
    request = MessageRequest(
        model="claude-sonnet-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        output_config={"effort": "low"},
        context_management=context_management,
    )

    with patch.object(
        service, "_get_or_create_session", AsyncMock(return_value=_fake_session())
    ):
        await service.handle_request(
            request,
            SimpleNamespace(invoke_model=fake_invoke),
            request_id="req-test",
            service_tier="standard",
        )

    assert captured, "Bedrock was never called"
    assert captured[0].output_config == {"effort": "low"}
    assert captured[0].context_management == context_management


async def test_ptc_finalize_forwards_output_config_from_state():
    from app.services.ptc_service import PTCService

    service = PTCService()
    captured = []

    async def fake_invoke(req, *args, **kwargs):
        captured.append(req)
        return _end_turn_response(req.model)

    original_request = MessageRequest(
        model="claude-sonnet-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "compute"}],
    )
    execution_state = PTCExecutionState(
        session_id="sess-test",
        code_execution_tool_id="srvtoolu_test",
        original_model="claude-sonnet-4-5",
        original_max_tokens=128,
        # State carries the output_config; original_request has none —
        # the continuation must use the preserved value.
        original_output_config={"effort": "max"},
        original_assistant_content=[
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "execute_code",
                "input": {"code": "print(42)"},
            }
        ],
        original_execute_code_id="toolu_test",
    )

    await service._finalize_code_execution(
        result=ExecutionResult(success=True, stdout="42", stderr="", return_code=0),
        code_execution_tool_id="srvtoolu_test",
        original_request=original_request,
        bedrock_service=SimpleNamespace(invoke_model=fake_invoke),
        request_id="req-test",
        service_tier="standard",
        session=_fake_session(),
        ptc_callable_tools=[],
        code="print(42)",
        execution_state=execution_state,
    )

    assert captured, "Bedrock was never called"
    assert captured[0].output_config == {"effort": "max"}


async def test_ptc_finalize_falls_back_to_request_output_config():
    from app.services.ptc_service import PTCService

    service = PTCService()
    captured = []

    async def fake_invoke(req, *args, **kwargs):
        captured.append(req)
        return _end_turn_response(req.model)

    original_request = MessageRequest(
        model="claude-sonnet-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "compute"}],
        output_config={"effort": "low"},
    )

    await service._finalize_code_execution(
        result=ExecutionResult(success=True, stdout="42", stderr="", return_code=0),
        code_execution_tool_id="srvtoolu_test",
        original_request=original_request,
        bedrock_service=SimpleNamespace(invoke_model=fake_invoke),
        request_id="req-test",
        service_tier="standard",
        session=_fake_session(),
        ptc_callable_tools=[],
        code="print(42)",
        execution_state=None,
    )

    assert captured, "Bedrock was never called"
    assert captured[0].output_config == {"effort": "low"}
