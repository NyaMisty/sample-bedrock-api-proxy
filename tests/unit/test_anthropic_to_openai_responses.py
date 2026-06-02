"""
Unit tests for AnthropicToOpenAIResponsesConverter.

Verifies conversion of Anthropic MessageRequest objects into kwargs dicts
suitable for the OpenAI SDK ``client.responses.create(**kwargs)`` call.
"""
import json

from app.converters.anthropic_to_openai_responses import (
    AnthropicToOpenAIResponsesConverter,
)
from app.schemas.anthropic import (
    Base64ImageSource,
    ImageContent,
    Message,
    MessageRequest,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolInputSchema,
    ToolResultContent,
    ToolUseContent,
)


def _converter() -> AnthropicToOpenAIResponsesConverter:
    return AnthropicToOpenAIResponsesConverter()


def test_plain_user_text_string():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=1024,
        messages=[Message(role="user", content="Hello there")],
    )
    result = _converter().convert_request(request)

    assert result["model"] == "openai.gpt-5.5"
    assert result["store"] is False
    assert result["max_output_tokens"] == 1024
    assert result["input"] == [{"role": "user", "content": "Hello there"}]
    assert "tools" not in result
    assert "instructions" not in result
    assert "tool_choice" not in result


def test_system_string_becomes_instructions():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        system="You are helpful.",
        messages=[Message(role="user", content="Hi")],
    )
    result = _converter().convert_request(request)
    assert result["instructions"] == "You are helpful."


def test_system_list_of_blocks_joined():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        system=[
            SystemMessage(text="Part one."),
            SystemMessage(text="Part two."),
        ],
        messages=[Message(role="user", content="Hi")],
    )
    result = _converter().convert_request(request)
    assert result["instructions"] == "Part one.\nPart two."


def test_assistant_tool_use_block():
    tool_input = {"query": "weather in SF", "count": 3}
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(role="user", content="What's the weather?"),
            Message(
                role="assistant",
                content=[
                    ToolUseContent(
                        id="toolu_123",
                        name="web_search",
                        input=tool_input,
                    )
                ],
            ),
        ],
    )
    result = _converter().convert_request(request)

    fn_calls = [i for i in result["input"] if i.get("type") == "function_call"]
    assert len(fn_calls) == 1
    item = fn_calls[0]
    assert item["call_id"] == "toolu_123"
    assert item["name"] == "web_search"
    assert json.loads(item["arguments"]) == tool_input
    # No extra keys beyond the verified empirical shape.
    assert set(item.keys()) == {"type", "call_id", "name", "arguments"}


def test_user_tool_result_string_content():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultContent(
                        tool_use_id="toolu_123",
                        content="It is sunny.",
                    )
                ],
            ),
        ],
    )
    result = _converter().convert_request(request)

    outputs = [i for i in result["input"] if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    item = outputs[0]
    assert item["call_id"] == "toolu_123"
    assert item["output"] == "It is sunny."
    assert set(item.keys()) == {"type", "call_id", "output"}


def test_tool_result_list_content_text_joined():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultContent(
                        tool_use_id="toolu_9",
                        content=[
                            TextContent(text="line one"),
                            TextContent(text="line two"),
                        ],
                    )
                ],
            ),
        ],
    )
    result = _converter().convert_request(request)
    outputs = [i for i in result["input"] if i.get("type") == "function_call_output"]
    assert outputs[0]["output"] == "line one\nline two"


def test_tools_conversion():
    tool = Tool(
        name="get_weather",
        description="Get the weather",
        input_schema=ToolInputSchema(
            properties={"location": {"type": "string"}},
            required=["location"],
        ),
    )
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        tools=[tool],
        messages=[Message(role="user", content="hi")],
    )
    result = _converter().convert_request(request)

    assert result["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]


def test_tools_conversion_raw_dict():
    # The web-search agentic loop passes tools as raw dicts, not Tool objects.
    tool_dict = {
        "name": "web_search",
        "description": "Search the web for information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    }
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        tools=[tool_dict],
        messages=[Message(role="user", content="hi")],
    )
    result = _converter().convert_request(request)

    assert result["tools"] == [
        {
            "type": "function",
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        }
    ]


def test_tool_choice_auto():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        tool_choice="auto",
        messages=[Message(role="user", content="hi")],
    )
    result = _converter().convert_request(request)
    assert result["tool_choice"] == "auto"


def test_tool_choice_any_becomes_required():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        tool_choice="any",
        messages=[Message(role="user", content="hi")],
    )
    result = _converter().convert_request(request)
    assert result["tool_choice"] == "required"


def test_tool_choice_specific_tool_becomes_function():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        tool_choice={"type": "tool", "name": "web_search"},
        messages=[Message(role="user", content="hi")],
    )
    result = _converter().convert_request(request)
    assert result["tool_choice"] == {"type": "function", "name": "web_search"}


def test_image_and_thinking_blocks_skipped():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(
                role="user",
                content=[
                    TextContent(text="describe this"),
                    ImageContent(
                        source=Base64ImageSource(
                            media_type="image/png", data="abc123"
                        )
                    ),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(thinking="hmm let me think"),
                    TextContent(text="a cat"),
                ],
            ),
        ],
    )
    result = _converter().convert_request(request)

    # Only text contributes; image and thinking are skipped silently.
    assert result["input"] == [
        {"role": "user", "content": "describe this"},
        {"role": "assistant", "content": "a cat"},
    ]


def test_assistant_thinking_only_produces_no_item():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content=[ThinkingContent(thinking="just thinking, no output")],
            ),
        ],
    )
    result = _converter().convert_request(request)
    # Thinking-only assistant message contributes nothing and does not crash.
    assert result["input"] == [{"role": "user", "content": "hi"}]


def test_consecutive_text_blocks_coalesced():
    request = MessageRequest(
        model="openai.gpt-5.5",
        max_tokens=512,
        messages=[
            Message(
                role="user",
                content=[
                    TextContent(text="first"),
                    TextContent(text="second"),
                ],
            ),
        ],
    )
    result = _converter().convert_request(request)
    assert result["input"] == [{"role": "user", "content": "first\nsecond"}]
