"""
Converter from Anthropic Messages API format to OpenAI Responses API format.

Renders an Anthropic ``MessageRequest`` (including conversation state threaded
as ``messages``) into a kwargs dict suitable for the OpenAI SDK
``client.responses.create(**kwargs)`` call.

Used by the proxy's server-side agentic loops (e.g. web search) to drive
non-Claude, Responses-API-only models (e.g. ``openai.gpt-5.5``) through the
OpenAI Responses API. The proxy maintains conversation state itself, so the
request is always stateless (``store=False``) and the full conversation is
re-rendered into the ``input`` array on every call.
"""
import json
from typing import Any, Dict, List

from app.schemas.anthropic import (
    Message,
    MessageRequest,
    SystemMessage,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)


class AnthropicToOpenAIResponsesConverter:
    """Converts Anthropic Messages API format to OpenAI Responses API format."""

    def convert_request(self, request: MessageRequest) -> Dict[str, Any]:
        """Convert an Anthropic MessageRequest to OpenAI Responses API kwargs.

        Args:
            request: Anthropic MessageRequest object.

        Returns:
            Dictionary suitable for ``client.responses.create(**kwargs)``.
        """
        result: Dict[str, Any] = {
            "model": request.model,
            # Proxy maintains conversation state itself; never persist server-side.
            "store": False,
            "max_output_tokens": request.max_tokens,
        }

        # System content → instructions
        if request.system:
            instructions = self._convert_system(request.system)
            if instructions:
                result["instructions"] = instructions

        # Conversation messages → input array
        input_items: List[Dict[str, Any]] = []
        for msg in request.messages:
            input_items.extend(self._convert_message(msg))
        result["input"] = input_items

        # Tools
        if request.tools:
            result["tools"] = self._convert_tools(request.tools)

        # Tool choice (translate Anthropic shapes to Responses API shapes)
        if request.tool_choice is not None:
            result["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        # Sampling params (temperature/top_p/stop/stream) are intentionally
        # omitted: the server-side web-search loop controls sampling itself.

        return result

    def _convert_system(self, system: Any) -> str:
        """Flatten the Anthropic system prompt into a plain string.

        ``system`` may be a string or a list of SystemMessage blocks. The
        field_validator on MessageRequest converts strings into a
        list-of-SystemMessage, so a list is the common case.
        """
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            texts: List[str] = []
            for block in system:
                if isinstance(block, SystemMessage):
                    texts.append(block.text)
                elif isinstance(block, dict):
                    texts.append(block.get("text", ""))
            return "\n".join(texts)
        return ""

    def _convert_message(self, message: Message) -> List[Dict[str, Any]]:
        """Convert a single Anthropic message into Responses input items.

        A plain-string message becomes a single role item. A message with
        content blocks may expand into multiple items (a coalesced text item
        plus function_call / function_call_output items).
        """
        role = message.role
        content = message.content

        if isinstance(content, str):
            return [{"role": role, "content": content}]

        if not isinstance(content, list):
            return [{"role": role, "content": str(content)}]

        items: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        def flush_text() -> None:
            if text_parts:
                items.append({"role": role, "content": "\n".join(text_parts)})
                text_parts.clear()

        for block in content:
            if isinstance(block, TextContent) or (
                isinstance(block, dict) and block.get("type") == "text"
            ):
                text = (
                    block.text if isinstance(block, TextContent)
                    else block.get("text", "")
                )
                text_parts.append(text)

            elif isinstance(block, ToolUseContent) or (
                isinstance(block, dict) and block.get("type") == "tool_use"
            ):
                flush_text()
                if isinstance(block, ToolUseContent):
                    call_id = block.id
                    name = block.name
                    tool_input = block.input
                else:
                    call_id = block.get("id", "")
                    name = block.get("name", "")
                    tool_input = block.get("input", {})
                items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(tool_input),
                })

            elif isinstance(block, ToolResultContent) or (
                isinstance(block, dict) and block.get("type") == "tool_result"
            ):
                flush_text()
                items.append(self._convert_tool_result(block))

            # Any other block type (image, thinking, etc.) is skipped:
            # the agentic loop this serves is text + tools only.

        flush_text()
        return items

    def _convert_tool_result(self, block: Any) -> Dict[str, Any]:
        """Convert a tool_result block into a function_call_output item."""
        if isinstance(block, ToolResultContent):
            call_id = block.tool_use_id
            content = block.content
        else:
            call_id = block.get("tool_use_id", "")
            content = block.get("content", "")

        if isinstance(content, str):
            output = content
        elif isinstance(content, list):
            parts: List[str] = []
            has_text = False
            for item in content:
                if isinstance(item, TextContent):
                    parts.append(item.text)
                    has_text = True
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                    has_text = True
            if has_text:
                output = "\n".join(parts)
            else:
                # Non-text content (e.g. images) — fall back to a JSON dump.
                output = self._dump_content(content)
        else:
            output = str(content)

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    @staticmethod
    def _dump_content(content: Any) -> str:
        """Best-effort JSON serialization of arbitrary tool-result content."""
        def _default(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        try:
            return json.dumps(content, default=_default)
        except (TypeError, ValueError):
            return str(content)

    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert Anthropic Tool definitions to Responses function tools."""
        openai_tools: List[Dict[str, Any]] = []
        for tool in tools:
            # Default name/description to "" — mantle rejects null values.
            name = getattr(tool, "name", "") or ""
            description = getattr(tool, "description", "") or ""
            input_schema = getattr(tool, "input_schema", None)

            if input_schema is not None and hasattr(input_schema, "model_dump"):
                parameters = input_schema.model_dump(exclude_none=True)
            elif isinstance(input_schema, dict):
                parameters = input_schema
            else:
                parameters = {}

            openai_tools.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            })
        return openai_tools

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        """Translate Anthropic tool_choice into the Responses API shape.

        Mirrors the sibling Chat Completions converter:
        - "auto" → "auto"
        - "any" → "required"
        - {"type":"tool","name":X} → {"type":"function","name":X}
        - "none" / unknown → passed through sensibly.
        """
        if isinstance(tool_choice, str):
            if tool_choice == "any":
                return "required"
            return tool_choice  # "auto", "none", etc. pass through

        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type", "")
            if tc_type == "auto":
                return "auto"
            elif tc_type == "any":
                return "required"
            elif tc_type == "none":
                return "none"
            elif tc_type == "tool":
                return {"type": "function", "name": tool_choice.get("name", "")}
        return "auto"
