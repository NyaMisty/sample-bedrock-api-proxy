"""OpenAI Passthrough — accepts OpenAI Chat Completions and Responses API
calls from clients and forwards them to AWS bedrock-mantle.
"""
from app.api.openai_passthrough.router import router

__all__ = ["router"]
