"""Model ID resolution for the OpenAI passthrough endpoints.

Looks up the client-supplied model in the existing model_mapping table; if a
mapping exists, substitute it. Otherwise, pass through unchanged so callers
can use Bedrock-native IDs (e.g. ``openai.gpt-oss-120b``) directly without
needing to register them.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_model_id(model: str, model_mapping_manager) -> str:
    """Resolve a client-supplied model ID via the mapping table, with fallback.

    Args:
        model: The ``model`` field from the client request.
        model_mapping_manager: An app.db.dynamodb.ModelMappingManager instance.

    Returns:
        The resolved Bedrock model ID, or the original string if no mapping
        exists or the lookup fails.
    """
    if not model:
        return model
    try:
        mapped = model_mapping_manager.get_mapping(model)
    except Exception as exc:
        logger.warning("[OPENAI-PASSTHROUGH] model mapping lookup failed for %r: %s", model, exc)
        return model
    return mapped or model
