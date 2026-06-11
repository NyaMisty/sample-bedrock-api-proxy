"""Tests for admin portal model-mapping override semantics.

Default mappings (from config.py) are editable via the admin portal:
editing one writes a DynamoDB override row that shadows the default at
resolution time; deleting the override restores the default.
"""
import pytest
from fastapi import HTTPException
from moto import mock_aws

from app.core.config import settings


DEFAULT_ID = "claude-fable-5"
DEFAULT_TARGET = settings.default_model_mapping[DEFAULT_ID]
OVERRIDE_TARGET = "us.anthropic.claude-fable-5"


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=settings.dynamodb_model_mapping_table,
            KeySchema=[{"AttributeName": "anthropic_model_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "anthropic_model_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield dynamodb


@pytest.fixture
def api(mock_dynamodb):
    from admin_portal.backend.api import model_mapping

    return model_mapping


@pytest.fixture
def schemas():
    from admin_portal.backend.schemas.model_mapping import (
        ModelMappingCreate,
        ModelMappingUpdate,
    )

    return ModelMappingCreate, ModelMappingUpdate


async def test_put_default_creates_override(api, schemas):
    _, ModelMappingUpdate = schemas
    resp = await api.update_model_mapping(
        DEFAULT_ID, ModelMappingUpdate(bedrock_model_id=OVERRIDE_TARGET)
    )
    assert resp.source == "override"
    assert resp.bedrock_model_id == OVERRIDE_TARGET
    assert resp.default_bedrock_model_id == DEFAULT_TARGET


async def test_list_shows_override_once(api, schemas):
    _, ModelMappingUpdate = schemas
    await api.update_model_mapping(
        DEFAULT_ID, ModelMappingUpdate(bedrock_model_id=OVERRIDE_TARGET)
    )
    listing = await api.list_model_mappings(search=None)
    entries = [i for i in listing.items if i.anthropic_model_id == DEFAULT_ID]
    assert len(entries) == 1
    assert entries[0].source == "override"
    assert entries[0].bedrock_model_id == OVERRIDE_TARGET


async def test_override_used_at_resolution_time(api, schemas):
    _, ModelMappingUpdate = schemas
    await api.update_model_mapping(
        DEFAULT_ID, ModelMappingUpdate(bedrock_model_id=OVERRIDE_TARGET)
    )
    from app.converters.anthropic_to_bedrock import AnthropicToBedrockConverter
    from app.db.dynamodb import DynamoDBClient

    converter = AnthropicToBedrockConverter(DynamoDBClient())
    assert converter._convert_model_id(DEFAULT_ID) == OVERRIDE_TARGET


async def test_delete_override_restores_default(api, schemas):
    _, ModelMappingUpdate = schemas
    await api.update_model_mapping(
        DEFAULT_ID, ModelMappingUpdate(bedrock_model_id=OVERRIDE_TARGET)
    )
    result = await api.delete_model_mapping(DEFAULT_ID)
    assert result["restored_default"] is True

    restored = await api.get_model_mapping(DEFAULT_ID)
    assert restored.source == "default"
    assert restored.bedrock_model_id == DEFAULT_TARGET


async def test_delete_pure_default_rejected(api):
    with pytest.raises(HTTPException) as exc_info:
        await api.delete_model_mapping(DEFAULT_ID)
    assert exc_info.value.status_code == 400


async def test_put_unknown_model_404(api, schemas):
    _, ModelMappingUpdate = schemas
    with pytest.raises(HTTPException) as exc_info:
        await api.update_model_mapping(
            "no-such-model", ModelMappingUpdate(bedrock_model_id="x")
        )
    assert exc_info.value.status_code == 404


async def test_custom_mapping_crud_unchanged(api, schemas):
    ModelMappingCreate, ModelMappingUpdate = schemas
    created = await api.create_model_mapping(
        ModelMappingCreate(anthropic_model_id="my-alias", bedrock_model_id="zai.glm-5")
    )
    assert created.source == "custom"
    assert created.default_bedrock_model_id is None

    updated = await api.update_model_mapping(
        "my-alias", ModelMappingUpdate(bedrock_model_id="moonshotai.kimi-k2.5")
    )
    assert updated.source == "custom"

    result = await api.delete_model_mapping("my-alias")
    assert result["restored_default"] is False


async def test_post_over_default_reports_override(api, schemas):
    ModelMappingCreate, _ = schemas
    created = await api.create_model_mapping(
        ModelMappingCreate(
            anthropic_model_id=DEFAULT_ID, bedrock_model_id=OVERRIDE_TARGET
        )
    )
    assert created.source == "override"
    assert created.default_bedrock_model_id == DEFAULT_TARGET
