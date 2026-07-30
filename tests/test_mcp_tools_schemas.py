"""Exact JSON-Schema conformance to SPEC §10.2 for the five in-scope
local-surface tools. No database or MCP transport required — this
tests the hand-authored dicts in `imsg.mcp.tools.schemas` directly."""

from __future__ import annotations

from imsg.mcp.tools.schemas import (
    CHECK_PERMISSIONS,
    GET_ATTACHMENT_TEXT,
    GET_CONVERSATION,
    LIST_PEOPLE,
    SEARCH_MESSAGES,
    STANDARD_ANNOTATIONS,
    TOOL_DEFINITIONS,
    TOOL_DEFINITIONS_BY_NAME,
)


def test_exactly_five_tools_are_registered() -> None:
    # SPEC §10.2 closed surface — this build's scope is exactly these
    # five; find_similar_attachments/mark_relevant are out of scope,
    # run_sql/any escape hatch is never added.
    assert {t.name for t in TOOL_DEFINITIONS} == {
        "search_messages",
        "get_conversation",
        "list_people",
        "get_attachment_text",
        "check_permissions",
    }
    assert "run_sql" not in TOOL_DEFINITIONS_BY_NAME
    assert "find_similar_attachments" not in TOOL_DEFINITIONS_BY_NAME
    assert "mark_relevant" not in TOOL_DEFINITIONS_BY_NAME


def test_every_tool_uses_the_standard_readonly_annotations() -> None:
    for tool in TOOL_DEFINITIONS:
        assert tool.annotations == STANDARD_ANNOTATIONS
        assert tool.annotations["readOnlyHint"] is True
        assert tool.annotations["destructiveHint"] is False
        assert tool.annotations["idempotentHint"] is True
        assert tool.annotations["openWorldHint"] is False


def test_search_messages_schema_matches_spec_exactly() -> None:
    schema = SEARCH_MESSAGES.input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["query"]
    props = schema["properties"]
    assert props["query"] == {"type": "string", "minLength": 1, "maxLength": 1000}
    assert props["people"]["type"] == "array"
    assert props["people"]["items"] == {"type": "string"}
    assert props["people"]["maxItems"] == 20
    assert props["after"] == {"type": "string", "format": "date"}
    assert props["before"] == {"type": "string", "format": "date"}
    assert props["has_attachment"] == {"type": "boolean"}
    assert props["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 10,
    }
    assert set(props) == {"query", "people", "after", "before", "has_attachment", "limit"}


def test_get_conversation_schema_matches_spec_exactly() -> None:
    schema = GET_CONVERSATION.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["thread_id"]
    props = schema["properties"]
    assert props["thread_id"]["type"] == "string"
    assert props["anchor"]["type"] == "string"
    assert props["window"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 200,
        "default": 20,
        "description": "messages on each side of the anchor",
    }
    assert set(props) == {"thread_id", "anchor", "window"}


def test_list_people_schema_matches_spec_plus_local_extension() -> None:
    schema = LIST_PEOPLE.input_schema
    assert schema["additionalProperties"] is False
    assert "required" not in schema
    props = schema["properties"]
    assert props["query"]["type"] == "string"
    assert props["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 500,
        "default": 100,
    }
    # Local-only extension (SPEC §10.2): the public registration (not
    # built by this local-surface-only wave) would omit this property.
    assert props["include_handles"]["type"] == "boolean"
    assert set(props) == {"query", "limit", "include_handles"}


def test_get_attachment_text_schema_matches_spec_exactly() -> None:
    schema = GET_ATTACHMENT_TEXT.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["attachment_key"]
    props = schema["properties"]
    assert props["attachment_key"] == {
        "type": "string",
        "minLength": 16,
        "maxLength": 128,
    }
    assert set(props) == {"attachment_key"}


def test_check_permissions_schema_is_an_empty_object() -> None:
    schema = CHECK_PERMISSIONS.input_schema
    assert schema == {"type": "object", "properties": {}, "additionalProperties": False}
