"""
Tests for AIGenerator — verifies it calls the CourseSearchTool when given
a course-related query and returns a coherent final response.
All Anthropic API calls are mocked.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator
from search_tools import ToolManager


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_name, tool_id, input_dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = input_dict
    return block


def make_response(stop_reason, content_blocks):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return AIGenerator(api_key="test-key", model="claude-test-model")


@pytest.fixture
def tool_manager():
    manager = ToolManager()
    mock_tool = MagicMock()
    mock_tool.get_tool_definition.return_value = {
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
    mock_tool.execute.return_value = "[Test Course - Lesson 1]\nRAG combines retrieval with generation."
    mock_tool.last_sources = [{"label": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}]
    manager.tools["search_course_content"] = mock_tool
    return manager


# ---------------------------------------------------------------------------
# Direct response (no tool call)
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def test_returns_text_without_tool_call(self, generator):
        direct_response = make_response("end_turn", [make_text_block("Paris is the capital of France.")])
        with patch.object(generator.client.messages, "create", return_value=direct_response):
            result = generator.generate_response("What is the capital of France?")
        assert result == "Paris is the capital of France."

    def test_no_tool_manager_skips_tool_handling(self, generator):
        direct_response = make_response("end_turn", [make_text_block("Answer.")])
        with patch.object(generator.client.messages, "create", return_value=direct_response):
            result = generator.generate_response("Some question", tool_manager=None)
        assert result == "Answer."


# ---------------------------------------------------------------------------
# Tool call flow
# ---------------------------------------------------------------------------

class TestToolCallFlow:

    def test_triggers_tool_use_for_course_query(self, generator, tool_manager):
        """When Claude returns stop_reason=tool_use, the tool should be executed."""
        tool_block = make_tool_use_block(
            "search_course_content",
            "call_123",
            {"query": "What is RAG?"}
        )
        first_response = make_response("tool_use", [tool_block])
        final_response = make_response("end_turn", [make_text_block("RAG is Retrieval Augmented Generation.")])

        with patch.object(generator.client.messages, "create", side_effect=[first_response, final_response]):
            result = generator.generate_response(
                query="What is RAG?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

        assert result == "RAG is Retrieval Augmented Generation."

    def test_api_called_twice_for_tool_use(self, generator, tool_manager):
        """Two API calls should be made: initial + follow-up after tool execution."""
        tool_block = make_tool_use_block("search_course_content", "call_abc", {"query": "RAG"})
        first_response = make_response("tool_use", [tool_block])
        final_response = make_response("end_turn", [make_text_block("Final answer.")])

        with patch.object(generator.client.messages, "create", side_effect=[first_response, final_response]) as mock_create:
            generator.generate_response(
                query="Tell me about RAG",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )
        assert mock_create.call_count == 2

    def test_tool_executed_with_correct_query(self, generator, tool_manager):
        """The tool should be called with the exact input Claude specified."""
        tool_block = make_tool_use_block("search_course_content", "call_xyz", {"query": "vector stores"})
        first_response = make_response("tool_use", [tool_block])
        final_response = make_response("end_turn", [make_text_block("Answer about vectors.")])

        with patch.object(generator.client.messages, "create", side_effect=[first_response, final_response]):
            generator.generate_response(
                query="Tell me about vector stores",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

        tool_manager.tools["search_course_content"].execute.assert_called_once_with(query="vector stores")

    def test_tool_result_included_in_followup_messages(self, generator, tool_manager):
        """The follow-up API call should include the tool result in its messages."""
        tool_block = make_tool_use_block("search_course_content", "call_001", {"query": "RAG"})
        first_response = make_response("tool_use", [tool_block])
        final_response = make_response("end_turn", [make_text_block("Done.")])

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return final_response

        with patch.object(generator.client.messages, "create", side_effect=[first_response, capture_create]):
            generator.generate_response(
                query="About RAG",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

        messages = captured_params.get("messages", [])
        tool_result_messages = [
            m for m in messages
            if isinstance(m.get("content"), list) and
            any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_messages) > 0

    def test_second_api_call_has_no_tools(self, generator, tool_manager):
        """The follow-up call after tool execution should NOT include tools (to get final text)."""
        tool_block = make_tool_use_block("search_course_content", "call_002", {"query": "RAG"})
        first_response = make_response("tool_use", [tool_block])
        final_response = make_response("end_turn", [make_text_block("Done.")])

        call_params = []

        def capturing_create(**kwargs):
            call_params.append(kwargs)
            if len(call_params) == 1:
                return first_response
            return final_response

        with patch.object(generator.client.messages, "create", side_effect=capturing_create):
            generator.generate_response(
                query="About RAG",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

        assert len(call_params) == 2
        assert "tools" not in call_params[1]

    def test_conversation_history_included_in_system(self, generator, tool_manager):
        """Conversation history should appear in the system prompt."""
        direct_response = make_response("end_turn", [make_text_block("Answer.")])

        with patch.object(generator.client.messages, "create", return_value=direct_response) as mock_create:
            generator.generate_response(
                query="Follow-up question",
                conversation_history="User: Hello\nAssistant: Hi there",
            )

        call_kwargs = mock_create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]
        assert "Hello" in call_kwargs["system"]
