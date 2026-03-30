"""
Tests for RAGSystem.query() — end-to-end flow with a real VectorStore
(ephemeral ChromaDB) and mocked Anthropic API.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import pytest
from unittest.mock import MagicMock, patch
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from session_manager import SessionManager
from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Helpers
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
# Config stub
# ---------------------------------------------------------------------------

class StubConfig:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-test-model"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_rag_system():
    """RAGSystem with real ephemeral VectorStore, seeded with one course."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StubConfig()
        config.CHROMA_PATH = tmpdir

        rag = RAGSystem(config)

        course = Course(
            title="Introduction to Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Smith",
            lessons=[
                Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/ml-lesson1"),
                Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml-lesson2"),
            ]
        )
        rag.vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(content="Lesson 1 content: Machine learning is a subset of AI that learns from data.", course_title="Introduction to Machine Learning", lesson_number=1, chunk_index=0),
            CourseChunk(content="Supervised learning uses labeled training data to learn a mapping from inputs to outputs.", course_title="Introduction to Machine Learning", lesson_number=2, chunk_index=1),
            CourseChunk(content="Common supervised learning algorithms include linear regression, decision trees, and neural networks.", course_title="Introduction to Machine Learning", lesson_number=2, chunk_index=2),
        ]
        rag.vector_store.add_course_content(chunks)

        yield rag


# ---------------------------------------------------------------------------
# query() — response tuple structure
# ---------------------------------------------------------------------------

class TestRAGSystemQueryStructure:

    def test_query_returns_tuple(self, seeded_rag_system):
        direct = make_response("end_turn", [make_text_block("ML is cool.")])
        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            result = seeded_rag_system.query("What is machine learning?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_first_element_is_string(self, seeded_rag_system):
        direct = make_response("end_turn", [make_text_block("Answer here.")])
        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            answer, _ = seeded_rag_system.query("Tell me something.")
        assert isinstance(answer, str)

    def test_query_second_element_is_list(self, seeded_rag_system):
        direct = make_response("end_turn", [make_text_block("Answer.")])
        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            _, sources = seeded_rag_system.query("Tell me something.")
        assert isinstance(sources, list)


# ---------------------------------------------------------------------------
# query() — tool call path (content-related query)
# ---------------------------------------------------------------------------

class TestRAGSystemToolCallPath:

    def test_query_with_tool_call_returns_final_answer(self, seeded_rag_system):
        tool_block = make_tool_use_block(
            "search_course_content", "id_001",
            {"query": "What is supervised learning?"}
        )
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("Supervised learning uses labeled data.")])

        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", side_effect=[first_resp, final_resp]):
            answer, sources = seeded_rag_system.query("What is supervised learning?")

        assert answer == "Supervised learning uses labeled data."

    def test_query_with_tool_call_returns_sources(self, seeded_rag_system):
        tool_block = make_tool_use_block(
            "search_course_content", "id_002",
            {"query": "supervised learning algorithms"}
        )
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("Here are the algorithms.")])

        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", side_effect=[first_resp, final_resp]):
            answer, sources = seeded_rag_system.query("What supervised learning algorithms exist?")

        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_sources_have_label_field(self, seeded_rag_system):
        tool_block = make_tool_use_block(
            "search_course_content", "id_003",
            {"query": "machine learning AI"}
        )
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("ML is a subset of AI.")])

        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", side_effect=[first_resp, final_resp]):
            _, sources = seeded_rag_system.query("What is ML?")

        for source in sources:
            assert "label" in source

    def test_sources_reset_between_queries(self, seeded_rag_system):
        """Sources from query 1 must not bleed into query 2 if query 2 has no tool call."""
        # Query 1: triggers tool call
        tool_block = make_tool_use_block("search_course_content", "id_004", {"query": "ML"})
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("ML answer.")])

        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", side_effect=[first_resp, final_resp]):
            seeded_rag_system.query("What is ML?")

        # Query 2: direct answer, no tool call
        direct = make_response("end_turn", [make_text_block("General answer.")])
        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            _, sources = seeded_rag_system.query("What is 2+2?")

        assert sources == []


# ---------------------------------------------------------------------------
# Session / conversation history
# ---------------------------------------------------------------------------

class TestRAGSystemSession:

    def test_query_with_session_id_stores_history(self, seeded_rag_system):
        direct = make_response("end_turn", [make_text_block("ML answer.")])
        session_id = seeded_rag_system.session_manager.create_session()

        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            seeded_rag_system.query("What is ML?", session_id=session_id)

        history = seeded_rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is ML?" in history

    def test_query_without_session_does_not_error(self, seeded_rag_system):
        direct = make_response("end_turn", [make_text_block("Answer.")])
        with patch.object(seeded_rag_system.ai_generator.client.messages, "create", return_value=direct):
            answer, sources = seeded_rag_system.query("Hello")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# VectorStore search integration (no mocking — real ChromaDB)
# ---------------------------------------------------------------------------

class TestVectorStoreSearch:

    def test_search_returns_results_for_known_content(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search("What is machine learning?")
        assert not results.is_empty()
        assert len(results.documents) > 0

    def test_search_returns_results_with_course_filter(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search(
            "supervised learning",
            course_name="Introduction to Machine Learning"
        )
        assert not results.is_empty()

    def test_search_returns_results_with_lesson_filter(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search(
            "labeled data",
            lesson_number=2
        )
        assert not results.is_empty()

    def test_search_metadata_has_course_title(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search("machine learning")
        for meta in results.metadata:
            assert "course_title" in meta

    def test_search_metadata_has_lesson_number(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search("machine learning")
        for meta in results.metadata:
            assert "lesson_number" in meta

    def test_search_with_combined_filter(self, seeded_rag_system):
        results = seeded_rag_system.vector_store.search(
            "algorithms",
            course_name="Introduction to Machine Learning",
            lesson_number=2
        )
        assert not results.is_empty()
