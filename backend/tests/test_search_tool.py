"""
Tests for CourseSearchTool.execute() and _format_results().
Uses a real (ephemeral) ChromaDB instance seeded with fixture data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import pytest
from unittest.mock import MagicMock, patch
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def temp_vector_store():
    """Real VectorStore backed by a temp directory, pre-seeded with one course."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(
            chroma_path=tmpdir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        course = Course(
            title="Test RAG Course",
            course_link="https://example.com/course",
            instructor="Jane Doe",
            lessons=[
                Lesson(lesson_number=1, title="Intro to RAG", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Vector Stores", lesson_link="https://example.com/lesson2"),
            ]
        )
        store.add_course_metadata(course)

        chunks = [
            CourseChunk(content="Lesson 1 content: RAG stands for Retrieval Augmented Generation. It combines search with LLMs.", course_title="Test RAG Course", lesson_number=1, chunk_index=0),
            CourseChunk(content="Vector stores index embeddings for fast semantic search.", course_title="Test RAG Course", lesson_number=2, chunk_index=1),
            CourseChunk(content="ChromaDB is a popular open source vector database.", course_title="Test RAG Course", lesson_number=2, chunk_index=2),
        ]
        store.add_course_content(chunks)

        yield store


@pytest.fixture
def search_tool(temp_vector_store):
    return CourseSearchTool(temp_vector_store)


# ---------------------------------------------------------------------------
# execute() tests
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:

    def test_execute_returns_string(self, search_tool):
        result = search_tool.execute(query="What is RAG?")
        assert isinstance(result, str)

    def test_execute_finds_relevant_content(self, search_tool):
        result = search_tool.execute(query="What is RAG?")
        assert len(result) > 0
        assert "No relevant content found" not in result

    def test_execute_with_course_name_filter(self, search_tool):
        result = search_tool.execute(query="vector stores", course_name="Test RAG Course")
        assert "No relevant content found" not in result
        assert len(result) > 0

    def test_execute_with_lesson_number_filter(self, search_tool):
        result = search_tool.execute(query="vector database", lesson_number=2)
        assert "No relevant content found" not in result

    def test_execute_nonexistent_course_returns_error(self, search_tool):
        result = search_tool.execute(query="anything", course_name="Nonexistent Course XYZ")
        assert "No course found" in result or "No relevant content found" in result

    def test_execute_populates_last_sources(self, search_tool):
        search_tool.last_sources = []
        search_tool.execute(query="What is RAG?")
        assert len(search_tool.last_sources) > 0

    def test_execute_sources_have_label_and_url(self, search_tool):
        search_tool.last_sources = []
        search_tool.execute(query="What is RAG?")
        for source in search_tool.last_sources:
            assert "label" in source
            assert "url" in source

    def test_execute_source_label_contains_course_title(self, search_tool):
        search_tool.last_sources = []
        search_tool.execute(query="What is RAG?")
        labels = [s["label"] for s in search_tool.last_sources]
        assert any("Test RAG Course" in label for label in labels)

    def test_execute_source_url_is_lesson_link(self, search_tool):
        search_tool.last_sources = []
        search_tool.execute(query="RAG retrieval augmented generation")
        urls = [s["url"] for s in search_tool.last_sources if s["url"]]
        assert any("example.com/lesson" in url for url in urls)

    def test_execute_empty_query_returns_results_or_empty_message(self, search_tool):
        result = search_tool.execute(query="zzzznotarealword12345")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _format_results() tests
# ---------------------------------------------------------------------------

class TestFormatResults:

    def test_format_results_includes_course_header(self, search_tool):
        results = SearchResults(
            documents=["Some content about RAG"],
            metadata=[{"course_title": "Test RAG Course", "lesson_number": 1}],
            distances=[0.1]
        )
        output = search_tool._format_results(results)
        assert "[Test RAG Course" in output
        assert "Lesson 1" in output

    def test_format_results_stores_sources(self, search_tool):
        search_tool.last_sources = []
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Test RAG Course", "lesson_number": 1}],
            distances=[0.1]
        )
        search_tool._format_results(results)
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["label"] == "Test RAG Course - Lesson 1"

    def test_format_results_no_lesson_number(self, search_tool):
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Test RAG Course", "lesson_number": None}],
            distances=[0.1]
        )
        output = search_tool._format_results(results)
        assert "[Test RAG Course]" in output

    def test_format_results_multiple_docs(self, search_tool):
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[
                {"course_title": "Test RAG Course", "lesson_number": 1},
                {"course_title": "Test RAG Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2]
        )
        output = search_tool._format_results(results)
        assert "Lesson 1" in output
        assert "Lesson 2" in output
        assert len(search_tool.last_sources) == 2


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------

class TestToolManager:

    def test_tool_manager_registers_tool(self, temp_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(temp_vector_store)
        manager.register_tool(tool)
        assert "search_course_content" in manager.tools

    def test_tool_manager_execute_tool(self, temp_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(temp_vector_store)
        manager.register_tool(tool)
        result = manager.execute_tool("search_course_content", query="What is RAG?")
        assert isinstance(result, str)

    def test_tool_manager_get_last_sources(self, temp_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(temp_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="What is RAG?")
        sources = manager.get_last_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_tool_manager_reset_sources(self, temp_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(temp_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="What is RAG?")
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_get_tool_definitions_returns_list(self, temp_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(temp_vector_store)
        manager.register_tool(tool)
        defs = manager.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"
