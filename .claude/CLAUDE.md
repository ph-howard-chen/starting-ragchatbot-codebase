# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies**
```bash
uv sync
```

**Run the application** (from project root)
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The server runs on `http://localhost:8000`. Frontend is served as static files; API docs at `/docs`.

**Run a single Python file directly**
```bash
cd backend && uv run python <file>.py
```

There are no tests in this codebase currently.

## Architecture

This is a full-stack RAG chatbot. The FastAPI backend serves the frontend as static files — there is no separate frontend server.

**Request flow for `POST /api/query`:**
1. `backend/app.py` — FastAPI route, creates session if needed, delegates to `RAGSystem`
2. `backend/rag_system.py` — orchestrates the pipeline: retrieves session history, calls `AIGenerator`
3. `backend/ai_generator.py` — calls Claude API with a `search_course_content` tool; if Claude invokes the tool, executes it and sends a follow-up call to get the final answer
4. `backend/search_tools.py` — tool definition and execution; delegates to `VectorStore`
5. `backend/vector_store.py` — embeds the query with `all-MiniLM-L6-v2`, runs semantic search against ChromaDB

**Key design decisions:**
- Claude uses **tool calling** (not a pre-retrieval pipeline) — the LLM decides whether to call `search_course_content` based on the query
- Two ChromaDB collections: `course_catalog` (course metadata for fuzzy name resolution) and `course_content` (chunked lesson text for search)
- Sessions are **in-memory** only (`session_manager.py`) and lost on restart; history is capped at 2 exchanges
- The `docs/` folder (project root) is loaded into ChromaDB on startup; ChromaDB persists to `backend/chroma_db/`

**Configuration** (`backend/config.py`): `ANTHROPIC_MODEL`, `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100), `MAX_RESULTS` (5), `MAX_HISTORY` (2), `CHROMA_PATH`.

**Document format** expected in `docs/`:
```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: Title
Lesson Link: ...
[content]
```

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
