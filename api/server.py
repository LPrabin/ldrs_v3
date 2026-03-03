"""
FastAPI server for the LDRS v3 Hybrid RAG Deep Agent System.

Endpoints:
  POST /query           — Run the full 6-stage pipeline
  POST /batch-query     — Run multiple queries
  POST /index           — Index a markdown file
  POST /index-directory — Index all .md files in a directory
  GET  /corpus          — Corpus summary
  GET  /corpus/stats    — Detailed corpus statistics
  POST /corpus/rebuild  — Re-index all documents
  GET  /sessions        — List VFS sessions
  DELETE /sessions/{id} — Delete a VFS session
  POST /sessions/cleanup — Delete all VFS sessions
  GET  /config          — Get current LLM/embedding provider settings
  PUT  /config          — Update LLM/embedding provider settings at runtime
  GET  /health          — Health check (includes model info)

Usage::

    uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.config import AgentConfig
from agent.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Get the shared Pipeline instance."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Server not started properly.")
    return _pipeline


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    global _pipeline

    logger.info("Server starting up...")
    config = AgentConfig()
    _pipeline = Pipeline(config)
    await _pipeline.startup()
    logger.info("Server ready.")

    yield

    logger.info("Server shutting down...")
    if _pipeline is not None:
        await _pipeline.shutdown()
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LDRS v3",
    description="Living Document RAG System — Hybrid RAG Deep Agent",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /query."""

    query: str = Field(..., description="The user's question")
    conversation_summary: Optional[str] = Field(None, description="Summary of prior conversation")
    recent_turns: Optional[List[Dict[str, str]]] = Field(
        None, description="Recent conversation turns"
    )
    db_context: Optional[Dict[str, Any]] = Field(None, description="Optional database context")
    cleanup_session: bool = Field(False, description="Delete VFS session after query")


class QueryResponse(BaseModel):
    """Response body for /query."""

    answer: str
    intent_type: Optional[str] = None
    selected_files: Optional[List[str]] = None
    candidates_count: int = 0
    ranked_count: int = 0
    session_id: str = ""
    citations: Optional[List[str]] = None
    claims_checked: int = 0
    claims_supported: int = 0
    claims_flagged: int = 0
    re_grounded: bool = False
    usage: Dict[str, Any] = {}
    total_time_ms: float = 0.0
    success: bool = True
    error: str = ""


class BatchQueryRequest(BaseModel):
    """Request body for /batch-query."""

    queries: List[str] = Field(..., description="List of queries to run")
    conversation_summary: Optional[str] = None
    cleanup_sessions: bool = Field(False, description="Delete VFS sessions after each query")


class IndexRequest(BaseModel):
    """Request body for /index."""

    md_path: str = Field(..., description="Path to the .md file to index")
    tags: Optional[List[str]] = Field(None, description="Tags for the document")
    summary: str = Field("", description="Document summary")
    if_thinning: bool = Field(False, description="Enable tree thinning")


class IndexDirectoryRequest(BaseModel):
    """Request body for /index-directory."""

    directory: Optional[str] = Field(None, description="Directory path (defaults to docs_dir)")
    tags: Optional[List[str]] = None
    if_thinning: bool = False


class IndexResponse(BaseModel):
    """Response body for /index."""

    md_path: str = ""
    doc_name: str = ""
    index_path: str = ""
    node_count: int = 0
    section_count: int = 0
    embedded_count: int = 0
    success: bool = True
    error: str = ""


class CorpusSummary(BaseModel):
    """Response body for /corpus."""

    total_files: int = 0
    total_tokens: int = 0
    total_nodes: int = 0
    files_with_embeddings: int = 0
    file_names: List[str] = []


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str = "ok"
    pipeline_started: bool = False
    corpus_files: int = 0
    version: str = "3.0.0"
    default_model: str = ""
    embedding_model: str = ""
    api_base: str = ""
    api_key_set: bool = False


class LLMConfigResponse(BaseModel):
    """Response body for GET /config."""

    default_model: str = ""
    embedding_model: str = ""
    api_base: str = ""
    api_key_set: bool = False


class LLMConfigUpdateRequest(BaseModel):
    """Request body for PUT /config."""

    default_model: Optional[str] = Field(None, description="LiteLLM model string for chat")
    embedding_model: Optional[str] = Field(None, description="LiteLLM model string for embeddings")
    api_key: Optional[str] = Field(None, description="API key (set to empty string to clear)")
    api_base: Optional[str] = Field(None, description="Base URL for custom endpoints")


# ---------------------------------------------------------------------------
# Helper: PipelineResult -> QueryResponse
# ---------------------------------------------------------------------------


def _to_query_response(result: PipelineResult) -> QueryResponse:
    """Convert a PipelineResult to a QueryResponse."""
    resp = QueryResponse(
        answer=result.answer,
        candidates_count=result.candidates_count,
        ranked_count=result.ranked_count,
        session_id=result.session_id,
        usage=result.usage,
        total_time_ms=result.total_time_ms,
        success=result.success,
        error=result.error,
    )

    if result.intent:
        resp.intent_type = result.intent.intent_type
        resp.selected_files = result.intent.selected_paths

    if result.agent_result:
        resp.citations = result.agent_result.citations

    if result.grounding_result:
        resp.claims_checked = result.grounding_result.claims_checked
        resp.claims_supported = result.grounding_result.claims_supported
        resp.claims_flagged = result.grounding_result.claims_flagged
        resp.re_grounded = result.grounding_result.re_grounded

    return resp


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the full 6-stage pipeline for a user query."""
    logger.info("POST /query  query_len=%d", len(request.query))
    start = time.time()
    pipeline = get_pipeline()

    result = await pipeline.query(
        query=request.query,
        conversation_summary=request.conversation_summary,
        recent_turns=request.recent_turns,
        db_context=request.db_context,
        cleanup_session=request.cleanup_session,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "POST /query  done  answer_len=%d  success=%s  elapsed_ms=%.0f",
        len(result.answer),
        result.success,
        elapsed_ms,
    )
    return _to_query_response(result)


@app.post("/batch-query", response_model=List[QueryResponse])
async def batch_query(request: BatchQueryRequest) -> List[QueryResponse]:
    """Run multiple queries sequentially."""
    logger.info("POST /batch-query  queries=%d", len(request.queries))
    start = time.time()
    pipeline = get_pipeline()
    responses = []

    for i, q in enumerate(request.queries):
        logger.debug("POST /batch-query  running query %d/%d", i + 1, len(request.queries))
        result = await pipeline.query(
            query=q,
            conversation_summary=request.conversation_summary,
            cleanup_session=request.cleanup_sessions,
        )
        responses.append(_to_query_response(result))

    elapsed_ms = (time.time() - start) * 1000
    logger.info("POST /batch-query  done  queries=%d  elapsed_ms=%.0f", len(responses), elapsed_ms)
    return responses


@app.post("/index", response_model=IndexResponse)
async def index_file(request: IndexRequest) -> IndexResponse:
    """Index a single markdown file."""
    logger.info("POST /index  md_path=%s", request.md_path)
    pipeline = get_pipeline()

    if not os.path.isfile(request.md_path):
        logger.warning("POST /index  file not found: %s", request.md_path)
        raise HTTPException(status_code=404, detail=f"File not found: {request.md_path}")

    result = await pipeline.index_file(
        md_path=request.md_path,
        tags=request.tags,
        summary=request.summary,
        if_thinning=request.if_thinning,
    )

    logger.info(
        "POST /index  done  doc=%s  success=%s  sections=%d",
        result.doc_name,
        result.success,
        result.section_count,
    )
    return IndexResponse(
        md_path=result.md_path,
        doc_name=result.doc_name,
        index_path=result.index_path,
        node_count=result.node_count,
        section_count=result.section_count,
        embedded_count=result.embedded_count,
        success=result.success,
        error=result.error,
    )


@app.post("/index-directory", response_model=List[IndexResponse])
async def index_directory(request: IndexDirectoryRequest) -> List[IndexResponse]:
    """Index all markdown files in a directory."""
    logger.info("POST /index-directory  directory=%s", request.directory)
    pipeline = get_pipeline()

    results = await pipeline.index_directory(
        directory=request.directory,
        tags=request.tags,
        if_thinning=request.if_thinning,
    )

    logger.info("POST /index-directory  done  files=%d", len(results))
    return [
        IndexResponse(
            md_path=r.md_path,
            doc_name=r.doc_name,
            index_path=r.index_path,
            node_count=r.node_count,
            section_count=r.section_count,
            embedded_count=r.embedded_count,
            success=r.success,
            error=r.error,
        )
        for r in results
    ]


@app.get("/corpus", response_model=CorpusSummary)
async def corpus_summary() -> CorpusSummary:
    """Get a summary of the current corpus."""
    logger.debug("GET /corpus")
    pipeline = get_pipeline()
    summary = pipeline.get_corpus_summary()

    return CorpusSummary(
        total_files=summary.get("total_files", 0),
        total_tokens=summary.get("total_tokens", 0),
        total_nodes=summary.get("total_nodes", 0),
        files_with_embeddings=summary.get("files_with_embeddings", 0),
        file_names=summary.get("file_names", []),
    )


@app.get("/corpus/stats", response_model=Dict[str, Any])
async def corpus_stats() -> Dict[str, Any]:
    """Get detailed corpus statistics including per-file info."""
    logger.debug("GET /corpus/stats")
    pipeline = get_pipeline()
    summary = pipeline.get_corpus_summary()
    files = pipeline.get_corpus_files()

    return {
        "summary": summary,
        "files": {
            path: {
                "summary": info.get("summary", ""),
                "tags": info.get("tags", []),
                "sections_count": len(info.get("sections", [])),
                "size_tokens": info.get("size_tokens", 0),
                "has_embeddings": info.get("has_embeddings", False),
                "last_modified": info.get("last_modified", ""),
            }
            for path, info in files.items()
        },
    }


@app.post("/corpus/rebuild", response_model=List[IndexResponse])
async def corpus_rebuild() -> List[IndexResponse]:
    """Re-index all documents in the docs directory."""
    logger.info("POST /corpus/rebuild")
    pipeline = get_pipeline()
    results = await pipeline.index_directory()

    return [
        IndexResponse(
            md_path=r.md_path,
            doc_name=r.doc_name,
            index_path=r.index_path,
            node_count=r.node_count,
            section_count=r.section_count,
            embedded_count=r.embedded_count,
            success=r.success,
            error=r.error,
        )
        for r in results
    ]


@app.get("/sessions", response_model=List[str])
async def list_sessions() -> List[str]:
    """List all VFS session IDs."""
    logger.debug("GET /sessions")
    pipeline = get_pipeline()
    sessions = pipeline.list_sessions()
    logger.debug("GET /sessions  count=%d", len(sessions))
    return sessions


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a specific VFS session."""
    logger.info("DELETE /sessions/%s", session_id)
    pipeline = get_pipeline()
    try:
        pipeline.cleanup_session(session_id)
        logger.info("DELETE /sessions/%s  done", session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.warning("DELETE /sessions/%s  not found: %s", session_id, e)
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/sessions/cleanup")
async def cleanup_all_sessions() -> Dict[str, str]:
    """Delete all VFS sessions."""
    logger.info("POST /sessions/cleanup")
    pipeline = get_pipeline()
    sessions = pipeline.list_sessions()
    pipeline.cleanup_all_sessions()
    logger.info("POST /sessions/cleanup  deleted=%d", len(sessions))
    return {"status": "cleaned", "sessions_deleted": str(len(sessions))}


@app.get("/config", response_model=LLMConfigResponse)
async def get_config() -> LLMConfigResponse:
    """Get current LLM/embedding provider configuration."""
    logger.debug("GET /config")
    pipeline = get_pipeline()
    settings = pipeline.config.get_llm_settings()
    return LLMConfigResponse(**settings)


@app.put("/config", response_model=Dict[str, Any])
async def update_config(request: LLMConfigUpdateRequest) -> Dict[str, Any]:
    """Update LLM/embedding provider configuration at runtime.

    Only non-null fields are applied.  Components that depend on changed
    settings are re-created on the next query.
    """
    logger.info(
        "PUT /config  default_model=%s  embedding_model=%s  api_base=%s  api_key=%s",
        request.default_model,
        request.embedding_model,
        request.api_base,
        "***" if request.api_key is not None else None,
    )
    pipeline = get_pipeline()

    result = pipeline.update_llm_config(
        default_model=request.default_model,
        embedding_model=request.embedding_model,
        api_key=request.api_key,
        api_base=request.api_base,
    )

    # Return current settings alongside the change report
    current = pipeline.config.get_llm_settings()
    logger.info("PUT /config  changed=%s  current=%s", result["changed"], current)
    return {
        "changed": result["changed"],
        "changes": result["fields"],
        "current": current,
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    pipeline_started = _pipeline is not None and _pipeline._started
    corpus_files = 0
    llm_settings: Dict[str, Any] = {}
    if _pipeline is not None:
        try:
            corpus_files = len(_pipeline.get_corpus_files())
        except Exception:
            pass
        llm_settings = _pipeline.config.get_llm_settings()

    return HealthResponse(
        status="ok" if pipeline_started else "starting",
        pipeline_started=pipeline_started,
        corpus_files=corpus_files,
        default_model=llm_settings.get("default_model", ""),
        embedding_model=llm_settings.get("embedding_model", ""),
        api_base=llm_settings.get("api_base", ""),
        api_key_set=llm_settings.get("api_key_set", False),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
