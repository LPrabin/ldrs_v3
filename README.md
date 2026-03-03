# LDRS v3 — Hybrid RAG Deep Agent System

**Living Document RAG System, version 3** — a 6-stage retrieval-augmented
generation pipeline that combines hierarchical grep search, pgvector
semantic similarity, BM25 fusion ranking, and an agentic reasoning loop
with grounding verification. Designed for multi-document technical
knowledge bases with full Nepali/Devanagari Unicode support.

```
Python 3.12+   |   PostgreSQL 16 + pgvector   |   OpenAI-compatible API
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Quick Start](#quick-start)
4. [Configuration Reference](#configuration-reference)
5. [API Reference](#api-reference)
6. [Streamlit UI](#streamlit-ui)
7. [Module Reference](#module-reference)
8. [Indexing & File Watcher](#indexing--file-watcher)
9. [Virtual Filesystem (VFS)](#virtual-filesystem-vfs)
10. [Agent Tools & Function Calling](#agent-tools--function-calling)
11. [Grounding Verification](#grounding-verification)
12. [Monitoring & LangSmith](#monitoring--langsmith)
13. [Citation Format](#citation-format)
14. [Testing](#testing)
15. [Project Structure](#project-structure)
16. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

LDRS v3 processes user queries through a deterministic 6-stage pipeline
where each stage has a well-defined input/output contract. Three stages
involve LLM calls (Intent Classification, Agent Loop, Grounding); the
remaining stages are deterministic.

```
                          ┌──────────────────────────┐
                          │   Source .md Files        │
                          │   (docs/ directory)       │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │   File Watcher (Stage 0)  │
                          │   watchdog + debounce     │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │   PageIndex (md_to_tree)  │
                          │   .md → structure JSON    │
                          └────────────┬─────────────┘
                                       │
                     ┌─────────────────┼─────────────────┐
                     │                 │                  │
            ┌────────▼───────┐ ┌──────▼────────┐ ┌──────▼───────┐
            │ Structure JSON │ │  Embed into   │ │  Registry    │
            │ (results/)     │ │  pgvector     │ │  (JSON)      │
            └────────────────┘ └───────────────┘ └──────────────┘


  User Query ──────────────────────────────────────────────────────────
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 1: Intent Classification (LLM)               │
            │  → intent_type, selected_files, query_variants,     │
            │    pattern_hints, needs_db, likely_multihop          │
            └────────┬────────────────────────────────────────────┘
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 2: Parallel Retrieval                        │
            │  ┌──────────────┐    ┌──────────────────────┐       │
            │  │  TreeGrep    │    │  pgvector Similarity  │       │
            │  │  (pattern    │ ∥  │  (query_variants →    │       │
            │  │   hints)     │    │   cosine search)      │       │
            │  └──────┬───────┘    └──────────┬───────────┘       │
            │         └──────────┬────────────┘                   │
            │                    ▼                                │
            │         Section Pool (deduplicated)                 │
            └────────┬────────────────────────────────────────────┘
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 3: BM25 Fusion Ranking                       │
            │  score = (w_bm25 × BM25) + (w_vec × vector)        │
            │        + (w_grep × grep_density)                    │
            │  × recency_factor × tag_boost                       │
            │  Weights shift by intent_type                       │
            └────────┬────────────────────────────────────────────┘
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 4: VFS Population                            │
            │  /sessions/{id}/manifest.json                       │
            │  /sessions/{id}/retrieved/rank1__doc__section.md     │
            │  /sessions/{id}/conversation/                       │
            │  /sessions/{id}/working/scratchpad.md                │
            └────────┬────────────────────────────────────────────┘
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 5: Agent Loop (LLM + Function Calling)       │
            │  Read manifest → self-sufficiency check →            │
            │  read_section / fetch_section / scratchpad →          │
            │  cited answer synthesis                              │
            └────────┬────────────────────────────────────────────┘
                     │
            ┌────────▼────────────────────────────────────────────┐
            │  Stage 6: Grounding Verification (LLM)              │
            │  Extract claims → entailment check per claim →       │
            │  flag unsupported → caveat insertion →                │
            │  optional re-grounding loop                         │
            └────────┬────────────────────────────────────────────┘
                     │
                     ▼
              Final Verified Answer
```

### Key Design Decisions

- **OpenAI function calling** is used directly (not through a framework)
  for the agent loop, giving fine-grained control over tool execution,
  message history, and forced synthesis on iteration limits.

- **Dual retrieval** (TreeGrep + pgvector) ensures both keyword-exact and
  semantic matches are captured. BM25 fusion ranking merges these signals
  with intent-aware weight presets.

- **VFS-per-session** gives the agent a self-contained filesystem with
  ranked sections, conversation context, and a scratchpad. The agent
  navigates via `manifest.json` and reads sections selectively.

- **Grounding verification** runs after every agent answer. Each cited
  claim is checked via LLM entailment against its source section.
  Unsupported claims get caveats; high flag ratios trigger re-grounding.

- **NFC normalization** is applied at every text boundary for safe
  handling of Nepali/Devanagari characters.

---

## Pipeline Stages

### Stage 0: File Watcher

Monitors directories for `.md` file changes using the `watchdog` library.
On create/modify, the file is re-indexed (parsed, embedded, registered).
On delete, embeddings and registry entries are removed. A configurable
debounce (default 2 seconds) prevents rapid successive re-indexing.

### Stage 1: Intent Classification

A single LLM call receives the user query and the compact registry JSON.
It outputs a structured JSON with:

| Field | Description |
|---|---|
| `intent_type` | One of: `exact`, `conceptual`, `comparative`, `multihop`, `db_query`, `hybrid` |
| `selected_files` | Files likely to contain the answer, with confidence scores |
| `query_variants` | 2-4 rephrased queries for diverse retrieval |
| `pattern_hints` | `{literals, phrases, prefix_wildcards}` for TreeGrep |
| `needs_db` | Whether database access is needed |
| `likely_multihop` | Whether multi-hop reasoning is required |

**Fast paths**: empty registry returns defaults without an LLM call.

### Stage 2: Parallel Retrieval

Two retrieval methods run concurrently:

1. **TreeGrep** — searches structure JSONs using `pattern_hints` from
   Stage 1. Three-tier matching: title (3.0), summary (2.0), body (1.0).
   Supports exact substring and word-level matching with stop word
   filtering and configurable minimum match ratio (0.3).

2. **pgvector** — embeds each `query_variant` and runs cosine similarity
   search against the `sections` table. Multi-query results are
   deduplicated by `(doc_name, section_id)`, keeping the highest
   similarity score.

Results are merged into a unified `SectionCandidate` pool with combined
grep and vector signals.

### Stage 3: BM25 Fusion Ranking

Scores each candidate using a weighted fusion:

```
raw_score = (w_bm25 × BM25_norm) + (w_vector × similarity) + (w_grep × grep_density)
final_score = raw_score × recency_factor × tag_boost
```

**Weight presets by intent type:**

| Intent | BM25 | Vector | Grep |
|---|---|---|---|
| `exact` | 0.30 | 0.35 | 0.35 |
| `conceptual` | 0.25 | 0.55 | 0.20 |
| `comparative` | 0.35 | 0.40 | 0.25 |
| `multihop` | 0.35 | 0.40 | 0.25 |
| `db_query` | 0.30 | 0.40 | 0.30 |
| `hybrid` | 0.35 | 0.40 | 0.25 |

**Metadata boosts:**
- `recency_factor`: exponential decay with 365-day half-life, range [0.5, 1.0]
- `tag_boost`: fraction of query tokens matching registry tags, range [1.0, 1.5]

For `comparative` intent, a round-robin interleave ensures balanced
multi-file coverage.

### Stage 4: VFS Population

Creates a per-session directory with:
- `manifest.json` — ranked section list with metadata and scores
- `retrieved/` — individual `.md` files per ranked section
- `conversation/` — history summary and recent turns
- `db_context/` — optional database query results
- `working/scratchpad.md` — agent working memory

The manifest is the agent's primary navigation interface.

### Stage 5: Agent Loop

An iterative OpenAI function-calling loop:

1. The agent receives the query, intent type, and manifest
2. It calls tools (`read_section`, `fetch_section`, `write_scratchpad`, etc.)
3. Each iteration is an LLM call with `tool_choice="auto"`
4. When the LLM produces a text response without tool calls, the loop ends
5. If max iterations are reached, a forced synthesis prompt is sent

**Tool definitions** are provided in OpenAI function-calling format.
Only sections actually read via `read_section` or `fetch_section` may
be cited.

### Stage 6: Grounding Verification

Post-answer verification of every cited claim:

1. Extract `(claim, citation)` pairs from the answer text
2. Locate cited section content in the VFS
3. LLM entailment check: "Does this section support this claim?"
4. Flag unsupported claims; insert caveat text
5. If flag ratio > 0.4, set `re_grounded=True`; the pipeline re-runs
   Stage 5 with grounding feedback appended to the query

Flagged claims are logged to `results/hallucination_log.jsonl`.

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL + pgvector)
- An OpenAI-compatible API endpoint (local or remote)

### 1. Clone and install

```bash
git clone <repo-url> ldrs_v3
cd ldrs_v3
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e ".[dev]"
```

### 2. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

This starts `pgvector/pgvector:pg16` on port 5432 and runs
`scripts/init_db.sql` to create the `sections` and `hallucination_log`
tables with the pgvector extension.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API key, base URL, and other settings
```

### 4. Index documents

Place your `.md` source files in `docs/`, then index:

```bash
# Via API (start server first)
uvicorn api.server:app --host 0.0.0.0 --port 8001

# Then index via HTTP
curl -X POST http://localhost:8001/index-directory
```

Or programmatically:

```python
import asyncio
from agent.config import AgentConfig
from agent.pipeline import Pipeline

async def main():
    pipeline = Pipeline(AgentConfig())
    await pipeline.startup()
    results = await pipeline.index_directory("docs/")
    for r in results:
        print(f"{r.doc_name}: {r.node_count} nodes, {r.embedded_count} embedded")
    await pipeline.shutdown()

asyncio.run(main())
```

### 5. Query

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does OAuth2 token refresh work?"}'
```

### 6. Launch the UI

```bash
streamlit run ui/streamlit_app.py
```

---

## Configuration Reference

All configuration is centralized in `AgentConfig` (`agent/config.py`).
Values are read from environment variables with sensible defaults.
Constructor arguments override env values.

### LLM Settings

| Env Var | Field | Default | Description |
|---|---|---|---|
| `API_KEY` | `api_key` | `""` | API key for the OpenAI-compatible endpoint |
| `BASE_URL` | `base_url` | `http://localhost:8000/v1` | Base URL for chat and embeddings |
| `DEFAULT_MODEL` | `default_model` | `qwen3-vl` | Default model for chat completions |
| `EMBEDDING_MODEL` | `embedding_model` | `text-embedding-3-small` | Model name for embeddings |

### PostgreSQL

| Env Var | Field | Default | Description |
|---|---|---|---|
| `POSTGRES_HOST` | `postgres_host` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `postgres_port` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `postgres_db` | `ldrs_v3` | Database name |
| `POSTGRES_USER` | `postgres_user` | `ldrs` | Database user |
| `POSTGRES_PASSWORD` | `postgres_password` | `ldrs_secret` | Database password |
| — | `embedding_dim` | `1536` | Embedding vector dimension |

### LangSmith Monitoring

| Env Var | Field | Default | Description |
|---|---|---|---|
| `LANGSMITH_TRACING` | `langsmith_tracing` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | `langsmith_api_key` | `""` | LangSmith API key |
| `LANGSMITH_PROJECT` | `langsmith_project` | `ldrs-v3` | LangSmith project name |

### File Watcher

| Env Var | Field | Default | Description |
|---|---|---|---|
| `WATCH_DIRS` | `watch_dirs` | `./docs` | Comma-separated directories to monitor |
| `WATCH_DEBOUNCE` | `watch_debounce` | `2.0` | Debounce interval in seconds |

### Retrieval & Agent Tuning

| Env Var | Field | Default | Description |
|---|---|---|---|
| `MAX_VFS_SECTIONS` | `max_vfs_sections` | `15` | Max sections in VFS per query |
| — | `max_context_chars` | `15000` | Character budget for context |
| — | `max_grep_results` | `50` | Max TreeGrep results per document |
| `MAX_AGENT_ITERATIONS` | `max_agent_iterations` | `10` | Max agent loop iterations |

### Fusion Weights (Defaults)

| Field | Default | Description |
|---|---|---|
| `bm25_weight` | `0.4` | Default BM25 weight |
| `vector_weight` | `0.4` | Default vector similarity weight |
| `grep_weight` | `0.2` | Default grep density weight |

These defaults are overridden by the intent-specific presets in
`FusionRanker`.

### Directories

| Field | Default | Description |
|---|---|---|
| `results_dir` | `./results` | Structure JSONs and registry |
| `docs_dir` | `./docs` | Source `.md` files |
| `sessions_dir` | `./sessions` | VFS session data |
| `registry_path` | `./results/registry.json` | Auto-derived from `results_dir` |

### Derived Properties

- `postgres_dsn` — full PostgreSQL connection string
- `async_postgres_dsn` — async-compatible connection string (for asyncpg)

---

## API Reference

The FastAPI server runs on port 8001 (configurable via `PORT` env var).

### POST `/query`

Run the full 6-stage pipeline for a user query.

**Request body:**
```json
{
  "query": "How does OAuth2 token refresh work?",
  "conversation_summary": "optional prior conversation summary",
  "recent_turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "db_context": null,
  "cleanup_session": false
}
```

**Response:**
```json
{
  "answer": "OAuth2 token refresh works by...",
  "intent_type": "conceptual",
  "selected_files": ["authentication.md"],
  "candidates_count": 12,
  "ranked_count": 8,
  "session_id": "a1b2c3d4e5f6",
  "citations": ["authentication.md § Token Refresh"],
  "claims_checked": 3,
  "claims_supported": 3,
  "claims_flagged": 0,
  "re_grounded": false,
  "usage": {
    "total_input_tokens": 2500,
    "total_output_tokens": 800,
    "total_tokens": 3300,
    "total_cost_usd": 0.0,
    "total_llm_calls": 4,
    "total_query_time_ms": 3200.5,
    "stage_breakdown": {},
    "stage_timings_ms": {}
  },
  "total_time_ms": 3450.2,
  "success": true,
  "error": ""
}
```

### POST `/batch-query`

Run multiple queries sequentially.

**Request body:**
```json
{
  "queries": ["question 1", "question 2"],
  "conversation_summary": null,
  "cleanup_sessions": false
}
```

**Response:** `List[QueryResponse]`

### POST `/index`

Index a single markdown file.

**Request body:**
```json
{
  "md_path": "/absolute/path/to/document.md",
  "tags": ["auth", "security"],
  "summary": "OAuth2 authentication documentation",
  "if_thinning": false
}
```

**Response:**
```json
{
  "md_path": "/absolute/path/to/document.md",
  "doc_name": "document",
  "index_path": "results/document_structure.json",
  "node_count": 15,
  "section_count": 12,
  "embedded_count": 12,
  "success": true,
  "error": ""
}
```

### POST `/index-directory`

Index all `.md` files in a directory.

**Request body:**
```json
{
  "directory": null,
  "tags": null,
  "if_thinning": false
}
```

If `directory` is null, defaults to `config.docs_dir`.

**Response:** `List[IndexResponse]`

### GET `/corpus`

Get a summary of the current corpus.

**Response:**
```json
{
  "total_files": 5,
  "total_tokens": 25000,
  "total_nodes": 120,
  "files_with_embeddings": 5,
  "file_names": ["auth.md", "api.md", "setup.md"]
}
```

### GET `/corpus/stats`

Detailed corpus statistics including per-file info.

**Response:**
```json
{
  "summary": { "...corpus summary..." },
  "files": {
    "auth.md": {
      "summary": "OAuth2 flow documentation",
      "tags": ["auth"],
      "sections_count": 8,
      "size_tokens": 1200,
      "has_embeddings": true,
      "last_modified": "2026-02-01"
    }
  }
}
```

### POST `/corpus/rebuild`

Re-index all documents in the docs directory. Returns `List[IndexResponse]`.

### GET `/sessions`

List all VFS session IDs. Returns `List[str]`.

### DELETE `/sessions/{session_id}`

Delete a specific VFS session.

**Response:** `{"status": "deleted", "session_id": "..."}`

### POST `/sessions/cleanup`

Delete all VFS sessions.

**Response:** `{"status": "cleaned", "sessions_deleted": "5"}`

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "pipeline_started": true,
  "corpus_files": 5,
  "version": "3.0.0"
}
```

---

## Streamlit UI

The Streamlit chat interface (`ui/streamlit_app.py`) provides:

- **Chat interface** — send queries and view cited answers
- **Pipeline details** — expandable panel showing intent type, candidate
  counts, ranked counts, claims checked/flagged, timing
- **Corpus info** — sidebar showing file count, token count, node count
- **Usage stats** — expandable JSON view of per-stage token usage
- **Health status** — sidebar indicator of API connection status

Launch:
```bash
streamlit run ui/streamlit_app.py
```

The UI connects to the FastAPI server (default `http://localhost:8001`).
The API URL is configurable via the sidebar text input.

---

## Module Reference

All modules live under `agent/`. The package uses lazy imports via
`__getattr__` in `agent/__init__.py` to avoid pulling in heavy
dependencies on package load.

### `config.py` — AgentConfig

Centralized `@dataclass` configuration. All settings loaded from
environment variables with sensible defaults. Provides `postgres_dsn`
and `async_postgres_dsn` computed properties.

### `monitoring.py` — UsageTracker & LangSmith

- `setup_monitoring(config)` — configures LangSmith tracing env vars
- `UsageTracker` — per-query accumulator for token counts, latencies,
  and costs across all stages
- `LLMCallRecord` — individual LLM call record
- `StageTimer` — per-stage timing

### `registry.py` — Registry

JSON-based document registry following the AGENT_SYSTEM schema. Tracks:
- Document metadata (summary, tags, sections, token counts)
- Embedding status
- Last modified dates
- Index paths

Key methods:
- `add_file()` — add/update a file entry with structure tree
- `remove_file()` — remove a file entry
- `mark_embeddings()` — update embedding status
- `get_corpus_summary()` — aggregate corpus statistics
- `get_for_llm()` — compact version for Stage 1 LLM context
- `save()` — atomic write (tmp file + rename)

Uses NFC normalization on all text fields and tiktoken for token counting.

### `embedder.py` — Embedder

Async section-level embedding pipeline:
- PostgreSQL connection pool via `asyncpg` (min 2, max 10 connections)
- Batch embedding via OpenAI-compatible `/v1/embeddings` endpoint
- Upsert semantics (delete + insert per document)
- Cosine similarity search with optional `doc_names` scope
- Multi-query search with deduplication

### `indexer.py` — Indexer

End-to-end document indexing:
1. `md_to_tree()` — parse `.md` into structure tree via PageIndex
2. Save structure JSON to `results/`
3. Flatten tree into sections (with breadcrumbs)
4. Embed sections into pgvector
5. Register in Registry with metadata

Also handles `remove_file()` and `index_directory()`.

### `tree_grep.py` — TreeGrep

Hierarchical pattern search across structure JSON nodes. Three-tier
matching with configurable relevance scores:

| Field | Relevance Score |
|---|---|
| Title | 3.0 |
| Summary | 2.0 |
| Body text | 1.0 |

Two matching modes:
- **Tier 1**: Exact substring match (full score)
- **Tier 2**: Word-level match with ratio scaling (min ratio: 0.3)

Features:
- NFC normalization for Unicode safety
- Stop word filtering (200+ English stop words)
- Scope filtering by node_id or title
- Snippet extraction with configurable padding (60 chars)
- `search_from_hints()` for Intent Classifier pattern_hints
- Multi-pattern search with deduplication

### `watcher.py` — FileWatcher

File system monitoring using `watchdog`:
- Debounced event handling (configurable interval)
- Handles create, modify, delete, and move events
- Async indexing via `asyncio.run_coroutine_threadsafe()`
- Automatic registry timestamp updates
- Recursive directory watching

### `intent_classifier.py` — IntentClassifier

Stage 1 LLM call that produces:
- `IntentResult` with intent type, file selection, query variants
- `PatternHints` for TreeGrep routing
- `SelectedFile` list with confidence scores

Handles: malformed JSON, missing fields, invalid intent types. Falls
back to `conceptual` intent with original query on any failure.

### `retriever.py` — Retriever

Stage 2 parallel retrieval:
- Loads TreeGrep instances for selected files
- Runs grep and vector search concurrently via `asyncio.gather()`
- Merges results into `SectionCandidate` pool
- Deduplication by `(doc_name, section_id)`
- Combined signals: `grep_score`, `grep_hits`, `vector_similarity`

### `fusion_ranker.py` — FusionRanker

Stage 3 BM25 fusion ranking:
- BM25Okapi scoring from `rank-bm25` library
- BM25 normalization to [0, 1] range
- Intent-based weight presets
- Recency factor (exponential decay, 365-day half-life)
- Tag overlap boost
- Multi-file interleave for comparative intent
- Capped at `max_vfs_sections`

### `vfs.py` — VFS

Stage 4 virtual filesystem:
- Creates session directories with UUID-based IDs
- Writes ranked sections as individual `.md` files
- Builds `manifest.json` with metadata and score breakdowns
- Writes conversation context and db_context
- Supports on-demand section fetching (`add_fetched_section`)
- Session listing and cleanup

### `tools.py` — AgentTools

Five tools available to the agent during Stage 5:

| Tool | Description |
|---|---|
| `read_section(vfs_path)` | Read a section from the VFS manifest |
| `fetch_section(source_file, section_header)` | Pull additional section on demand |
| `search_conversation_history(query)` | Search prior conversation turns |
| `write_scratchpad(content)` | Write to private working memory |
| `read_scratchpad()` | Read working memory |

Tool definitions are generated in OpenAI function-calling format via
`get_tool_definitions()`. The `sections_read` property tracks which
sections were accessed (for citation validation).

### `agent_loop.py` — AgentLoop

Stage 5 iterative function-calling loop:
- System prompt defines the agent protocol (manifest-first, cite-inline)
- Iterates up to `max_agent_iterations` times
- Each iteration: LLM call with tools → execute tool calls → append results
- Terminates when the LLM produces content without tool calls
- Forced synthesis on iteration limit
- Citation extraction via regex: `[source: file.md § Section]`
- Records per-iteration token usage

### `grounding.py` — GroundingVerifier

Stage 6 grounding verification:
- Extracts `(claim, citation)` pairs from answer text
- Locates cited source content in the VFS manifest
- Per-claim LLM entailment check (strict verification prompt)
- Flags unsupported claims (max 10 claims checked per answer)
- Caveat insertion for flagged claims
- Re-grounding trigger at >40% flag ratio
- Hallucination logging to `results/hallucination_log.jsonl`
- Handles JSON parse failures and verification errors gracefully

### `pipeline.py` — Pipeline

End-to-end orchestrator:
- Lazy component initialization (all components created on first use)
- `startup()` / `shutdown()` lifecycle management
- `query()` runs all 6 stages sequentially
- Re-grounding loop with feedback to the agent
- Conversation state management (last 20 turns)
- Index pass-through (`index_file`, `index_directory`)
- Corpus info and session management utilities

---

## Indexing & File Watcher

### Document Format

Source documents are Markdown files. PageIndex parses them into a
hierarchical structure tree based on heading levels.

### Indexing Pipeline

```
.md file → md_to_tree() → structure JSON → flatten → embed → register
```

1. **PageIndex parsing** — `md_to_tree()` converts Markdown headings into
   a tree structure with node IDs, titles, text content, and page ranges.

2. **Structure JSON** — saved to `results/<doc_name>_structure.json`
   for TreeGrep and the fetch_section tool.

3. **Flattening** — the tree is flattened into a list of section dicts
   with `{node_id, title, text, line_num, breadcrumb}`. Only nodes with
   non-empty text are included.

4. **Embedding** — sections are batch-embedded via the OpenAI-compatible
   embeddings endpoint and upserted into the `sections` table in
   PostgreSQL with pgvector.

5. **Registration** — the document is added to `registry.json` with
   metadata (summary, tags, sections, token count, embedding status).

### File Watcher

The watcher monitors directories for `.md` file changes:

```python
from agent.watcher import FileWatcher
from agent.config import AgentConfig

config = AgentConfig()
watcher = FileWatcher(config)
await watcher.start()   # starts watchdog observer
# ... application runs ...
await watcher.stop()     # stops and cleans up
```

**Events handled:**
- `created` / `modified` → re-index the file
- `deleted` → remove embeddings + registry entry + structure JSON
- `moved` → treat source as deleted, destination as created

---

## Virtual Filesystem (VFS)

Each query creates a VFS session directory:

```
sessions/{session_id}/
  manifest.json              ← agent reads this first
  retrieved/
    rank1__docname__section.md
    rank2__docname__section.md
    ...
  conversation/
    history_summary.md        ← conversation context
    recent_turns.json         ← last N turns
  db_context/
    relevant_records.json     ← optional database results
  working/
    scratchpad.md             ← agent working memory
```

### manifest.json

```json
{
  "session_id": "a1b2c3d4e5f6",
  "created_at": "2026-02-28T10:00:00Z",
  "intent_type": "conceptual",
  "query_variants": ["original query", "variant 1"],
  "sections": [
    {
      "vfs_path": "retrieved/rank1__auth__oauth_flow.md",
      "source_file": "authentication.md",
      "section": "OAuth Flow",
      "one_line_summary": "Describes the OAuth2 authorization code flow...",
      "retrieval_method": "grep+vector",
      "final_score": 0.85,
      "score_breakdown": {
        "bm25": 0.7,
        "vector": 0.9,
        "grep_density": 0.3
      },
      "why_included": "vector=0.90; bm25=0.70; grep_density=0.30",
      "last_modified": "2026-02-01",
      "fetch_more_hint": false
    }
  ]
}
```

---

## Agent Tools & Function Calling

The agent has five tools available, defined in OpenAI function-calling
format:

### `read_section`

```json
{
  "name": "read_section",
  "parameters": {
    "vfs_path": "retrieved/rank1__auth__oauth_flow.md"
  }
}
```

Reads a section listed in the manifest. Only sections read via this tool
may be cited. The tool tracks all accessed paths in `sections_read`.

### `fetch_section`

```json
{
  "name": "fetch_section",
  "parameters": {
    "source_file": "authentication.md",
    "section_header": "Token Refresh"
  }
}
```

Pulls an additional section on demand from the structure JSON. The section
is added to the VFS `retrieved/` directory and manifest. Use when the
manifest hints at more content or a multihop reference is followed.

### `search_conversation_history`

```json
{
  "name": "search_conversation_history",
  "parameters": {
    "query": "what did we discuss about authentication"
  }
}
```

Searches prior conversation turns via keyword matching.

### `write_scratchpad`

```json
{
  "name": "write_scratchpad",
  "parameters": {
    "content": "## Reasoning\n..."
  }
}
```

Writes to the agent's private working memory. Content should follow the
structured format: `## Reasoning`, `## Key Facts`, `## Open Questions`,
`## Synthesis Plan`.

### `read_scratchpad`

No parameters. Reads the current scratchpad content.

---

## Grounding Verification

### Process

1. **Claim extraction** — regex matches sentences followed by
   `[source: ...]` citations
2. **Source lookup** — maps citations to VFS sections via manifest
3. **Entailment check** — LLM verifies each claim against its source
4. **Caveat insertion** — unsupported claims get a
   `[Note: This claim could not be fully verified...]` suffix
5. **Re-grounding** — if >40% of claims are flagged, the pipeline
   re-runs Stage 5 with explicit feedback about which claims failed

### Hallucination Log

Flagged claims are appended to `results/hallucination_log.jsonl`:

```json
{
  "timestamp": "2026-02-28T10:00:00Z",
  "session_id": "a1b2c3d4e5f6",
  "claim": "The refresh token expires after 30 days",
  "citation": "authentication.md § Token Refresh",
  "supported": false,
  "reason": "The source does not mention a 30-day expiry period"
}
```

### Cost Control

- Maximum 10 claims verified per answer (`MAX_CLAIMS_TO_VERIFY`)
- Source content truncated to 3000 chars per verification call
- Re-grounding runs at most once (`MAX_REGROUND_ATTEMPTS = 1`)

---

## Monitoring & LangSmith

### UsageTracker

Every query creates a `UsageTracker` that records:
- Per-call: stage, model, input/output tokens, latency, cost
- Per-stage: start/end timing
- Query-level: total query time

The `summary()` method returns a dict with:
```python
{
    "total_input_tokens": 2500,
    "total_output_tokens": 800,
    "total_tokens": 3300,
    "total_cost_usd": 0.0,
    "total_llm_calls": 4,
    "total_query_time_ms": 3200.5,
    "stage_breakdown": {
        "intent_classifier": {"input_tokens": 500, "output_tokens": 200, ...},
        "agent_loop": {"input_tokens": 1500, "output_tokens": 400, ...},
        "grounding": {"input_tokens": 500, "output_tokens": 200, ...},
    },
    "stage_timings_ms": {
        "intent_classifier": 850.2,
        "retrieval": 200.5,
        "fusion_ranking": 15.3,
        "vfs_population": 25.1,
        "agent_loop": 1800.0,
        "grounding": 310.5,
    },
}
```

### LangSmith Integration

Enable LangSmith tracing via environment variables:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=ldrs-v3
```

`setup_monitoring(config)` sets the appropriate environment variables
that LangSmith/LangChain read automatically. Must be called before any
LangChain model is instantiated (handled by `Pipeline.startup()`).

---

## Citation Format

The agent uses inline citations in its answers:

**Single source:**
```
OAuth2 uses refresh tokens to obtain new access tokens. [source: auth.md § Token Refresh]
```

**Multiple sources:**
```
Both OAuth2 and API keys support scoped permissions. [source: auth.md § Scopes, api_keys.md § Permissions]
```

Citations are extracted by the `AgentLoop._extract_citations()` method
using the regex pattern `\[source:\s*([^\]]+)\]`.

The grounding verifier parses citations in the format
`file.md § Section Name` to locate source content in the VFS.

---

## Testing

### Running Tests

```bash
# From the project root
python -m pytest tests/ -v

# Or with the development install
pytest tests/ -v
```

### Test Suite

- **74 tests** covering all modules and pipeline stages
- **All tests pass** in approximately 1.5 seconds
- **Fully offline** — no LLM or database calls
- Uses `unittest.mock.AsyncMock` for async mocking
- Uses `pytest-asyncio` with `asyncio_mode = "auto"`
- Test fixtures use `tmp_path` for filesystem isolation

### Test Coverage

Tests cover:
- `AgentConfig` — default values, environment overrides, DSN properties
- `UsageTracker` — recording, timing, summary computation
- `Registry` — add/remove files, save/load, corpus summary, LLM format
- `TreeGrep` — exact/word-level matching, regex, scope, multi-pattern,
  `search_from_hints`, deduplication
- `IntentClassifier` — response parsing, malformed JSON, empty registry
- `SectionCandidate` — signal merging
- `FusionRanker` — weight presets, multi-file interleave, recency/tag boosts
- `VFS` — session creation, manifest, section read/write, scratchpad,
  add fetched section, cleanup
- `AgentTools` — all five tools, error handling
- `AgentLoop` — citation extraction
- `GroundingVerifier` — claim extraction, verification parsing, caveating,
  flag logging
- `Pipeline` — lifecycle, conversation management, corpus info
- Indexer — section flattening, breadcrumbs, recursive node counting

### Test Data

- `tests/fixtures/` — 9 structure JSON files for TreeGrep testing
- `tests/markdown/` — 12 `.md` files for indexer testing

---

## Project Structure

```
ldrs_v3/
├── README.md                    ← this file
├── pyproject.toml               ← project metadata, dependencies, tool config
├── requirements.txt             ← pinned dependencies
├── .env.example                 ← environment variable template
├── .gitignore
├── docker-compose.yml           ← PostgreSQL + pgvector (pg16)
│
├── agent/                       ← core pipeline modules
│   ├── __init__.py              ← lazy imports, __all__ exports
│   ├── config.py                ← AgentConfig dataclass
│   ├── monitoring.py            ← UsageTracker, LangSmith setup
│   ├── registry.py              ← document registry (JSON)
│   ├── embedder.py              ← pgvector embedding pipeline
│   ├── indexer.py               ← md → tree → embed → register
│   ├── tree_grep.py             ← hierarchical pattern search
│   ├── watcher.py               ← file system watcher (watchdog)
│   ├── intent_classifier.py     ← Stage 1: intent + routing
│   ├── retriever.py             ← Stage 2: parallel retrieval
│   ├── fusion_ranker.py         ← Stage 3: BM25 fusion ranking
│   ├── vfs.py                   ← Stage 4: VFS population
│   ├── tools.py                 ← Stage 5: agent tools
│   ├── agent_loop.py            ← Stage 5: agent reasoning loop
│   ├── grounding.py             ← Stage 6: grounding verification
│   └── pipeline.py              ← end-to-end orchestrator
│
├── api/
│   ├── __init__.py
│   └── server.py                ← FastAPI server (11 endpoints)
│
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py         ← Streamlit chat interface
│
├── pageindex/                   ← PageIndex library (md_to_tree)
│   ├── __init__.py
│   ├── page_index_md.py         ← markdown → structure tree
│   ├── utils.py                 ← shared utilities
│   └── config.yaml              ← PageIndex configuration
│
├── scripts/
│   └── init_db.sql              ← PostgreSQL schema (sections + hallucination_log)
│
├── tests/
│   ├── test_ldrs_v3.py          ← 74 tests (all passing)
│   ├── fixtures/                ← 9 structure JSON test files
│   └── markdown/                ← 12 .md test files
│
├── docs/                        ← source .md files (user content)
├── results/                     ← runtime: structure JSONs + registry.json
└── sessions/                    ← runtime: VFS session directories
```

---

## Database Schema

PostgreSQL 16 with pgvector extension. Tables created by
`scripts/init_db.sql`:

### `sections` table

| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing ID |
| `doc_name` | `TEXT NOT NULL` | Document name |
| `section_id` | `TEXT NOT NULL` | node_id from PageIndex |
| `section_title` | `TEXT NOT NULL` | Section title |
| `source_file` | `TEXT NOT NULL` | Path to source `.md` file |
| `content` | `TEXT NOT NULL` | Full section text |
| `line_num` | `INTEGER` | Line number in source `.md` |
| `embedding` | `vector(1536)` | Embedding vector |
| `token_count` | `INTEGER DEFAULT 0` | Token count |
| `created_at` | `TIMESTAMPTZ` | Creation timestamp |
| `updated_at` | `TIMESTAMPTZ` | Update timestamp |

**Unique constraint:** `(doc_name, section_id)`

**Indexes:**
- `idx_sections_embedding` — IVFFlat cosine index (100 lists)
- `idx_sections_doc_name` — B-tree on `doc_name`

### `hallucination_log` table

| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing ID |
| `session_id` | `TEXT NOT NULL` | VFS session ID |
| `claim_text` | `TEXT NOT NULL` | The flagged claim |
| `cited_source` | `TEXT` | Citation reference |
| `cited_section` | `TEXT` | Section reference |
| `supported` | `BOOLEAN DEFAULT FALSE` | Verification result |
| `confidence` | `FLOAT` | Confidence score |
| `logged_at` | `TIMESTAMPTZ` | Timestamp |

---

## Troubleshooting

### PostgreSQL connection errors

Ensure Docker is running and the pgvector container is healthy:
```bash
docker compose ps
docker compose logs postgres
```

Verify the database is accessible:
```bash
psql postgresql://ldrs:ldrs_secret@localhost:5432/ldrs_v3
```

### Embedding dimension mismatch

The default embedding dimension is 1536 (matching `text-embedding-3-small`).
If using a different model, update both:
1. `embedding_dim` in `AgentConfig`
2. The `vector(1536)` column type in `init_db.sql`

### Empty search results

- Verify documents are indexed: `GET /corpus`
- Check that structure JSONs exist in `results/`
- Ensure embeddings were generated: `files_with_embeddings > 0`
- Check logs for embedding errors

### Agent loop hitting max iterations

Increase `MAX_AGENT_ITERATIONS` or check if the manifest has enough
sections. The agent may be trying to fetch sections that don't exist.

### LangSmith tracing not working

Ensure all three variables are set:
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<valid-key>
LANGSMITH_PROJECT=ldrs-v3
```

And that `setup_monitoring()` is called before any LLM calls
(handled automatically by `Pipeline.startup()`).

### Unicode / Nepali text issues

All text processing applies `unicodedata.normalize("NFC", text)` at
boundaries. If you see matching failures with Devanagari text, check
that:
1. Source files are saved as UTF-8
2. NFC normalization is applied before comparison
3. The database encoding is UTF-8

---

## License

MIT
