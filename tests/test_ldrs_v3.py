"""
LDRS v3 — Fresh Test Suite.

Covers all 15 agent modules + pipeline orchestrator + API server.
Tests are fully offline — no LLM calls, no database connections.
Uses pytest + pytest-asyncio with unittest.mock.

Usage::

    source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate
    cd /Users/urgensingtan/Desktop/ldrs_v3
    python -m pytest tests/test_ldrs_v3.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TEST_DIR, "fixtures")
MARKDOWN_DIR = os.path.join(TEST_DIR, "markdown")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    return str(tmp_path)


@pytest.fixture
def sample_structure():
    """Minimal structure JSON for testing."""
    return {
        "doc_name": "test_doc.md",
        "structure": [
            {
                "title": "Introduction",
                "start_index": 1,
                "end_index": 2,
                "node_id": "0001",
                "summary": "An introduction to the topic.",
            },
            {
                "title": "Methods",
                "start_index": 3,
                "end_index": 5,
                "node_id": "0002",
                "summary": "Research methods used.",
            },
            {
                "title": "Results",
                "start_index": 6,
                "end_index": 8,
                "node_id": "0003",
                "summary": "Key findings and results.",
                "nodes": [
                    {
                        "title": "Statistical Analysis",
                        "start_index": 6,
                        "end_index": 7,
                        "node_id": "0004",
                        "summary": "Detailed statistical analysis.",
                    },
                ],
            },
        ],
    }


@pytest.fixture
def sample_config(tmp_output_dir):
    """AgentConfig for testing with temp dirs."""
    from agent.config import AgentConfig

    return AgentConfig(
        api_key="test-key",
        api_base="http://localhost:9999/v1",
        default_model="test-model",
        embedding_model="test-embed",
        results_dir=tmp_output_dir,
        docs_dir=MARKDOWN_DIR,
        sessions_dir=os.path.join(tmp_output_dir, "sessions"),
        registry_path=os.path.join(tmp_output_dir, "registry.json"),
    )


@pytest.fixture
def sample_registry(sample_config, sample_structure):
    """Registry with one document loaded."""
    from agent.registry import Registry

    reg = Registry(sample_config.registry_path)
    reg.add_file(
        "test_doc.md",
        sample_structure["structure"],
        summary="A test document about profitability.",
        tags=["test", "banking"],
    )
    reg.save()
    return reg


@pytest.fixture
def earthmover_structure():
    """Load earthmover_structure.json from fixtures."""
    path = os.path.join(FIXTURES_DIR, "earthmover_structure.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def all_structures():
    """Load all structure JSONs from fixtures."""
    structures = {}
    for fname in os.listdir(FIXTURES_DIR):
        if fname.endswith("_structure.json"):
            path = os.path.join(FIXTURES_DIR, fname)
            with open(path, encoding="utf-8") as f:
                structures[fname] = json.load(f)
    return structures


# ============================================================================
# Test: Config (Phase 3)
# ============================================================================


class TestConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        from agent.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.default_model is not None
        assert cfg.embedding_dim == 1536
        assert cfg.max_vfs_sections == 15
        assert cfg.max_agent_iterations == 10

    def test_registry_path_derived(self, tmp_output_dir):
        from agent.config import AgentConfig

        cfg = AgentConfig(results_dir=tmp_output_dir)
        assert cfg.registry_path is not None
        assert "registry.json" in cfg.registry_path

    def test_postgres_dsn(self):
        from agent.config import AgentConfig

        cfg = AgentConfig(
            postgres_host="myhost",
            postgres_port=5433,
            postgres_db="mydb",
            postgres_user="myuser",
            postgres_password="mypw",
        )
        assert "myhost" in cfg.postgres_dsn
        assert "5433" in cfg.postgres_dsn
        assert "mydb" in cfg.postgres_dsn

    def test_custom_values(self):
        from agent.config import AgentConfig

        cfg = AgentConfig(
            api_key="custom-key",
            api_base="http://custom:8000/v1",
            default_model="custom-model",
            max_vfs_sections=20,
        )
        assert cfg.api_key == "custom-key"
        assert cfg.default_model == "custom-model"
        assert cfg.max_vfs_sections == 20


# ============================================================================
# Test: Monitoring (Phase 3)
# ============================================================================


class TestMonitoring:
    """Tests for UsageTracker."""

    def test_basic_tracking(self):
        from agent.monitoring import UsageTracker

        tracker = UsageTracker()
        tracker.start_query()
        tracker.start_stage("intent")

        tracker.record_llm_call(
            stage="intent",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            cost_usd=0.001,
        )

        tracker.end_stage("intent")
        tracker.end_query()

        summary = tracker.summary()
        assert summary["total_input_tokens"] == 100
        assert summary["total_output_tokens"] == 50
        assert summary["total_tokens"] == 150
        assert summary["total_llm_calls"] == 1

    def test_multi_stage_tracking(self):
        from agent.monitoring import UsageTracker

        tracker = UsageTracker()
        tracker.start_query()

        for stage in ["intent", "retrieval", "agent_loop"]:
            tracker.start_stage(stage)
            tracker.record_llm_call(
                stage=stage,
                model="test",
                input_tokens=50,
                output_tokens=25,
            )
            tracker.end_stage(stage)

        tracker.end_query()
        summary = tracker.summary()
        assert summary["total_llm_calls"] == 3
        assert summary["total_tokens"] == 225
        assert len(summary["stage_timings_ms"]) == 3

    def test_setup_monitoring(self, sample_config):
        from agent.monitoring import setup_monitoring

        setup_monitoring(sample_config)
        # Should not raise


# ============================================================================
# Test: Registry (Phase 5)
# ============================================================================


class TestRegistry:
    """Tests for Registry."""

    def test_create_empty(self, tmp_output_dir):
        from agent.registry import Registry

        path = os.path.join(tmp_output_dir, "registry.json")
        reg = Registry(path)
        assert len(reg.files) == 0

    def test_add_file(self, sample_registry, sample_structure):
        assert "test_doc.md" in sample_registry.files
        entry = sample_registry.files["test_doc.md"]
        assert entry["summary"] == "A test document about profitability."
        assert "test" in entry["tags"]
        assert len(entry["sections"]) > 0

    def test_remove_file(self, sample_registry):
        assert sample_registry.remove_file("test_doc.md")
        assert "test_doc.md" not in sample_registry.files

    def test_remove_nonexistent(self, sample_registry):
        assert not sample_registry.remove_file("nonexistent.md")

    def test_save_and_reload(self, sample_registry, sample_config):
        from agent.registry import Registry

        sample_registry.save()
        reg2 = Registry(sample_config.registry_path)
        assert "test_doc.md" in reg2.files
        assert reg2.files["test_doc.md"]["summary"] == "A test document about profitability."

    def test_corpus_summary(self, sample_registry):
        summary = sample_registry.get_corpus_summary()
        assert summary["total_files"] == 1
        assert "test_doc.md" in summary["file_names"]

    def test_get_for_llm(self, sample_registry):
        llm_view = sample_registry.get_for_llm()
        assert "files" in llm_view
        assert "test_doc.md" in llm_view["files"]
        entry = llm_view["files"]["test_doc.md"]
        assert "summary" in entry

    def test_mark_embeddings(self, sample_registry):
        sample_registry.mark_embeddings("test_doc.md", True)
        assert sample_registry.files["test_doc.md"]["has_embeddings"] is True

    def test_multiple_files(self, tmp_output_dir):
        from agent.registry import Registry

        path = os.path.join(tmp_output_dir, "registry.json")
        reg = Registry(path)
        for i in range(5):
            reg.add_file(
                f"doc_{i}.md",
                [{"title": f"Section {i}", "node_id": f"000{i}", "start_index": 1, "end_index": 1}],
                summary=f"Document {i}",
                tags=[f"tag_{i}"],
            )
        assert len(reg.files) == 5
        summary = reg.get_corpus_summary()
        assert summary["total_files"] == 5


# ============================================================================
# Test: TreeGrep (Phase 7b)
# ============================================================================


class TestTreeGrep:
    """Tests for TreeGrep."""

    def test_load_structure(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        assert tg.doc_name == "test_doc.md"
        assert len(tg.structure) > 0

    def test_search_title_match(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        results = tg.search("Introduction")
        assert len(results) > 0
        assert results[0].title == "Introduction"
        assert results[0].relevance_score == 3.0  # title match

    def test_search_summary_match(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        results = tg.search("findings")
        # Should match "Key findings and results" in summary of Results
        has_summary_match = any(r.matched_field == "summary" for r in results)
        assert has_summary_match or len(results) > 0

    def test_search_no_match(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        results = tg.search("xyznonexistent123")
        assert len(results) == 0

    def test_search_multi(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        results = tg.search_multi(["Introduction", "Methods"])
        assert len(results) >= 2

    def test_search_from_hints(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        hints = {
            "literals": ["Introduction"],
            "phrases": ["statistical analysis"],
            "prefix_wildcards": [],
        }
        results = tg.search_from_hints(hints)
        assert len(results) > 0

    def test_earthmover_search(self, earthmover_structure, tmp_output_dir):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "earthmover_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(earthmover_structure, f)

        tg = TreeGrep(path)
        results = tg.search("Earth Mover")
        assert len(results) > 0
        # Should find the main title
        titles = [r.title for r in results]
        assert any("Earth Mover" in t for t in titles)

    def test_max_results(self, tmp_output_dir, sample_structure):
        from agent.tree_grep import TreeGrep

        path = os.path.join(tmp_output_dir, "test_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        tg = TreeGrep(path)
        results = tg.search("a", max_results=2)
        assert len(results) <= 2

    def test_all_fixture_structures(self, all_structures, tmp_output_dir):
        """Verify TreeGrep loads all 9 fixture structures."""
        from agent.tree_grep import TreeGrep

        for fname, structure in all_structures.items():
            path = os.path.join(tmp_output_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(structure, f)
            tg = TreeGrep(path)
            assert tg.doc_name == structure["doc_name"]
            assert len(tg.structure) > 0


# ============================================================================
# Test: Intent Classifier (Phase 9)
# ============================================================================


class TestIntentClassifier:
    """Tests for IntentClassifier — all mocked."""

    @pytest.mark.asyncio
    async def test_empty_registry(self, sample_config):
        """Empty registry → default result, no LLM call."""
        from agent.intent_classifier import IntentClassifier

        clf = IntentClassifier(sample_config)
        result = await clf.classify(
            query="test query",
            registry={},
        )
        assert result.intent_type == "conceptual"
        assert len(result.query_variants) >= 1

    @pytest.mark.asyncio
    async def test_classify_with_mock(self, sample_config):
        """Mock LLM response for classification."""
        from agent.intent_classifier import IntentClassifier

        llm_response = json.dumps(
            {
                "intent_type": "exact",
                "selected_files": [{"path": "auth.md", "confidence": 0.9}],
                "query_variants": ["OAuth2 refresh flow", "token renewal"],
                "pattern_hints": {
                    "literals": ["OAuth2"],
                    "phrases": ["token refresh"],
                    "prefix_wildcards": [],
                },
                "needs_db": False,
                "likely_multihop": False,
            }
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = llm_response
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        clf = IntentClassifier(sample_config)

        registry_data = {
            "files": {
                "auth.md": {"summary": "Auth system", "tags": ["auth"], "sections": ["OAuth"]}
            }
        }

        with patch(
            "agent.intent_classifier.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await clf.classify(
                query="How does OAuth2 token refresh work?",
                registry=registry_data,
            )

        assert result.intent_type == "exact"
        assert len(result.selected_files) == 1
        assert result.selected_files[0].path == "auth.md"
        assert "OAuth2" in result.pattern_hints.literals

    @pytest.mark.asyncio
    async def test_intent_result_properties(self, sample_config):
        from agent.intent_classifier import IntentResult, SelectedFile, PatternHints

        result = IntentResult(
            intent_type="comparative",
            selected_files=[
                SelectedFile(path="a.md", confidence=0.8),
                SelectedFile(path="b.md", confidence=0.7),
            ],
            query_variants=["compare A and B"],
            pattern_hints=PatternHints(
                literals=["A", "B"],
                phrases=[],
                prefix_wildcards=[],
            ),
        )
        assert result.selected_paths == ["a.md", "b.md"]


# ============================================================================
# Test: Retriever (Phase 10)
# ============================================================================


class TestRetriever:
    """Tests for Retriever — mocked embedder."""

    @pytest.mark.asyncio
    async def test_retrieve_with_mock(self, sample_config, sample_structure, tmp_output_dir):
        from agent.intent_classifier import IntentResult, PatternHints, SelectedFile
        from agent.retriever import Retriever, SectionCandidate

        # Write structure file for TreeGrep
        struct_path = os.path.join(sample_config.results_dir, "test_doc_structure.json")
        with open(struct_path, "w", encoding="utf-8") as f:
            json.dump(sample_structure, f)

        # Setup registry for retriever to find structure files
        from agent.registry import Registry

        reg = Registry(sample_config.registry_path)
        reg.add_file(
            "test_doc.md",
            sample_structure["structure"],
            index_path=struct_path,
        )
        reg.save()

        # Mock embedder
        mock_embedder = AsyncMock()
        mock_embedder.search_multi = AsyncMock(return_value=[])

        retriever = Retriever(sample_config, embedder=mock_embedder)

        intent = IntentResult(
            intent_type="exact",
            selected_files=[SelectedFile(path="test_doc.md", confidence=0.9)],
            query_variants=["Introduction topic"],
            pattern_hints=PatternHints(
                literals=["Introduction"],
                phrases=[],
                prefix_wildcards=[],
            ),
        )

        candidates = await retriever.retrieve(intent=intent)
        # Should have grep results (TreeGrep finds "Introduction")
        assert isinstance(candidates, list)


# ============================================================================
# Test: FusionRanker (Phase 11)
# ============================================================================


class TestFusionRanker:
    """Tests for FusionRanker — no external deps."""

    def test_rank_basic(self, sample_config):
        from agent.fusion_ranker import FusionRanker
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.retriever import SectionCandidate

        ranker = FusionRanker(sample_config)

        candidates = [
            SectionCandidate(
                doc_name="doc.md",
                section_id="0001",
                section_title="Introduction",
                content="This is the introduction to the study of banking.",
                source_file="doc.md",
                grep_score=3.0,
                grep_hits=2,
                vector_similarity=0.85,
                retrieval_methods=["grep", "vector"],
            ),
            SectionCandidate(
                doc_name="doc.md",
                section_id="0002",
                section_title="Methods",
                content="Research methods overview for the study.",
                source_file="doc.md",
                grep_score=1.0,
                grep_hits=1,
                vector_similarity=0.60,
                retrieval_methods=["vector"],
            ),
        ]

        intent = IntentResult(
            intent_type="conceptual",
            selected_files=[],
            query_variants=["banking study"],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )

        ranked = ranker.rank(candidates=candidates, intent=intent)
        assert len(ranked) == 2
        # Higher-scoring section should be first
        assert ranked[0].final_score >= ranked[1].final_score
        assert ranked[0].section_title == "Introduction"

    def test_rank_empty(self, sample_config):
        from agent.fusion_ranker import FusionRanker
        from agent.intent_classifier import IntentResult, PatternHints

        ranker = FusionRanker(sample_config)

        intent = IntentResult(
            intent_type="exact",
            selected_files=[],
            query_variants=[],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )
        ranked = ranker.rank(candidates=[], intent=intent)
        assert ranked == []

    def test_rank_respects_max_sections(self, sample_config):
        from agent.fusion_ranker import FusionRanker
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.retriever import SectionCandidate

        sample_config.max_vfs_sections = 3
        ranker = FusionRanker(sample_config)

        candidates = [
            SectionCandidate(
                doc_name="doc.md",
                section_id=f"000{i}",
                section_title=f"Section {i}",
                content=f"Content for section {i} with enough text to be useful.",
                source_file="doc.md",
                grep_score=float(i),
                vector_similarity=0.5,
                retrieval_methods=["grep"],
            )
            for i in range(10)
        ]

        intent = IntentResult(
            intent_type="exact",
            selected_files=[],
            query_variants=[],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )

        ranked = ranker.rank(candidates=candidates, intent=intent)
        assert len(ranked) <= 3


# ============================================================================
# Test: VFS (Phase 12)
# ============================================================================


class TestVFS:
    """Tests for VFS — fully filesystem-based."""

    def test_create_session(self, sample_config):
        from agent.fusion_ranker import RankedSection
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)

        ranked = [
            RankedSection(
                doc_name="doc.md",
                section_id="0001",
                section_title="Introduction",
                content="This is the introduction content with enough text.",
                source_file="doc.md",
                final_score=0.9,
                bm25_score=0.5,
                vector_score=0.8,
                retrieval_method="grep+vector",
            ),
        ]

        intent = IntentResult(
            intent_type="conceptual",
            selected_files=[],
            query_variants=["test query"],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )

        session = vfs.create_session(ranked_sections=ranked, intent=intent)
        assert session.session_id != ""
        assert session.section_count == 1
        assert os.path.exists(session.manifest_path)

    def test_read_manifest(self, sample_config):
        from agent.fusion_ranker import RankedSection
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)

        ranked = [
            RankedSection(
                doc_name="doc.md",
                section_id="0001",
                section_title="Intro",
                content="Introduction content here.",
                source_file="doc.md",
                final_score=0.9,
                retrieval_method="grep",
            ),
        ]

        intent = IntentResult(
            intent_type="exact",
            selected_files=[],
            query_variants=[],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )

        session = vfs.create_session(ranked_sections=ranked, intent=intent)
        manifest = vfs.read_manifest(session.session_id)
        assert "sections" in manifest
        assert len(manifest["sections"]) == 1

    def test_scratchpad(self, sample_config):
        from agent.fusion_ranker import RankedSection
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)

        session = vfs.create_session(
            ranked_sections=[],
            intent=IntentResult(
                intent_type="exact",
                selected_files=[],
                query_variants=[],
                pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
            ),
        )

        vfs.write_scratchpad(session.session_id, "test notes")
        content = vfs.read_scratchpad(session.session_id)
        assert content == "test notes"

    def test_cleanup_session(self, sample_config):
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        session = vfs.create_session(
            ranked_sections=[],
            intent=IntentResult(
                intent_type="exact",
                selected_files=[],
                query_variants=[],
                pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
            ),
        )
        sid = session.session_id
        assert sid in vfs.list_sessions()
        vfs.cleanup_session(sid)
        assert sid not in vfs.list_sessions()

    def test_list_sessions(self, sample_config):
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        sessions_before = len(vfs.list_sessions())

        vfs.create_session(
            ranked_sections=[],
            intent=IntentResult(
                intent_type="exact",
                selected_files=[],
                query_variants=[],
                pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
            ),
        )
        assert len(vfs.list_sessions()) == sessions_before + 1


# ============================================================================
# Test: Agent Tools (Phase 13)
# ============================================================================


class TestAgentTools:
    """Tests for AgentTools."""

    def test_get_tool_definitions(self, sample_config):
        from agent.tools import AgentTools
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        tools = AgentTools(config=sample_config, vfs=vfs, session_id="dummy")

        defs = tools.get_tool_definitions()
        assert len(defs) == 5
        names = [d["function"]["name"] for d in defs]
        assert "read_section" in names
        assert "fetch_section" in names
        assert "search_conversation_history" in names
        assert "write_scratchpad" in names
        assert "read_scratchpad" in names

    def test_scratchpad_via_tools(self, sample_config):
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.tools import AgentTools
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        session = vfs.create_session(
            ranked_sections=[],
            intent=IntentResult(
                intent_type="exact",
                selected_files=[],
                query_variants=[],
                pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
            ),
        )

        tools = AgentTools(config=sample_config, vfs=vfs, session_id=session.session_id)

        result = tools.write_scratchpad("Test scratchpad content")
        assert (
            "success" in result.lower()
            or "written" in result.lower()
            or "scratchpad" in result.lower()
        )

        content = tools.read_scratchpad()
        assert "Test scratchpad content" in content


# ============================================================================
# Test: Agent Loop (Phase 14)
# ============================================================================


class TestAgentLoop:
    """Tests for AgentLoop — mocked LLM."""

    @pytest.mark.asyncio
    async def test_run_simple_answer(self, sample_config):
        from agent.agent_loop import AgentLoop, AgentResult
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        session = vfs.create_session(
            ranked_sections=[],
            intent=IntentResult(
                intent_type="exact",
                selected_files=[],
                query_variants=[],
                pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
            ),
        )

        agent = AgentLoop(sample_config, vfs=vfs)

        # Mock LLM to return a direct answer (no tool calls)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "The answer is 42. [source: doc.md § Section 1]"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100)

        with patch(
            "agent.agent_loop.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                query="What is the answer?",
                session_id=session.session_id,
                intent_type="exact",
            )

        assert isinstance(result, AgentResult)
        assert "42" in result.answer
        assert len(result.citations) > 0
        assert "doc.md § Section 1" in result.citations

    @pytest.mark.asyncio
    async def test_extract_citations(self, sample_config):
        from agent.agent_loop import AgentLoop
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        agent = AgentLoop(sample_config, vfs=vfs)

        citations = agent._extract_citations(
            "Some claim [source: auth.md § OAuth Flow]. "
            "Another claim [source: billing.md § Payments, auth.md § Tokens]."
        )
        assert "auth.md § OAuth Flow" in citations
        assert "billing.md § Payments" in citations
        assert "auth.md § Tokens" in citations


# ============================================================================
# Test: Grounding Verifier (Phase 15)
# ============================================================================


class TestGroundingVerifier:
    """Tests for GroundingVerifier — mocked LLM."""

    def test_extract_claims(self, sample_config):
        from agent.grounding import GroundingVerifier
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        verifier = GroundingVerifier(sample_config, vfs=vfs)

        claims = verifier._extract_claims(
            "OAuth2 uses refresh tokens for renewal. [source: auth.md § OAuth Flow] "
            "The system supports SSO. [source: auth.md § SSO]"
        )
        assert len(claims) == 2
        assert claims[0][0].strip().endswith("renewal.")
        assert claims[0][1] == "auth.md § OAuth Flow"

    def test_extract_claims_empty(self, sample_config):
        from agent.grounding import GroundingVerifier
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        verifier = GroundingVerifier(sample_config, vfs=vfs)
        claims = verifier._extract_claims("No citations here.")
        assert len(claims) == 0

    @pytest.mark.asyncio
    async def test_verify_no_claims(self, sample_config):
        from agent.grounding import GroundingVerifier
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        verifier = GroundingVerifier(sample_config, vfs=vfs)

        result = await verifier.verify(
            answer="A simple answer with no citations.",
            session_id="dummy",
        )
        assert result.claims_checked == 0
        assert result.verified_answer == "A simple answer with no citations."
        assert not result.re_grounded

    def test_caveat_unsupported(self, sample_config):
        from agent.grounding import ClaimVerification, GroundingVerifier
        from agent.vfs import VFS

        vfs = VFS(sample_config)
        verifier = GroundingVerifier(sample_config, vfs=vfs)

        answer = "OAuth2 uses refresh tokens. The sky is green."
        flags = [
            ClaimVerification(
                claim="The sky is green.",
                citation="doc.md § Colors",
                source_content="",
                supported=False,
                reason="Not supported",
            ),
        ]

        result = verifier._caveat_unsupported(answer, flags)
        assert "[Note:" in result
        assert "OAuth2 uses refresh tokens." in result  # unchanged


# ============================================================================
# Test: Pipeline Orchestrator (Phase 16)
# ============================================================================


class TestPipeline:
    """Tests for Pipeline — fully mocked."""

    def test_pipeline_init(self, sample_config):
        from agent.pipeline import Pipeline

        pipeline = Pipeline(sample_config)
        assert pipeline.config == sample_config
        assert not pipeline._started

    def test_pipeline_lazy_properties(self, sample_config):
        from agent.pipeline import Pipeline

        pipeline = Pipeline(sample_config)
        # Accessing properties should create instances lazily
        assert pipeline.registry is not None
        assert pipeline.vfs is not None
        assert pipeline.fusion_ranker is not None

    @pytest.mark.asyncio
    async def test_pipeline_query_mocked(self, sample_config):
        from agent.agent_loop import AgentResult
        from agent.grounding import GroundingResult
        from agent.intent_classifier import IntentResult, PatternHints
        from agent.pipeline import Pipeline

        pipeline = Pipeline(sample_config)

        # Mock all stage methods
        mock_intent = IntentResult(
            intent_type="conceptual",
            selected_files=[],
            query_variants=["test"],
            pattern_hints=PatternHints(literals=[], phrases=[], prefix_wildcards=[]),
        )

        pipeline._stage_intent = AsyncMock(return_value=mock_intent)
        pipeline._stage_retrieve = AsyncMock(return_value=[])
        pipeline._stage_rank = MagicMock(return_value=[])
        pipeline._stage_vfs = MagicMock(
            return_value=MagicMock(
                session_id="test-session",
                section_count=0,
            )
        )
        pipeline._stage_agent = AsyncMock(
            return_value=AgentResult(
                answer="The answer is here.",
                citations=[],
            )
        )
        pipeline._stage_grounding = AsyncMock(
            return_value=GroundingResult(
                original_answer="The answer is here.",
                verified_answer="The answer is here.",
                re_grounded=False,
            )
        )

        # Mark as started to skip startup
        pipeline._started = True

        result = await pipeline.query("Test question")
        assert result.success
        assert result.answer == "The answer is here."
        assert result.intent is not None
        assert result.total_time_ms > 0

    def test_pipeline_conversation_management(self, sample_config):
        from agent.pipeline import Pipeline

        pipeline = Pipeline(sample_config)
        pipeline._update_conversation("Hello", "Hi there!")
        assert len(pipeline._conversation_history) == 2
        assert pipeline._conversation_summary != ""

        pipeline.clear_conversation()
        assert len(pipeline._conversation_history) == 0
        assert pipeline._conversation_summary == ""

    def test_pipeline_result_fields(self):
        from agent.pipeline import PipelineResult

        result = PipelineResult(
            answer="Test answer",
            success=True,
            total_time_ms=123.4,
        )
        assert result.answer == "Test answer"
        assert result.success
        assert result.error == ""

    def test_corpus_methods(self, sample_config, sample_structure):
        from agent.pipeline import Pipeline
        from agent.registry import Registry

        pipeline = Pipeline(sample_config)
        # Add a file to registry
        pipeline.registry.add_file(
            "test.md",
            sample_structure["structure"],
            summary="Test doc",
        )

        summary = pipeline.get_corpus_summary()
        assert summary["total_files"] == 1

        files = pipeline.get_corpus_files()
        assert "test.md" in files


# ============================================================================
# Test: API Server (Phase 17)
# ============================================================================


class TestAPIServer:
    """Tests for FastAPI server — import and route registration."""

    def test_app_imports(self):
        from api.server import app

        assert app is not None
        assert app.title == "LDRS v3"

    def test_routes_registered(self):
        from api.server import app

        routes = {}
        for route in app.routes:
            if hasattr(route, "methods"):
                for method in route.methods:
                    routes.setdefault(route.path, []).append(method)

        assert "POST" in routes.get("/query", [])
        assert "POST" in routes.get("/batch-query", [])
        assert "POST" in routes.get("/index", [])
        assert "GET" in routes.get("/corpus", [])
        assert "GET" in routes.get("/corpus/stats", [])
        assert "POST" in routes.get("/corpus/rebuild", [])
        assert "GET" in routes.get("/sessions", [])
        assert "GET" in routes.get("/health", [])

    def test_request_models(self):
        from api.server import (
            BatchQueryRequest,
            IndexRequest,
            QueryRequest,
            QueryResponse,
        )

        # Verify models can be instantiated
        qr = QueryRequest(query="test")
        assert qr.query == "test"

        ir = IndexRequest(md_path="/tmp/test.md")
        assert ir.md_path == "/tmp/test.md"

    def test_health_without_pipeline(self):
        import api.server as srv

        # When pipeline is not initialized
        old = srv._pipeline
        srv._pipeline = None

        from api.server import HealthResponse

        # Can't call the endpoint directly without async, just verify model
        resp = HealthResponse(
            status="starting",
            pipeline_started=False,
            corpus_files=0,
        )
        assert resp.status == "starting"
        srv._pipeline = old


# ============================================================================
# Test: Data classes and edge cases
# ============================================================================


class TestDataClasses:
    """Test data class consistency across modules."""

    def test_section_candidate(self):
        from agent.retriever import SectionCandidate

        sc = SectionCandidate(
            doc_name="doc.md",
            section_id="0001",
            section_title="Intro",
            content="Content here",
            source_file="doc.md",
            retrieval_methods=["grep", "vector"],
        )
        assert sc.retrieval_method_str in ("grep+vector", "vector+grep")

    def test_ranked_section(self):
        from agent.fusion_ranker import RankedSection

        rs = RankedSection(
            doc_name="doc.md",
            section_id="0001",
            section_title="Intro",
            content="Content",
            source_file="doc.md",
            final_score=0.95,
        )
        assert rs.final_score == 0.95

    def test_manifest_entry(self):
        from agent.vfs import ManifestEntry

        me = ManifestEntry(
            vfs_path="retrieved/doc_0001.md",
            source_file="doc.md",
            section="Introduction",
            one_line_summary="Intro section",
            retrieval_method="grep",
            final_score=0.9,
            score_breakdown={"bm25": 0.5, "vector": 0.4},
            why_included="Top ranked",
        )
        d = me.to_dict()
        assert d["vfs_path"] == "retrieved/doc_0001.md"
        assert d["final_score"] == 0.9

    def test_agent_result(self):
        from agent.agent_loop import AgentResult

        ar = AgentResult(
            answer="Answer text",
            citations=["doc.md § Section"],
            sections_read=["retrieved/doc_0001.md"],
            tool_calls_made=3,
            iterations=2,
        )
        assert ar.tool_calls_made == 3
        assert len(ar.citations) == 1

    def test_grounding_result(self):
        from agent.grounding import ClaimVerification, GroundingResult

        gr = GroundingResult(
            original_answer="Original",
            verified_answer="Verified",
            claims_checked=5,
            claims_supported=4,
            claims_flagged=1,
            flags=[
                ClaimVerification(
                    claim="Bad claim",
                    citation="doc.md § S1",
                    source_content="",
                    supported=False,
                    reason="Not found",
                )
            ],
            all_verifications=[],
            re_grounded=False,
        )
        assert gr.claims_flagged == 1
        assert not gr.re_grounded

    def test_pipeline_result(self):
        from agent.pipeline import PipelineResult

        pr = PipelineResult(
            answer="Final answer",
            success=True,
            total_time_ms=1500.0,
            usage={"total_tokens": 500},
        )
        assert pr.answer == "Final answer"
        assert pr.usage["total_tokens"] == 500

    def test_index_result(self):
        from agent.indexer import IndexResult

        ir = IndexResult(
            md_path="/tmp/doc.md",
            doc_name="doc.md",
            index_path="/tmp/doc_structure.json",
            node_count=10,
            section_count=5,
            embedded_count=5,
            success=True,
        )
        assert ir.success
        assert ir.node_count == 10


# ============================================================================
# Test: Package-level imports
# ============================================================================


class TestPackageImports:
    """Verify __init__.py lazy imports work."""

    def test_lazy_pipeline(self):
        import agent

        assert agent.Pipeline is not None

    def test_lazy_config(self):
        import agent

        assert agent.AgentConfig is not None

    def test_lazy_tracker(self):
        import agent

        assert agent.UsageTracker is not None

    def test_lazy_registry(self):
        import agent

        assert agent.Registry is not None

    def test_all_exports(self):
        import agent

        for name in agent.__all__:
            obj = getattr(agent, name)
            assert obj is not None, f"Failed to import {name}"


# ============================================================================
# Test: NFC normalization in TreeGrep
# ============================================================================


class TestNFCNormalization:
    """Test NFC normalization for Nepali/Devanagari text."""

    def test_nfc_in_treegrep(self, tmp_output_dir):
        import unicodedata
        from agent.tree_grep import TreeGrep

        structure = {
            "doc_name": "nepali_doc.md",
            "structure": [
                {
                    "title": unicodedata.normalize("NFC", "परिचय"),
                    "start_index": 1,
                    "end_index": 2,
                    "node_id": "0001",
                    "summary": unicodedata.normalize("NFC", "यो एक परिचय हो।"),
                },
            ],
        }

        path = os.path.join(tmp_output_dir, "nepali_structure.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False)

        tg = TreeGrep(path)
        results = tg.search("परिचय")
        assert len(results) > 0


# ============================================================================
# Test: Embedder data classes (Phase 6)
# ============================================================================


class TestEmbedder:
    """Test Embedder data classes (actual DB tests need docker)."""

    def test_embedding_result(self):
        from agent.embedder import EmbeddingResult

        er = EmbeddingResult(
            doc_name="doc.md",
            section_id="0001",
            section_title="Intro",
            content="Content text",
            similarity=0.95,
            source_file="doc.md",
        )
        assert er.similarity == 0.95

    def test_embedder_init(self, sample_config):
        from agent.embedder import Embedder

        emb = Embedder(sample_config)
        assert emb._pool is None  # not connected yet


# ============================================================================
# Test: Fixture integrity
# ============================================================================


class TestFixtures:
    """Verify test fixture files are valid."""

    def test_all_structures_valid(self, all_structures):
        assert len(all_structures) == 9
        for fname, structure in all_structures.items():
            assert "doc_name" in structure, f"{fname} missing doc_name"
            assert "structure" in structure, f"{fname} missing structure"
            assert len(structure["structure"]) > 0, f"{fname} has empty structure"

    def test_all_structures_have_node_ids(self, all_structures):
        def check_nodes(nodes, fname):
            for node in nodes:
                assert "node_id" in node, f"{fname}: node missing node_id: {node.get('title')}"
                if "nodes" in node:
                    check_nodes(node["nodes"], fname)

        for fname, structure in all_structures.items():
            check_nodes(structure["structure"], fname)

    def test_markdown_files_exist(self):
        md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
        assert len(md_files) >= 9, f"Expected >= 9 markdown files, found {len(md_files)}"

    def test_earthmover_has_expected_structure(self, earthmover_structure):
        assert earthmover_structure["doc_name"] == "earthmover.pdf"
        titles = [n["title"] for n in earthmover_structure["structure"]]
        assert any("Earth Mover" in t for t in titles)
        assert any("ABSTRACT" in t for t in titles)
