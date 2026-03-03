"""
LDRS v3 — Hybrid RAG Deep Agent System.

A 6-stage retrieval-augmented generation pipeline:

  Stage 0: File Watcher     — monitors .md dirs, auto-indexes on change
  Stage 1: Intent Classifier — single LLM call for intent + routing
  Stage 2: Parallel Retrieval — TreeGrep + pgvector similarity
  Stage 3: BM25 Fusion Ranking — hybrid scoring with intent-based weights
  Stage 4: VFS Population   — session filesystem with manifest
  Stage 5: Agent Loop       — OpenAI function calling with custom tools
  Stage 6: Grounding        — semantic entailment verification

Modules:
  - config:             Centralised configuration (AgentConfig)
  - monitoring:         LangSmith integration + usage tracking
  - registry:           Document registry (AGENT_SYSTEM schema)
  - embedder:           Section embedding → pgvector
  - indexer:            MD → PageIndex → embed → register
  - watcher:            File system watcher (watchdog)
  - intent_classifier:  Intent classification + routing
  - retriever:          Parallel retrieval (TreeGrep + vector)
  - tree_grep:          Hierarchical word-level pattern search
  - fusion_ranker:      BM25 fusion ranking
  - vfs:                Virtual filesystem population
  - tools:              Agent tools (read_section, fetch_section, etc.)
  - agent_loop:         Deep agent loop (OpenAI function calling)
  - grounding:          Grounding verification
  - pipeline:           End-to-end orchestration

All imports are lazy to avoid import errors when optional dependencies
(asyncpg, openai, etc.) are not installed. Use direct module imports
when you need a specific class::

    from agent.pipeline import Pipeline
    from agent.config import AgentConfig
"""

import logging

_logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Lazy imports to avoid pulling in heavy dependencies on package load."""
    _lazy_map = {
        "AgentConfig": "agent.config",
        "setup_monitoring": "agent.monitoring",
        "UsageTracker": "agent.monitoring",
        "Registry": "agent.registry",
        "Embedder": "agent.embedder",
        "Indexer": "agent.indexer",
        "TreeGrep": "agent.tree_grep",
        "IntentClassifier": "agent.intent_classifier",
        "Retriever": "agent.retriever",
        "FusionRanker": "agent.fusion_ranker",
        "VFS": "agent.vfs",
        "AgentTools": "agent.tools",
        "AgentLoop": "agent.agent_loop",
        "GroundingVerifier": "agent.grounding",
        "Pipeline": "agent.pipeline",
        "PipelineResult": "agent.pipeline",
    }
    if name in _lazy_map:
        import importlib

        module_path = _lazy_map[name]
        _logger.debug("agent.__getattr__  lazy import  %s from %s", name, module_path)
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")


__all__ = [
    "AgentConfig",
    "setup_monitoring",
    "UsageTracker",
    "Registry",
    "Embedder",
    "Indexer",
    "TreeGrep",
    "IntentClassifier",
    "Retriever",
    "FusionRanker",
    "VFS",
    "AgentTools",
    "AgentLoop",
    "GroundingVerifier",
    "Pipeline",
    "PipelineResult",
]
