"""
Pipeline — End-to-end orchestrator for the 6-stage Hybrid RAG Deep Agent System.

Wires together all stages into a single ``query()`` method:

  Stage 1: Intent Classification  →  IntentResult
  Stage 2: Parallel Retrieval     →  List[SectionCandidate]
  Stage 3: BM25 Fusion Ranking    →  List[RankedSection]
  Stage 4: VFS Population         →  SessionInfo
  Stage 5: Agent Loop             →  AgentResult
  Stage 6: Grounding Verification →  GroundingResult

Also provides lifecycle management (``startup()`` / ``shutdown()``) and
an ``index_file()`` / ``index_directory()`` pass-through to the Indexer.

Usage::

    config = AgentConfig()
    pipeline = Pipeline(config)
    await pipeline.startup()

    result = await pipeline.query("How does OAuth2 token refresh work?")
    print(result.answer)
    print(result.usage)

    await pipeline.shutdown()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.agent_loop import AgentLoop, AgentResult
from agent.config import AgentConfig
from agent.embedder import Embedder
from agent.fusion_ranker import FusionRanker, RankedSection
from agent.grounding import GroundingResult, GroundingVerifier
from agent.indexer import IndexResult, Indexer
from agent.intent_classifier import IntentClassifier, IntentResult
from agent.monitoring import UsageTracker, setup_monitoring
from agent.registry import Registry
from agent.retriever import Retriever, SectionCandidate
from agent.vfs import VFS, SessionInfo

logger = logging.getLogger(__name__)

# Maximum re-grounding attempts before giving up
MAX_REGROUND_ATTEMPTS = 1


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete result returned by ``Pipeline.query()``."""

    # Final answer (after grounding verification)
    answer: str = ""

    # Stage outputs (for inspection / debugging)
    intent: Optional[IntentResult] = None
    candidates_count: int = 0
    ranked_count: int = 0
    session_id: str = ""
    agent_result: Optional[AgentResult] = None
    grounding_result: Optional[GroundingResult] = None

    # Monitoring
    usage: Dict[str, Any] = field(default_factory=dict)
    total_time_ms: float = 0.0

    # Error handling
    error: str = ""
    success: bool = True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    End-to-end orchestrator for the Hybrid RAG Deep Agent System.

    Owns all components and manages their lifecycle. Components are
    created lazily (on first ``query()`` or ``startup()``) and shared
    across queries.

    Args:
        config: AgentConfig instance. If ``None``, reads from environment.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # Lifecycle state
        self._started = False

        # Shared components (created in startup or lazily)
        self._embedder: Optional[Embedder] = None
        self._registry: Optional[Registry] = None
        self._indexer: Optional[Indexer] = None
        self._vfs: Optional[VFS] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._retriever: Optional[Retriever] = None
        self._fusion_ranker: Optional[FusionRanker] = None
        self._agent_loop: Optional[AgentLoop] = None
        self._grounding_verifier: Optional[GroundingVerifier] = None

        # Conversation state
        self._conversation_history: List[Dict[str, str]] = []
        self._conversation_summary: str = ""

    # ------------------------------------------------------------------
    # Lazy component properties
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.config)
            logger.debug("Pipeline  created Embedder")
        return self._embedder

    @property
    def registry(self) -> Registry:
        if self._registry is None:
            self._registry = Registry(self.config.registry_path)
            logger.debug("Pipeline  created Registry  path=%s", self.config.registry_path)
        return self._registry

    @property
    def indexer(self) -> Indexer:
        if self._indexer is None:
            self._indexer = Indexer(
                self.config,
                embedder=self.embedder,
                registry=self.registry,
            )
            logger.debug("Pipeline  created Indexer")
        return self._indexer

    @property
    def vfs(self) -> VFS:
        if self._vfs is None:
            self._vfs = VFS(self.config)
            logger.debug("Pipeline  created VFS  sessions_dir=%s", self.config.sessions_dir)
        return self._vfs

    @property
    def intent_classifier(self) -> IntentClassifier:
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier(self.config)
            logger.debug("Pipeline  created IntentClassifier  model=%s", self.config.default_model)
        return self._intent_classifier

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.config, embedder=self.embedder)
            logger.debug("Pipeline  created Retriever")
        return self._retriever

    @property
    def fusion_ranker(self) -> FusionRanker:
        if self._fusion_ranker is None:
            self._fusion_ranker = FusionRanker(self.config)
            logger.debug("Pipeline  created FusionRanker")
        return self._fusion_ranker

    @property
    def agent_loop(self) -> AgentLoop:
        if self._agent_loop is None:
            self._agent_loop = AgentLoop(self.config, vfs=self.vfs)
            logger.debug("Pipeline  created AgentLoop")
        return self._agent_loop

    @property
    def grounding_verifier(self) -> GroundingVerifier:
        if self._grounding_verifier is None:
            self._grounding_verifier = GroundingVerifier(self.config, vfs=self.vfs)
            logger.debug("Pipeline  created GroundingVerifier")
        return self._grounding_verifier

    # ------------------------------------------------------------------
    # Runtime config update
    # ------------------------------------------------------------------

    def update_llm_config(
        self,
        *,
        default_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update LLM/embedding settings and invalidate affected components.

        Only non-None arguments are applied.  Returns a dict describing
        what changed.

        Components that depend on chat kwargs (IntentClassifier, AgentLoop,
        GroundingVerifier) are invalidated when ``default_model``, ``api_key``,
        or ``api_base`` change.  Embedder is invalidated when
        ``embedding_model``, ``api_key``, or ``api_base`` change.
        """
        changes = self.config.update_llm_settings(
            default_model=default_model,
            embedding_model=embedding_model,
            api_key=api_key,
            api_base=api_base,
        )

        if not changes:
            logger.debug("Pipeline.update_llm_config  no changes")
            return {"changed": False, "fields": {}}

        changed_fields = set(changes.keys())

        # Determine which caches to invalidate
        chat_fields = {"default_model", "api_key", "api_base"}
        embed_fields = {"embedding_model", "api_key", "api_base"}

        if changed_fields & chat_fields:
            logger.info(
                "Pipeline.update_llm_config  invalidating chat components  fields=%s",
                changed_fields & chat_fields,
            )
            self._intent_classifier = None
            self._agent_loop = None
            self._grounding_verifier = None

        if changed_fields & embed_fields:
            logger.info(
                "Pipeline.update_llm_config  invalidating embedding components  fields=%s",
                changed_fields & embed_fields,
            )
            # Note: We don't close the old embedder's DB pool here because
            # it may still be referenced.  The new embedder will create its
            # own pool when connect() is called during the next query.
            self._embedder = None
            # Indexer and retriever depend on embedder, so invalidate them too
            self._indexer = None
            self._retriever = None

        return {"changed": True, "fields": changes}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Initialize all components and connect to external services.

        Call this once at application start (e.g. in a FastAPI lifespan).
        """
        if self._started:
            logger.debug("Pipeline.startup  already started")
            return

        logger.info("Pipeline.startup  initializing...")
        setup_monitoring(self.config)

        # Connect embedder to PostgreSQL
        await self.embedder.connect()

        # Force registry load
        _ = self.registry

        logger.info(
            "Pipeline.startup  done  registry_files=%d",
            len(self.registry.files),
        )
        self._started = True

    async def shutdown(self) -> None:
        """
        Clean up resources. Call on application exit.
        """
        logger.info("Pipeline.shutdown  cleaning up...")
        if self._embedder is not None:
            await self._embedder.close()
        self._started = False
        logger.info("Pipeline.shutdown  done")

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        *,
        conversation_summary: Optional[str] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        db_context: Optional[Dict[str, Any]] = None,
        cleanup_session: bool = False,
    ) -> PipelineResult:
        """
        Run the full 6-stage pipeline for a user query.

        Args:
            query:                  The user's question.
            conversation_summary:   Summary of prior conversation for context.
                                    If None, uses internal conversation state.
            recent_turns:           Recent conversation turns
                                    ``[{"role": "user", "content": "..."}, ...]``.
                                    If None, uses internal history.
            db_context:             Optional database query results for Stage 4.
            cleanup_session:        If True, delete VFS session after query.

        Returns:
            PipelineResult with the answer, stage outputs, and usage metrics.
        """
        if not self._started:
            await self.startup()

        tracker = UsageTracker()
        tracker.start_query()
        pipeline_start = time.time()

        # Use internal conversation state if not provided
        conv_summary = conversation_summary or self._conversation_summary
        turns = recent_turns or self._conversation_history[-10:]  # last 10 turns

        logger.info("Pipeline.query  query=%r  conv_turns=%d", query[:80], len(turns))

        try:
            # ---- Stage 1: Intent Classification ----
            intent = await self._stage_intent(query, conv_summary, tracker)
            logger.info(
                "Pipeline.query  stage1  intent=%s  files=%s  variants=%d",
                intent.intent_type,
                intent.selected_paths,
                len(intent.query_variants),
            )

            # ---- Stage 2: Parallel Retrieval ----
            candidates = await self._stage_retrieve(intent, tracker)
            logger.info("Pipeline.query  stage2  candidates=%d", len(candidates))

            # ---- Stage 3: BM25 Fusion Ranking ----
            ranked = self._stage_rank(candidates, intent, tracker)
            logger.info("Pipeline.query  stage3  ranked=%d", len(ranked))

            # ---- Stage 4: VFS Population ----
            session = self._stage_vfs(ranked, intent, conv_summary, turns, db_context, tracker)
            logger.info(
                "Pipeline.query  stage4  session=%s  sections=%d",
                session.session_id,
                session.section_count,
            )

            # ---- Stage 5: Agent Loop ----
            agent_result = await self._stage_agent(
                query, session.session_id, intent.intent_type, tracker
            )
            logger.info(
                "Pipeline.query  stage5  iterations=%d  tool_calls=%d  citations=%d",
                agent_result.iterations,
                agent_result.tool_calls_made,
                len(agent_result.citations),
            )

            # ---- Stage 6: Grounding Verification ----
            grounding_result = await self._stage_grounding(
                agent_result.answer, session.session_id, tracker
            )
            logger.info(
                "Pipeline.query  stage6  checked=%d  supported=%d  flagged=%d  re_grounded=%s",
                grounding_result.claims_checked,
                grounding_result.claims_supported,
                grounding_result.claims_flagged,
                grounding_result.re_grounded,
            )

            # ---- Re-grounding (if needed) ----
            if grounding_result.re_grounded:
                grounding_result = await self._handle_regrounding(
                    query,
                    session.session_id,
                    intent.intent_type,
                    grounding_result,
                    tracker,
                )

            # ---- Finalize ----
            tracker.end_query()

            # Update conversation history
            self._update_conversation(query, grounding_result.verified_answer)

            # Cleanup session if requested
            if cleanup_session:
                try:
                    self.vfs.cleanup_session(session.session_id)
                except Exception as e:
                    logger.warning("Pipeline.query  session cleanup failed: %s", e)

            total_time = (time.time() - pipeline_start) * 1000

            return PipelineResult(
                answer=grounding_result.verified_answer,
                intent=intent,
                candidates_count=len(candidates),
                ranked_count=len(ranked),
                session_id=session.session_id,
                agent_result=agent_result,
                grounding_result=grounding_result,
                usage=tracker.summary(),
                total_time_ms=total_time,
            )

        except Exception as e:
            logger.error("Pipeline.query  error=%s", e, exc_info=True)
            tracker.end_query()
            total_time = (time.time() - pipeline_start) * 1000
            return PipelineResult(
                answer="",
                usage=tracker.summary(),
                total_time_ms=total_time,
                error=str(e),
                success=False,
            )

    # ------------------------------------------------------------------
    # Individual stage methods
    # ------------------------------------------------------------------

    async def _stage_intent(
        self,
        query: str,
        conversation_summary: str,
        tracker: UsageTracker,
    ) -> IntentResult:
        """Stage 1: Intent Classification."""
        registry_for_llm = self.registry.get_for_llm()
        return await self.intent_classifier.classify(
            query=query,
            registry=registry_for_llm,
            conversation_summary=conversation_summary,
            tracker=tracker,
        )

    async def _stage_retrieve(
        self,
        intent: IntentResult,
        tracker: UsageTracker,
    ) -> List[SectionCandidate]:
        """Stage 2: Parallel Retrieval (TreeGrep + pgvector)."""
        # Ensure embedder is connected before retrieval
        # (it may have been lazily created after a config change)
        if self.embedder._pool is None:
            logger.info("Pipeline._stage_retrieve  connecting embedder")
            await self.embedder.connect()

        return await self.retriever.retrieve(
            intent=intent,
            tracker=tracker,
        )

    def _stage_rank(
        self,
        candidates: List[SectionCandidate],
        intent: IntentResult,
        tracker: UsageTracker,
    ) -> List[RankedSection]:
        """Stage 3: BM25 Fusion Ranking."""
        return self.fusion_ranker.rank(
            candidates=candidates,
            intent=intent,
            registry_files=self.registry.files,
            tracker=tracker,
        )

    def _stage_vfs(
        self,
        ranked: List[RankedSection],
        intent: IntentResult,
        conversation_summary: str,
        recent_turns: List[Dict[str, str]],
        db_context: Optional[Dict[str, Any]],
        tracker: UsageTracker,
    ) -> SessionInfo:
        """Stage 4: VFS Population."""
        return self.vfs.create_session(
            ranked_sections=ranked,
            intent=intent,
            conversation_summary=conversation_summary,
            recent_turns=recent_turns,
            db_context=db_context,
            registry_files=self.registry.files,
            tracker=tracker,
        )

    async def _stage_agent(
        self,
        query: str,
        session_id: str,
        intent_type: str,
        tracker: UsageTracker,
    ) -> AgentResult:
        """Stage 5: Agent Loop."""
        return await self.agent_loop.run(
            query=query,
            session_id=session_id,
            intent_type=intent_type,
            tracker=tracker,
        )

    async def _stage_grounding(
        self,
        answer: str,
        session_id: str,
        tracker: UsageTracker,
    ) -> GroundingResult:
        """Stage 6: Grounding Verification."""
        return await self.grounding_verifier.verify(
            answer=answer,
            session_id=session_id,
            tracker=tracker,
        )

    # ------------------------------------------------------------------
    # Re-grounding
    # ------------------------------------------------------------------

    async def _handle_regrounding(
        self,
        query: str,
        session_id: str,
        intent_type: str,
        grounding_result: GroundingResult,
        tracker: UsageTracker,
    ) -> GroundingResult:
        """
        Handle re-grounding when too many claims are flagged.

        Re-runs Stage 5 with explicit grounding feedback appended to
        the query, then re-verifies.
        """
        logger.warning(
            "Pipeline._handle_regrounding  flagged=%d/%d, re-running agent",
            grounding_result.claims_flagged,
            grounding_result.claims_checked,
        )

        # Build grounding feedback for the agent
        flag_details = []
        for flag in grounding_result.flags:
            flag_details.append(
                f'- Claim: "{flag.claim}"\n  Citation: {flag.citation}\n  Issue: {flag.reason}'
            )
        feedback = "\n".join(flag_details)

        regrounded_query = (
            f"{query}\n\n"
            "## IMPORTANT: Grounding Feedback\n"
            "Your previous answer had claims that could not be verified "
            "against their cited sources. Please revise your answer, "
            "ensuring every claim is directly supported by the section you cite.\n\n"
            "### Flagged Claims\n"
            f"{feedback}\n\n"
            "Please re-read the relevant sections and produce a corrected, "
            "well-grounded answer."
        )

        # Re-run agent with grounding feedback
        agent_result = await self._stage_agent(
            regrounded_query,
            session_id,
            intent_type,
            tracker,
        )

        # Re-verify
        new_grounding = await self._stage_grounding(
            agent_result.answer,
            session_id,
            tracker,
        )

        logger.info(
            "Pipeline._handle_regrounding  done  new_flagged=%d/%d",
            new_grounding.claims_flagged,
            new_grounding.claims_checked,
        )

        return new_grounding

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _update_conversation(self, query: str, answer: str) -> None:
        """Update internal conversation history."""
        self._conversation_history.append({"role": "user", "content": query})
        self._conversation_history.append({"role": "assistant", "content": answer})

        # Keep last 20 turns (10 exchanges)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # Update summary (simple truncation — could be LLM-generated)
        recent = self._conversation_history[-6:]  # last 3 exchanges
        parts = []
        for turn in recent:
            role = turn["role"]
            content = turn["content"][:200]
            parts.append(f"{role}: {content}")
        self._conversation_summary = "\n".join(parts)

        logger.debug(
            "Pipeline._update_conversation  history_len=%d  summary_len=%d",
            len(self._conversation_history),
            len(self._conversation_summary),
        )

    def clear_conversation(self) -> None:
        """Clear conversation history and summary."""
        self._conversation_history = []
        self._conversation_summary = ""
        logger.info("Pipeline  conversation cleared")

    # ------------------------------------------------------------------
    # Indexing pass-through
    # ------------------------------------------------------------------

    async def index_file(self, md_path: str, **kwargs) -> IndexResult:
        """
        Index a single markdown file through the full pipeline.

        Pass-through to ``Indexer.index_file()``.

        Args:
            md_path: Path to the .md file.
            **kwargs: Additional arguments for ``Indexer.index_file()``.

        Returns:
            IndexResult with indexing outcome.
        """
        if not self._started:
            await self.startup()

        # Ensure embedder is connected (may have been re-created after config change)
        if self.embedder._pool is None:
            logger.info("Pipeline.index_file  connecting embedder after config change")
            await self.embedder.connect()

        return await self.indexer.index_file(md_path, **kwargs)

    async def index_directory(self, directory: Optional[str] = None, **kwargs) -> List[IndexResult]:
        """
        Index all markdown files in a directory.

        Pass-through to ``Indexer.index_directory()``.

        Args:
            directory: Directory path. Defaults to ``config.docs_dir``.
            **kwargs:  Additional arguments for ``Indexer.index_file()``.

        Returns:
            List of IndexResult for each file.
        """
        if not self._started:
            await self.startup()

        # Ensure embedder is connected (may have been re-created after config change)
        if self.embedder._pool is None:
            logger.info("Pipeline.index_directory  connecting embedder after config change")
            await self.embedder.connect()

        return await self.indexer.index_directory(directory, **kwargs)

    # ------------------------------------------------------------------
    # Corpus info
    # ------------------------------------------------------------------

    def get_corpus_summary(self) -> Dict[str, Any]:
        """Return a summary of the current corpus from the registry."""
        return self.registry.get_corpus_summary()

    def get_corpus_files(self) -> Dict[str, Any]:
        """Return the full file registry for inspection."""
        return self.registry.files

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[str]:
        """List all VFS session IDs."""
        return self.vfs.list_sessions()

    def cleanup_session(self, session_id: str) -> None:
        """Delete a VFS session directory."""
        self.vfs.cleanup_session(session_id)

    def cleanup_all_sessions(self) -> None:
        """Delete all VFS sessions."""
        for sid in self.vfs.list_sessions():
            try:
                self.vfs.cleanup_session(sid)
            except Exception as e:
                logger.warning("Pipeline.cleanup_all_sessions  %s: %s", sid, e)
