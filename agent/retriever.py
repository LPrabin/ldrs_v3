"""
Retriever — Stage 2: Parallel Retrieval.

Runs TreeGrep and pgvector similarity search in parallel across the
files selected by the Intent Classifier (Stage 1).

The retriever produces a unified section pool that feeds into the
BM25 Fusion Ranker (Stage 3).

Architecture::

    Intent Classifier output
        ├── TreeGrep (pattern_hints → search_from_hints)
        │     Returns: GrepResults with relevance scores
        └── pgvector (query_variants → search_multi)
              Returns: EmbeddingResults with similarity scores
        │
        └── Section Pool (deduplicated by doc_name + section_id)

Usage::

    config = AgentConfig()
    retriever = Retriever(config, embedder=embedder)
    pool = await retriever.retrieve(intent_result)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.config import AgentConfig
from agent.embedder import Embedder, EmbeddingResult
from agent.intent_classifier import IntentResult
from agent.monitoring import UsageTracker
from agent.tree_grep import GrepResult, TreeGrep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section Pool item — unified format for fusion ranking
# ---------------------------------------------------------------------------


@dataclass
class SectionCandidate:
    """
    A candidate section from retrieval, ready for fusion ranking.

    Combines signals from both grep and vector retrieval into a single
    object that the BM25 Fusion Ranker can score.
    """

    doc_name: str
    section_id: str
    section_title: str
    content: str
    source_file: str
    line_num: int = 0
    breadcrumb: str = ""

    # Retrieval signals
    grep_score: float = 0.0  # Best TreeGrep relevance_score for this section
    grep_hits: int = 0  # Number of grep hits in this section
    vector_similarity: float = 0.0  # Best cosine similarity from pgvector
    retrieval_methods: List[str] = field(default_factory=list)  # ["grep", "vector"]

    @property
    def retrieval_method_str(self) -> str:
        """Return combined retrieval method string."""
        if "grep" in self.retrieval_methods and "vector" in self.retrieval_methods:
            return "grep+vector"
        return "+".join(self.retrieval_methods) if self.retrieval_methods else "unknown"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Stage 2: Parallel Retrieval.

    Runs TreeGrep and pgvector similarity search in parallel, then
    merges results into a deduplicated section pool.

    Args:
        config:   AgentConfig instance.
        embedder: Pre-connected Embedder for vector search.
    """

    def __init__(
        self,
        config: AgentConfig,
        embedder: Optional[Embedder] = None,
    ):
        self.config = config
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.config)
        return self._embedder

    def _load_tree_greps(self, selected_paths: List[str]) -> List[TreeGrep]:
        """
        Load TreeGrep instances for the selected files.

        Looks for structure JSONs in results_dir matching the selected paths.

        Args:
            selected_paths: File paths from IntentResult.selected_paths.

        Returns:
            List of TreeGrep instances (skips files without structure JSONs).
        """
        greps = []
        results_dir = os.path.abspath(self.config.results_dir)

        for path in selected_paths:
            # Derive structure JSON path from the file path
            basename = os.path.splitext(os.path.basename(path))[0]
            index_path = os.path.join(results_dir, f"{basename}_structure.json")

            if os.path.exists(index_path):
                try:
                    grep = TreeGrep(index_path=index_path)
                    greps.append(grep)
                    logger.debug("Retriever  loaded TreeGrep  doc=%s", basename)
                except Exception as e:
                    logger.warning(
                        "Retriever  failed to load TreeGrep  path=%s  error=%s", index_path, e
                    )
            else:
                logger.debug("Retriever  no structure JSON for %s", path)

        return greps

    async def _grep_search(
        self,
        intent: IntentResult,
    ) -> List[GrepResult]:
        """
        Run TreeGrep search across selected files using pattern_hints and query_variants.

        Args:
            intent: The IntentResult from Stage 1.

        Returns:
            Merged list of GrepResult across all selected files.
        """
        greps = self._load_tree_greps(intent.selected_paths)
        if not greps:
            logger.info("Retriever._grep_search  no TreeGrep instances loaded")
            return []

        all_results: List[GrepResult] = []
        max_per_doc = self.config.max_grep_results

        for grep in greps:
            # Use pattern_hints if available
            hints = intent.pattern_hints.to_dict()
            has_hints = any(hints.get(k) for k in ("literals", "phrases", "prefix_wildcards"))

            if has_hints:
                results = grep.search_from_hints(hints, max_results=max_per_doc)
            else:
                # Fall back to query_variants
                results = grep.search_multi(
                    intent.query_variants,
                    max_results=max_per_doc,
                )
            logger.debug(
                "Retriever._grep_search  doc=%s  hits=%d  method=%s",
                grep.doc_name,
                len(results),
                "hints" if has_hints else "variants",
            )
            all_results.extend(results)

        logger.info(
            "Retriever._grep_search  docs=%d  total_hits=%d",
            len(greps),
            len(all_results),
        )
        return all_results

    async def _vector_search(
        self,
        intent: IntentResult,
    ) -> List[EmbeddingResult]:
        """
        Run pgvector similarity search using query_variants.

        Args:
            intent: The IntentResult from Stage 1.

        Returns:
            List of EmbeddingResult from pgvector.
        """
        doc_names = []
        for path in intent.selected_paths:
            basename = os.path.splitext(os.path.basename(path))[0]
            doc_names.append(basename)

        results = await self.embedder.search_multi(
            queries=intent.query_variants,
            top_k_per_query=5,
            doc_names=doc_names if doc_names else None,
        )

        logger.info(
            "Retriever._vector_search  queries=%d  results=%d",
            len(intent.query_variants),
            len(results),
        )
        return results

    def _merge_results(
        self,
        grep_results: List[GrepResult],
        vector_results: List[EmbeddingResult],
    ) -> List[SectionCandidate]:
        """
        Merge grep and vector results into a deduplicated section pool.

        Deduplication key: (doc_name, section_id).
        Signals from both sources are combined into a single SectionCandidate.

        Args:
            grep_results:   Results from TreeGrep.
            vector_results: Results from pgvector.

        Returns:
            List of SectionCandidate with combined signals.
        """
        pool: Dict[tuple, SectionCandidate] = {}

        # Process grep results
        for gr in grep_results:
            key = (gr.doc_name, gr.node_id)
            if key not in pool:
                pool[key] = SectionCandidate(
                    doc_name=gr.doc_name,
                    section_id=gr.node_id,
                    section_title=gr.title,
                    content=gr.content,
                    source_file="",
                    line_num=gr.line_num,
                    breadcrumb=gr.breadcrumb,
                )
            candidate = pool[key]
            candidate.grep_score = max(candidate.grep_score, gr.relevance_score)
            candidate.grep_hits += 1
            if "grep" not in candidate.retrieval_methods:
                candidate.retrieval_methods.append("grep")
            # Update content if we have a better one
            if gr.content and not candidate.content:
                candidate.content = gr.content

        # Process vector results
        for vr in vector_results:
            key = (vr.doc_name, vr.section_id)
            is_new = key not in pool
            if is_new:
                pool[key] = SectionCandidate(
                    doc_name=vr.doc_name,
                    section_id=vr.section_id,
                    section_title=vr.section_title,
                    content=vr.content,
                    source_file=vr.source_file,
                    line_num=vr.line_num,
                )
            candidate = pool[key]
            candidate.vector_similarity = max(candidate.vector_similarity, vr.similarity)
            if "vector" not in candidate.retrieval_methods:
                candidate.retrieval_methods.append("vector")
            # Update source_file from vector results (more reliable)
            if vr.source_file:
                candidate.source_file = vr.source_file
            if vr.content and not candidate.content:
                candidate.content = vr.content
            logger.debug(
                "Retriever._merge_results  vector  doc=%s  section=%s  sim=%.4f  new=%s",
                vr.doc_name,
                vr.section_id,
                vr.similarity,
                is_new,
            )

        candidates = list(pool.values())
        logger.info(
            "Retriever._merge_results  grep=%d  vector=%d  merged=%d",
            len(grep_results),
            len(vector_results),
            len(candidates),
        )
        return candidates

    async def retrieve(
        self,
        intent: IntentResult,
        tracker: Optional[UsageTracker] = None,
    ) -> List[SectionCandidate]:
        """
        Run parallel retrieval and produce a merged section pool.

        Args:
            intent:  IntentResult from Stage 1.
            tracker: Optional UsageTracker.

        Returns:
            List of SectionCandidate ready for fusion ranking.
        """
        if tracker:
            tracker.start_stage("retrieval")

        logger.info(
            "Retriever.retrieve  files=%d  variants=%d",
            len(intent.selected_files),
            len(intent.query_variants),
        )

        # Run grep and vector search in parallel
        grep_task = self._grep_search(intent)
        vector_task = self._vector_search(intent)
        grep_results, vector_results = await asyncio.gather(grep_task, vector_task)

        # Merge into section pool
        pool = self._merge_results(grep_results, vector_results)

        if tracker:
            tracker.end_stage("retrieval")

        logger.info("Retriever.retrieve  done  pool_size=%d", len(pool))
        return pool
