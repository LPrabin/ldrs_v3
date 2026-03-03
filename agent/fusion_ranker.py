"""
Fusion Ranker — Stage 3: BM25 Fusion Ranking.

Takes the section pool from Stage 2 (Parallel Retrieval) and produces a
final ranked list using a weighted fusion of three signals:

  score = (w_bm25 × BM25) + (w_vector × vector_similarity) + (w_grep × grep_density)

Weights shift based on intent_type:
  - exact:       grep_density weight up (0.35)
  - conceptual:  vector weight up (0.55)
  - comparative: balanced, enforces multi-file coverage
  - multihop:    recency boost up, balanced retrieval

Metadata boosts:
  - recency_factor: newer docs score higher
  - tag_overlap:    query terms matching registry tags

Usage::

    config = AgentConfig()
    ranker = FusionRanker(config)
    ranked = ranker.rank(section_pool, intent_result, registry)
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from agent.config import AgentConfig
from agent.intent_classifier import IntentResult
from agent.monitoring import UsageTracker
from agent.retriever import SectionCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight presets by intent type
# ---------------------------------------------------------------------------

WEIGHT_PRESETS = {
    "exact": {"bm25": 0.30, "vector": 0.35, "grep": 0.35},
    "conceptual": {"bm25": 0.25, "vector": 0.55, "grep": 0.20},
    "comparative": {"bm25": 0.35, "vector": 0.40, "grep": 0.25},
    "multihop": {"bm25": 0.35, "vector": 0.40, "grep": 0.25},
    "db_query": {"bm25": 0.30, "vector": 0.40, "grep": 0.30},
    "hybrid": {"bm25": 0.35, "vector": 0.40, "grep": 0.25},
}

DEFAULT_WEIGHTS = {"bm25": 0.40, "vector": 0.40, "grep": 0.20}


# ---------------------------------------------------------------------------
# Ranked result
# ---------------------------------------------------------------------------


@dataclass
class RankedSection:
    """
    A section with its final fusion score and breakdown.

    Ready for VFS population (Stage 4).
    """

    doc_name: str
    section_id: str
    section_title: str
    content: str
    source_file: str
    line_num: int = 0
    breadcrumb: str = ""

    # Final score
    final_score: float = 0.0

    # Score breakdown
    bm25_score: float = 0.0
    vector_score: float = 0.0
    grep_density: float = 0.0

    # Metadata
    retrieval_method: str = ""
    recency_factor: float = 1.0
    tag_boost: float = 1.0

    # Why included (human-readable)
    why_included: str = ""


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 indexing. Simple whitespace + lowercasing."""
    return re.findall(r"\w+", text.lower())


def _compute_grep_density(candidate: SectionCandidate) -> float:
    """
    Compute grep density: hits_in_section / section_line_count.

    Density (not raw count) avoids rewarding long sections unfairly.
    """
    if not candidate.content:
        return 0.0
    line_count = max(1, candidate.content.count("\n") + 1)
    return candidate.grep_hits / line_count


def _compute_recency_factor(doc_name: str, registry_files: Dict[str, Any]) -> float:
    """
    Compute a recency multiplier based on last_modified from registry.

    Returns a value in [0.5, 1.0] where 1.0 = very recent.
    Docs without last_modified get 0.75 (neutral).
    """
    for path, entry in registry_files.items():
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if basename == doc_name or path == doc_name:
            last_mod = entry.get("last_modified", "")
            if last_mod:
                try:
                    mod_date = datetime.strptime(last_mod, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    days_old = (now - mod_date).days
                    # Exponential decay: half-life = 365 days
                    factor = 0.5 + 0.5 * math.exp(-days_old / 365.0)
                    return round(factor, 3)
                except (ValueError, TypeError):
                    pass
    return 0.75  # neutral default


def _compute_tag_boost(
    doc_name: str,
    query_tokens: List[str],
    registry_files: Dict[str, Any],
) -> float:
    """
    Compute tag overlap boost: fraction of query tokens that match registry tags.

    Returns a multiplier in [1.0, 1.5].
    """
    for path, entry in registry_files.items():
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if basename == doc_name or path == doc_name:
            tags = [t.lower() for t in entry.get("tags", [])]
            if not tags or not query_tokens:
                return 1.0
            matches = sum(1 for t in query_tokens if t in tags)
            overlap = matches / len(query_tokens)
            return round(1.0 + 0.5 * overlap, 3)
    return 1.0


# ---------------------------------------------------------------------------
# FusionRanker
# ---------------------------------------------------------------------------


class FusionRanker:
    """
    Stage 3: BM25 Fusion Ranking.

    Combines BM25, vector similarity, and grep density into a final score.
    Weights shift based on intent_type. Metadata boosts (recency, tag overlap)
    act as multipliers.

    Args:
        config: AgentConfig with weight defaults and max_vfs_sections.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def rank(
        self,
        candidates: List[SectionCandidate],
        intent: IntentResult,
        registry_files: Optional[Dict[str, Any]] = None,
        tracker: Optional[UsageTracker] = None,
    ) -> List[RankedSection]:
        """
        Rank the section pool and return top N sections for VFS.

        Args:
            candidates:     Section pool from Stage 2.
            intent:         IntentResult from Stage 1.
            registry_files: Registry files dict for metadata boosts.
            tracker:        Optional UsageTracker.

        Returns:
            Sorted list of RankedSection (highest score first),
            capped at config.max_vfs_sections.
        """
        if tracker:
            tracker.start_stage("fusion_ranking")

        if not candidates:
            logger.info("FusionRanker.rank  empty pool")
            if tracker:
                tracker.end_stage("fusion_ranking")
            return []

        registry_files = registry_files or {}

        # Select weights based on intent
        weights = WEIGHT_PRESETS.get(intent.intent_type, DEFAULT_WEIGHTS)
        logger.info(
            "FusionRanker.rank  intent=%s  weights=%s  candidates=%d",
            intent.intent_type,
            weights,
            len(candidates),
        )

        # --- BM25 scoring ---
        # Build corpus from candidate contents
        corpus = [_tokenize_for_bm25(c.content) for c in candidates]

        # Handle empty corpus
        non_empty = [tokens for tokens in corpus if tokens]
        if not non_empty:
            logger.warning("FusionRanker.rank  all candidates have empty content")
            if tracker:
                tracker.end_stage("fusion_ranking")
            return []

        bm25 = BM25Okapi(corpus)

        # Score each query variant and take the max per document
        query_tokens_list = [_tokenize_for_bm25(qv) for qv in intent.query_variants]
        bm25_scores = [0.0] * len(candidates)

        for qt in query_tokens_list:
            if qt:
                scores = bm25.get_scores(qt)
                for i, s in enumerate(scores):
                    bm25_scores[i] = max(bm25_scores[i], float(s))

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_scores = [s / max_bm25 for s in bm25_scores]
        logger.debug(
            "FusionRanker.rank  BM25 normalized  max_raw=%.4f  variants=%d",
            max_bm25,
            len(query_tokens_list),
        )

        # --- Build ranked sections ---
        # Query tokens for tag boost
        all_query_tokens = set()
        for qt in query_tokens_list:
            all_query_tokens.update(qt)

        ranked: List[RankedSection] = []
        for i, candidate in enumerate(candidates):
            # Signals
            bm25_score = bm25_scores[i]
            vector_score = candidate.vector_similarity
            grep_density = _compute_grep_density(candidate)

            # Metadata boosts
            recency = _compute_recency_factor(candidate.doc_name, registry_files)
            tag_boost = _compute_tag_boost(
                candidate.doc_name, list(all_query_tokens), registry_files
            )

            # Fusion score
            raw_score = (
                weights["bm25"] * bm25_score
                + weights["vector"] * vector_score
                + weights["grep"] * grep_density
            )
            final_score = raw_score * recency * tag_boost

            # Why included
            why_parts = []
            if grep_density > 0:
                why_parts.append(f"grep_density={grep_density:.2f}")
            if vector_score > 0:
                why_parts.append(f"vector={vector_score:.2f}")
            if bm25_score > 0:
                why_parts.append(f"bm25={bm25_score:.2f}")
            if recency != 1.0:
                why_parts.append(f"recency={recency:.2f}")
            if tag_boost != 1.0:
                why_parts.append(f"tag_boost={tag_boost:.2f}")

            ranked.append(
                RankedSection(
                    doc_name=candidate.doc_name,
                    section_id=candidate.section_id,
                    section_title=candidate.section_title,
                    content=candidate.content,
                    source_file=candidate.source_file,
                    line_num=candidate.line_num,
                    breadcrumb=candidate.breadcrumb,
                    final_score=round(final_score, 4),
                    bm25_score=round(bm25_score, 4),
                    vector_score=round(vector_score, 4),
                    grep_density=round(grep_density, 4),
                    retrieval_method=candidate.retrieval_method_str,
                    recency_factor=recency,
                    tag_boost=tag_boost,
                    why_included="; ".join(why_parts),
                )
            )
            logger.debug(
                "FusionRanker.rank  candidate  doc=%s  section=%s  "
                "bm25=%.4f  vector=%.4f  grep=%.4f  recency=%.3f  tag=%.3f  final=%.4f",
                candidate.doc_name,
                candidate.section_id,
                bm25_score,
                vector_score,
                grep_density,
                recency,
                tag_boost,
                final_score,
            )

        # Sort by final_score descending
        ranked.sort(key=lambda r: -r.final_score)

        # For comparative intent, enforce multi-file coverage
        if intent.intent_type == "comparative":
            ranked = self._enforce_multi_file(ranked)

        # Cap at max_vfs_sections
        top_n = self.config.max_vfs_sections
        if len(ranked) > top_n:
            ranked = ranked[:top_n]

        if tracker:
            tracker.end_stage("fusion_ranking")

        logger.info(
            "FusionRanker.rank  done  ranked=%d  top_score=%.4f",
            len(ranked),
            ranked[0].final_score if ranked else 0.0,
        )
        return ranked

    def _enforce_multi_file(self, ranked: List[RankedSection]) -> List[RankedSection]:
        """
        For comparative queries, ensure balanced coverage across files.

        Re-interleaves results so that no single file dominates the top positions.
        """
        if not ranked:
            return ranked

        # Group by doc_name
        by_doc: Dict[str, List[RankedSection]] = {}
        for r in ranked:
            by_doc.setdefault(r.doc_name, []).append(r)

        if len(by_doc) <= 1:
            return ranked  # Only one file, nothing to balance

        # Round-robin interleave from each file
        interleaved: List[RankedSection] = []
        iterators = {doc: iter(sections) for doc, sections in by_doc.items()}
        while iterators:
            exhausted = []
            for doc, it in iterators.items():
                try:
                    interleaved.append(next(it))
                except StopIteration:
                    exhausted.append(doc)
            for doc in exhausted:
                del iterators[doc]

        logger.debug(
            "FusionRanker._enforce_multi_file  files=%d  interleaved=%d",
            len(by_doc),
            len(interleaved),
        )
        return interleaved
