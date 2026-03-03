"""
TreeGrep — Hierarchical pattern search across structure JSON and Markdown.

Adapted from ldrs_v2's TreeGrep for the v3 architecture. Searches three
fields per node: title (3.0), summary (2.0), and body text (1.0).

Key features:
  - NFC normalization for Nepali/Devanagari text safety
  - Word-level tokenized matching with stop word filtering
  - Two-tier matching: exact substring (full score) then word-level (scaled)
  - Configurable min match ratio (default 0.3)
  - Scope filtering by node_id or title
  - Snippet extraction with configurable padding

In ldrs_v3, TreeGrep is used in Stage 2 (Parallel Retrieval) alongside
pgvector similarity search. The Intent Classifier's ``pattern_hints``
are translated into TreeGrep search patterns.

Usage::

    grep = TreeGrep(index_path="results/auth_structure.json", md_path="docs/auth.md")
    results = grep.search("OAuth2 token refresh", max_results=20)
    for r in results:
        print(f"{r.breadcrumb} [{r.matched_field}] score={r.relevance_score}")

    # Using pattern_hints from Intent Classifier:
    results = grep.search_from_hints(pattern_hints)
"""

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEVANCE_TITLE = 3.0
RELEVANCE_SUMMARY = 2.0
RELEVANCE_BODY = 1.0

DEFAULT_SNIPPET_PADDING = 60
DEFAULT_MAX_RESULTS = 50

# Minimum fraction of content words that must match for a word-level hit.
MIN_WORD_MATCH_RATIO = 0.3

# Stop words — stripped from sub-queries before word-level matching.
STOP_WORDS = frozenset(
    {
        # Articles & determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # Pronouns
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "his",
        "her",
        "its",
        "they",
        "them",
        "their",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "along",
        "until",
        "upon",
        "across",
        # Conjunctions
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        # Auxiliary / modal verbs
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        # Question words
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        # Other common function words
        "not",
        "no",
        "yes",
        "if",
        "then",
        "else",
        "than",
        "also",
        "as",
        "just",
        "only",
        "very",
        "too",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
        "much",
        "many",
        "such",
        "own",
        # Common verbs that rarely add search specificity
        "get",
        "got",
        "make",
        "made",
        "take",
        "taken",
        "give",
        "given",
        "go",
        "gone",
        "come",
        "came",
        "say",
        "said",
        "tell",
        "told",
        "know",
        "known",
        "see",
        "seen",
        "find",
        "found",
        "use",
        "used",
        # Misc
        "there",
        "here",
        "other",
        "like",
        "well",
        "back",
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GrepResult:
    """
    A single search hit inside the document tree.

    Attributes:
        doc_name:        The document name (from the structure JSON).
        node_id:         The node_id where the match was found.
        title:           The node's title.
        breadcrumb:      Ancestor path, e.g. ``"Chapter 1 > Section 2"``.
        matched_field:   Which field matched: ``"title"``, ``"summary"``, or ``"body"``.
        snippet:         Text excerpt around the match.
        relevance_score: Numeric relevance (title=3, summary=2, body=1).
        line_num:        Line number in the source .md file.
        content:         Full text content of the matched section (for fusion ranking).
    """

    doc_name: str
    node_id: str
    title: str
    breadcrumb: str
    matched_field: str
    snippet: str
    relevance_score: float = 1.0
    line_num: int = 0
    content: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nfc(text: str) -> str:
    """Normalize a string to NFC form."""
    return unicodedata.normalize("NFC", text)


def _build_snippet(text: str, start: int, end: int, padding: int = DEFAULT_SNIPPET_PADDING) -> str:
    """Extract a text snippet centred on text[start:end] with padding."""
    left = max(0, start - padding)
    right = min(len(text), end + padding)
    snippet = text[left:right].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


def _tokenize_query(pattern: str) -> List[str]:
    """
    Tokenize a search pattern into individual content words.

    Splits on non-alphanumeric boundaries, lowercases, removes stop words,
    and discards tokens shorter than 2 characters.

    Args:
        pattern: The raw search pattern (NFC-normalized).

    Returns:
        List of lowercase content words (stop words removed).
    """
    raw_tokens = re.split(r"[^\w]+", pattern.lower(), flags=re.UNICODE)
    tokens = [t for t in raw_tokens if t and len(t) >= 2 and t not in STOP_WORDS]
    return tokens


def _find_scope_nodes(nodes: List[dict], scope: Optional[str]) -> List[dict]:
    """
    Recursively locate subtree root(s) matching *scope*.

    A node matches if its ``node_id`` equals *scope* exactly or its
    ``title`` matches case-insensitively (NFC-normalized).
    """
    if scope is None:
        return nodes

    scope_nfc = _nfc(scope).lower()
    matches: List[dict] = []

    for node in nodes:
        nid = node.get("node_id", "")
        title = _nfc(node.get("title", "")).lower()
        if nid == scope or title == scope_nfc:
            matches.append(node)
            logger.debug(
                "_find_scope_nodes  matched  node_id=%s  title=%r  by=%s",
                nid,
                node.get("title", "")[:40],
                "node_id" if nid == scope else "title",
            )
        children = node.get("nodes") or []
        if children:
            matches.extend(_find_scope_nodes(children, scope))

    return matches


# ---------------------------------------------------------------------------
# TreeGrep class
# ---------------------------------------------------------------------------


class TreeGrep:
    """
    Hierarchical pattern search across a PageIndex structure tree.

    In ldrs_v3, the structure JSON stores the full text of each node
    (``if_add_node_text='yes'``), so body search uses the tree's ``text``
    field directly — no separate .md parsing needed.

    The search checks three fields per node:
      1. **title** (relevance = 3.0)
      2. **summary** (relevance = 2.0)
      3. **text/body** (relevance = 1.0)

    Results are sorted by relevance (highest first), then by node_id.

    Args:
        index_path: Path to the ``*_structure.json`` file.
        md_path:    Optional path to companion .md (for backward compat).
                    In v3, body text comes from the structure tree itself.
    """

    def __init__(
        self,
        index_path: str,
        md_path: Optional[str] = None,
    ):
        logger.info("TreeGrep init  index_path=%s", index_path)

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.structure: List[dict] = self.index.get("structure", [])
        self.doc_name: str = self.index.get("doc_name", os.path.basename(index_path))
        self.index_path = index_path

        logger.debug(
            "TreeGrep  doc=%s  top_nodes=%d",
            self.doc_name,
            len(self.structure),
        )

    # ------------------------------------------------------------------
    # Internal: recursive node search
    # ------------------------------------------------------------------

    def _search_node(
        self,
        node: dict,
        pattern_nfc: str,
        compiled_re: Optional[re.Pattern],
        content_tokens: List[str],
        word_regexes: List[re.Pattern],
        breadcrumb: List[str],
        results: List[GrepResult],
        use_regex: bool,
    ) -> None:
        """Recursively search a single node and its children."""
        title = _nfc(node.get("title", ""))
        summary = _nfc(node.get("summary", "") or node.get("prefix_summary", ""))
        body = _nfc(node.get("text", ""))
        node_id = node.get("node_id", "")
        line_num = node.get("line_num", 0)
        crumb = " > ".join(breadcrumb + [title]) if title else " > ".join(breadcrumb)

        def _check(field_name: str, text: str, base_relevance: float) -> None:
            """Check one field against the pattern and append a GrepResult."""
            if not text:
                return

            # Regex mode
            if use_regex and compiled_re:
                match = compiled_re.search(text)
                if match:
                    snippet = _build_snippet(text, match.start(), match.end())
                    results.append(
                        GrepResult(
                            doc_name=self.doc_name,
                            node_id=node_id,
                            title=title,
                            breadcrumb=crumb,
                            matched_field=field_name,
                            snippet=snippet,
                            relevance_score=base_relevance,
                            line_num=line_num,
                            content=body,
                        )
                    )
                return

            # Plain-text mode
            text_lower = text.lower()
            target_lower = pattern_nfc.lower()

            # Tier 1: exact substring match
            idx = text_lower.find(target_lower)
            if idx != -1:
                snippet = _build_snippet(text, idx, idx + len(target_lower))
                results.append(
                    GrepResult(
                        doc_name=self.doc_name,
                        node_id=node_id,
                        title=title,
                        breadcrumb=crumb,
                        matched_field=field_name,
                        snippet=snippet,
                        relevance_score=base_relevance,
                        line_num=line_num,
                        content=body,
                    )
                )
                return

            # Tier 2: word-level match
            if not content_tokens:
                return

            matched_count = 0
            first_match_start = len(text)
            first_match_end = 0

            for word_re in word_regexes:
                m = word_re.search(text)
                if m:
                    matched_count += 1
                    if m.start() < first_match_start:
                        first_match_start = m.start()
                        first_match_end = m.end()

            if matched_count == 0:
                return

            match_ratio = matched_count / len(content_tokens)
            if match_ratio < MIN_WORD_MATCH_RATIO:
                return

            scaled_relevance = base_relevance * match_ratio
            snippet = _build_snippet(text, first_match_start, first_match_end)
            results.append(
                GrepResult(
                    doc_name=self.doc_name,
                    node_id=node_id,
                    title=title,
                    breadcrumb=crumb,
                    matched_field=field_name,
                    snippet=snippet,
                    relevance_score=round(scaled_relevance, 3),
                    line_num=line_num,
                    content=body,
                )
            )

        # Check title, summary, and body
        pre_count = len(results)
        _check("title", title, RELEVANCE_TITLE)
        _check("summary", summary, RELEVANCE_SUMMARY)
        _check("body", body, RELEVANCE_BODY)
        hits_added = len(results) - pre_count
        if hits_added > 0:
            logger.debug(
                "TreeGrep._search_node  node_id=%s  title=%r  hits=%d",
                node_id,
                title[:40],
                hits_added,
            )

        # Recurse into children
        for child in node.get("nodes") or []:
            self._search_node(
                child,
                pattern_nfc,
                compiled_re,
                content_tokens,
                word_regexes,
                breadcrumb + [title] if title else breadcrumb,
                results,
                use_regex,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: str,
        scope: Optional[str] = None,
        regex: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[GrepResult]:
        """
        Search for *pattern* across the document tree.

        Uses two matching tiers in plain-text mode:
          1. Exact substring (full score)
          2. Word-level with score scaling by match ratio

        Args:
            pattern:     Search string or regex.
            scope:       Optional node_id or title to restrict search.
            regex:       Interpret pattern as regex.
            max_results: Maximum results to return.

        Returns:
            List of GrepResult sorted by relevance then node_id.
        """
        logger.info(
            "TreeGrep.search  pattern=%r  scope=%s  regex=%s  max=%d",
            pattern,
            scope,
            regex,
            max_results,
        )

        if not pattern:
            return []

        pattern_nfc = _nfc(pattern)

        compiled_re: Optional[re.Pattern] = None
        if regex:
            try:
                compiled_re = re.compile(
                    pattern_nfc,
                    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
                )
            except re.error as e:
                logger.error("TreeGrep.search  invalid regex %r: %s", pattern, e)
                return []

        content_tokens: List[str] = []
        word_regexes: List[re.Pattern] = []
        if not regex:
            content_tokens = _tokenize_query(pattern_nfc)
            word_regexes = [
                re.compile(re.escape(tok), re.IGNORECASE | re.DOTALL) for tok in content_tokens
            ]
            logger.debug("TreeGrep.search  content_tokens=%r", content_tokens)

        scope_roots = _find_scope_nodes(self.structure, scope)

        results: List[GrepResult] = []
        for root in scope_roots:
            self._search_node(
                root,
                pattern_nfc,
                compiled_re,
                content_tokens,
                word_regexes,
                [],
                results,
                regex,
            )

        results.sort(key=lambda r: (-r.relevance_score, r.node_id))

        if len(results) > max_results:
            results = results[:max_results]

        logger.info(
            "TreeGrep.search  done  hits=%d  pattern=%r",
            len(results),
            pattern[:50],
        )
        return results

    def search_multi(
        self,
        patterns: List[str],
        scope: Optional[str] = None,
        regex: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[GrepResult]:
        """
        Search for multiple patterns and merge results.

        Deduplicates on (node_id, matched_field), keeping highest score.

        Args:
            patterns:    List of search strings.
            scope:       Optional scope filter.
            regex:       Regex mode flag.
            max_results: Maximum total results.

        Returns:
            Merged, deduplicated, sorted list of GrepResult.
        """
        logger.info(
            "TreeGrep.search_multi  patterns=%d  scope=%s  max=%d",
            len(patterns),
            scope,
            max_results,
        )

        per_pattern_cap = (
            max(max_results, max_results * 2 // len(patterns)) if patterns else max_results
        )
        all_hits: List[GrepResult] = []
        for p in patterns:
            all_hits.extend(self.search(p, scope=scope, regex=regex, max_results=per_pattern_cap))

        # Deduplicate: keep highest-scoring hit per (node_id, matched_field)
        best: Dict[Tuple[str, str], GrepResult] = {}
        for hit in all_hits:
            key = (hit.node_id, hit.matched_field)
            if key not in best or hit.relevance_score > best[key].relevance_score:
                best[key] = hit
        deduped = list(best.values())

        deduped.sort(key=lambda r: (-r.relevance_score, r.node_id))
        if len(deduped) > max_results:
            deduped = deduped[:max_results]

        logger.info(
            "TreeGrep.search_multi  done  raw=%d  deduped=%d",
            len(all_hits),
            len(deduped),
        )
        return deduped

    def search_from_hints(
        self,
        pattern_hints: Dict[str, Any],
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[GrepResult]:
        """
        Search using pattern_hints from the Intent Classifier.

        The Intent Classifier outputs::

            {
                "literals": ["OAuth2", "401", "refresh_token"],
                "phrases": ["token refresh flow"],
                "prefix_wildcards": ["auth*", "token*"]
            }

        This method translates these into TreeGrep searches:
          - ``literals``: exact substring search for each
          - ``phrases``: exact substring search for each
          - ``prefix_wildcards``: regex search with word-boundary prefix

        Args:
            pattern_hints: Dict with ``literals``, ``phrases``, ``prefix_wildcards``.
            max_results:   Maximum total results.

        Returns:
            Merged, deduplicated results.
        """
        all_patterns: List[str] = []
        regex_patterns: List[str] = []

        # Literals — search as exact strings
        for lit in pattern_hints.get("literals", []):
            if lit:
                all_patterns.append(lit)

        # Phrases — search as exact strings
        for phrase in pattern_hints.get("phrases", []):
            if phrase:
                all_patterns.append(phrase)

        # Prefix wildcards — convert to regex
        for prefix in pattern_hints.get("prefix_wildcards", []):
            if prefix:
                # Remove trailing * and build regex
                clean = prefix.rstrip("*")
                if clean:
                    regex_patterns.append(re.escape(clean) + r"\w*")

        # Run plain-text searches
        results = self.search_multi(all_patterns, max_results=max_results) if all_patterns else []

        # Run regex searches and merge
        if regex_patterns:
            for rp in regex_patterns:
                regex_hits = self.search(rp, regex=True, max_results=max_results)
                results.extend(regex_hits)

        # Deduplicate again after merging regex results
        best: Dict[Tuple[str, str], GrepResult] = {}
        for hit in results:
            key = (hit.node_id, hit.matched_field)
            if key not in best or hit.relevance_score > best[key].relevance_score:
                best[key] = hit
        deduped = list(best.values())

        deduped.sort(key=lambda r: (-r.relevance_score, r.node_id))
        if len(deduped) > max_results:
            deduped = deduped[:max_results]

        logger.info(
            "TreeGrep.search_from_hints  literals=%d  phrases=%d  wildcards=%d  results=%d",
            len(pattern_hints.get("literals", [])),
            len(pattern_hints.get("phrases", [])),
            len(pattern_hints.get("prefix_wildcards", [])),
            len(deduped),
        )
        return deduped
