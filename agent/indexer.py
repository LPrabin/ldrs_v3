"""
Indexer — End-to-end document indexing pipeline.

Orchestrates the full indexing flow for a single markdown file:

  1. Run PageIndex ``md_to_tree()`` to parse the .md into a structure tree.
  2. Flatten the tree into section dicts suitable for embedding.
  3. Save the structure JSON to the results directory.
  4. Embed all sections into PostgreSQL via ``Embedder``.
  5. Register the document in ``Registry`` with full metadata.

Usage::

    config = AgentConfig()
    indexer = Indexer(config)
    await indexer.startup()           # connect to DB
    result = await indexer.index_file("docs/auth.md")
    await indexer.shutdown()          # close DB pool

    # Or index all .md files in a directory:
    results = await indexer.index_directory("docs/")
"""

import asyncio
import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.config import AgentConfig
from agent.embedder import Embedder
from agent.registry import Registry

logger = logging.getLogger(__name__)


def _nfc(text: str) -> str:
    """Normalize to NFC form."""
    return unicodedata.normalize("NFC", text)


def _flatten_sections(
    nodes: List[dict], parent_titles: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Recursively flatten a PageIndex structure tree into a list of section dicts.

    Each section dict has:
        - node_id: str
        - title: str
        - text: str (the section's own text content)
        - line_num: int
        - breadcrumb: str (ancestor path)

    Only leaf-level text is included (the text directly attached to each node).
    This avoids duplicating parent text that is already covered by children.

    Args:
        nodes:         List of structure tree nodes from PageIndex.
        parent_titles: Accumulated ancestor titles for breadcrumb.

    Returns:
        Flat list of section dicts.
    """
    if parent_titles is None:
        parent_titles = []

    sections = []
    for node in nodes:
        title = _nfc(node.get("title", ""))
        node_id = node.get("node_id", "")
        text = _nfc(node.get("text", ""))
        line_num = node.get("line_num", 0)
        children = node.get("nodes") or []

        breadcrumb = " > ".join(parent_titles + [title]) if title else " > ".join(parent_titles)

        if text.strip():
            sections.append(
                {
                    "node_id": node_id,
                    "title": title,
                    "text": text,
                    "line_num": line_num,
                    "breadcrumb": breadcrumb,
                }
            )
            logger.debug(
                "_flatten_sections  node_id=%s  title=%r  text_len=%d",
                node_id,
                title[:40],
                len(text),
            )

        # Recurse into children
        if children:
            sections.extend(
                _flatten_sections(children, parent_titles + [title] if title else parent_titles)
            )

    return sections


@dataclass
class IndexResult:
    """Result of indexing a single document."""

    md_path: str
    doc_name: str
    index_path: str
    node_count: int
    section_count: int
    embedded_count: int
    success: bool
    error: str = ""


class Indexer:
    """
    End-to-end document indexing pipeline.

    Orchestrates PageIndex parsing, structure JSON persistence,
    section embedding into pgvector, and registry updates.

    Args:
        config:   AgentConfig instance.
        embedder: Optional pre-created Embedder (shared across callers).
        registry: Optional pre-created Registry (shared across callers).
    """

    def __init__(
        self,
        config: AgentConfig,
        embedder: Optional[Embedder] = None,
        registry: Optional[Registry] = None,
    ):
        self.config = config
        self._embedder = embedder
        self._registry = registry
        self._owns_embedder = embedder is None

    @property
    def embedder(self) -> Embedder:
        """Lazy-init embedder if not injected."""
        if self._embedder is None:
            self._embedder = Embedder(self.config)
        return self._embedder

    @property
    def registry(self) -> Registry:
        """Lazy-init registry if not injected."""
        if self._registry is None:
            self._registry = Registry(self.config.registry_path)
        return self._registry

    async def startup(self) -> None:
        """Connect the embedder to PostgreSQL."""
        await self.embedder.connect()
        logger.info("Indexer started")

    async def shutdown(self) -> None:
        """Close the embedder connection pool (only if we own it)."""
        if self._owns_embedder and self._embedder is not None:
            await self._embedder.close()
        logger.info("Indexer shut down")

    async def index_file(
        self,
        md_path: str,
        *,
        if_thinning: bool = False,
        min_token_threshold: int = 5000,
        if_add_node_summary: str = "no",
        summary_token_threshold: int = 200,
        model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        summary: str = "",
    ) -> IndexResult:
        """
        Index a single markdown file through the full pipeline.

        Steps:
          1. Run ``md_to_tree()`` to parse into a structure tree.
          2. Save structure JSON to ``results_dir/<doc_name>_structure.json``.
          3. Flatten tree into sections and embed via ``Embedder``.
          4. Register in ``Registry`` with metadata.
          5. Save registry.

        Args:
            md_path:                  Path to the source .md file.
            if_thinning:              Enable tree thinning (merge small nodes).
            min_token_threshold:      Min tokens for thinning threshold.
            if_add_node_summary:      "yes" to generate LLM summaries per node.
            summary_token_threshold:  Token threshold for summary generation.
            model:                    LLM model override for summaries.
            tags:                     Optional keyword tags for registry.
            summary:                  Optional one-line summary for registry.

        Returns:
            IndexResult with details of the operation.
        """
        md_path = os.path.abspath(md_path)
        doc_name = os.path.splitext(os.path.basename(md_path))[0]

        logger.info("Indexer.index_file  md_path=%s  doc_name=%s", md_path, doc_name)

        if not os.path.exists(md_path):
            return IndexResult(
                md_path=md_path,
                doc_name=doc_name,
                index_path="",
                node_count=0,
                section_count=0,
                embedded_count=0,
                success=False,
                error=f"File not found: {md_path}",
            )

        try:
            # --- Step 1: Parse markdown into structure tree ---
            from pageindex.page_index_md import md_to_tree

            use_model = model or self.config.default_model
            step_start = time.time()
            structure_data = await md_to_tree(
                md_path=md_path,
                if_thinning=if_thinning,
                min_token_threshold=min_token_threshold,
                if_add_node_summary=if_add_node_summary,
                summary_token_threshold=summary_token_threshold,
                model=use_model,
                if_add_node_text="yes",  # Always include text for embedding
            )
            parse_ms = (time.time() - step_start) * 1000

            structure = structure_data.get("structure", [])
            doc_name_from_pi = structure_data.get("doc_name", doc_name)

            logger.info(
                "Indexer.index_file  parsed  doc=%s  top_nodes=%d  parse_ms=%.0f",
                doc_name_from_pi,
                len(structure),
                parse_ms,
            )

            # --- Step 2: Save structure JSON ---
            step_start = time.time()
            os.makedirs(self.config.results_dir, exist_ok=True)
            index_path = os.path.join(self.config.results_dir, f"{doc_name}_structure.json")
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(structure_data, f, indent=2, ensure_ascii=False)
            save_ms = (time.time() - step_start) * 1000
            logger.info(
                "Indexer.index_file  saved structure  path=%s  save_ms=%.0f", index_path, save_ms
            )

            # --- Step 3: Flatten and embed ---
            step_start = time.time()
            sections = _flatten_sections(structure)
            flatten_ms = (time.time() - step_start) * 1000
            logger.info(
                "Indexer.index_file  flattened  sections=%d  flatten_ms=%.0f",
                len(sections),
                flatten_ms,
            )

            embedded_count = 0
            if sections:
                step_start = time.time()
                embedded_count = await self.embedder.embed_document(
                    doc_name=doc_name,
                    sections=sections,
                    source_file=md_path,
                )
                embed_ms = (time.time() - step_start) * 1000
                logger.debug(
                    "Indexer.index_file  embed_ms=%.0f  embedded=%d", embed_ms, embedded_count
                )

            # --- Step 4: Register ---
            # Compute relative path for registry key
            rel_path = md_path
            if self.config.docs_dir and md_path.startswith(os.path.abspath(self.config.docs_dir)):
                rel_path = os.path.relpath(md_path, os.path.abspath(self.config.docs_dir))

            self.registry.add_file(
                file_path=rel_path,
                structure=structure,
                summary=summary,
                tags=tags,
                index_path=index_path,
                md_path=md_path,
                has_embeddings=embedded_count > 0,
            )
            self.registry.save()

            # Count total nodes recursively
            from agent.registry import _count_nodes

            node_count = _count_nodes(structure)

            result = IndexResult(
                md_path=md_path,
                doc_name=doc_name,
                index_path=index_path,
                node_count=node_count,
                section_count=len(sections),
                embedded_count=embedded_count,
                success=True,
            )
            logger.info(
                "Indexer.index_file  complete  doc=%s  nodes=%d  sections=%d  embedded=%d",
                doc_name,
                node_count,
                len(sections),
                embedded_count,
            )
            return result

        except Exception as e:
            logger.error("Indexer.index_file  failed  doc=%s  error=%s", doc_name, e)
            return IndexResult(
                md_path=md_path,
                doc_name=doc_name,
                index_path="",
                node_count=0,
                section_count=0,
                embedded_count=0,
                success=False,
                error=str(e),
            )

    async def remove_file(self, md_path: str) -> bool:
        """
        Remove a document from the index (embeddings + registry).

        Args:
            md_path: Path to the .md file to remove.

        Returns:
            True if the document was found and removed.
        """
        doc_name = os.path.splitext(os.path.basename(md_path))[0]
        logger.info("Indexer.remove_file  doc=%s", doc_name)

        # Remove embeddings
        removed = await self.embedder.remove_document(doc_name)

        # Remove from registry
        rel_path = md_path
        if self.config.docs_dir and md_path.startswith(os.path.abspath(self.config.docs_dir)):
            rel_path = os.path.relpath(md_path, os.path.abspath(self.config.docs_dir))

        found = self.registry.remove_file(rel_path)
        if found:
            self.registry.save()

        # Remove structure JSON
        index_path = os.path.join(self.config.results_dir, f"{doc_name}_structure.json")
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info("Indexer.remove_file  deleted structure  path=%s", index_path)

        return found or removed > 0

    async def index_directory(
        self,
        directory: Optional[str] = None,
        **kwargs,
    ) -> List[IndexResult]:
        """
        Index all .md files in a directory.

        Args:
            directory: Path to scan. Defaults to ``config.docs_dir``.
            **kwargs:  Passed through to ``index_file()``.

        Returns:
            List of IndexResult for each file.
        """
        directory = directory or self.config.docs_dir
        directory = os.path.abspath(directory)

        if not os.path.isdir(directory):
            logger.warning("Indexer.index_directory  not a directory: %s", directory)
            return []

        md_files = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".md") and not f.startswith(".")
        )

        logger.info("Indexer.index_directory  dir=%s  files=%d", directory, len(md_files))

        results = []
        for md_file in md_files:
            result = await self.index_file(md_file, **kwargs)
            results.append(result)

        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        logger.info(
            "Indexer.index_directory  done  total=%d  success=%d  failed=%d",
            len(results),
            successes,
            failures,
        )
        return results
