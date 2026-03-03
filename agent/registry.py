"""
Registry — Document registry following the AGENT_SYSTEM.md schema.

Maintains a corpus-level inventory of all indexed documents with rich
metadata: summaries, tags, section lists, token counts, embedding status,
and last-modified timestamps.

Registry Schema (per file)::

    {
        "registry_version": "1.0",
        "generated_at": "2026-02-28T10:00:00Z",
        "last_watcher_sync": "2026-02-28T09:58:44Z",
        "files": {
            "api/authentication.md": {
                "summary": "OAuth2 flow, token refresh, API key management",
                "tags": ["auth", "oauth", "security"],
                "sections": ["Overview", "OAuth2 Flow", "Token Refresh"],
                "last_modified": "2026-02-01",
                "size_tokens": 1200,
                "has_embeddings": true,
                "node_count": 15,
                "index_path": "results/authentication_structure.json",
                "md_path": "docs/authentication.md"
            }
        }
    }

Usage::

    registry = Registry("results/registry.json")
    registry.add_file("docs/auth.md", structure, token_count=1200, tags=["auth"])
    registry.save()
    print(registry.get_corpus_summary())
"""

import json
import logging
import os
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


def _nfc(text: str) -> str:
    """Normalize to NFC form."""
    return unicodedata.normalize("NFC", text)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (gpt-4o encoding)."""
    if not text:
        return 0
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))


def _extract_sections(structure: List[dict]) -> List[str]:
    """Extract top-level section titles from a structure tree."""
    sections = []
    for node in structure:
        title = node.get("title", "")
        if title:
            sections.append(_nfc(title))
    return sections


def _count_nodes(nodes: List[dict]) -> int:
    """Recursively count all nodes in a structure tree."""
    total = 0
    for node in nodes:
        if node.get("node_id"):
            total += 1
        total += _count_nodes(node.get("nodes") or [])
    return total


def _collect_all_text(nodes: List[dict]) -> str:
    """Recursively collect all text from a structure tree."""
    texts = []
    for node in nodes:
        text = node.get("text", "")
        if text:
            texts.append(text)
        summary = node.get("summary", "")
        if summary:
            texts.append(summary)
        children = node.get("nodes") or []
        if children:
            texts.append(_collect_all_text(children))
    return "\n".join(texts)


class Registry:
    """
    Document registry following the AGENT_SYSTEM.md schema.

    Manages the corpus inventory with rich metadata per file. The registry
    is the primary input to the Intent Classifier (Stage 1) — the LLM
    reads it to decide which files are relevant.

    Args:
        registry_path: Path to the registry.json file.
    """

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.data: Dict[str, Any] = {
            "registry_version": "1.0",
            "generated_at": "",
            "last_watcher_sync": "",
            "files": {},
        }
        self._load()

    def _load(self) -> None:
        """Load existing registry from disk if it exists."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(
                    "Registry loaded  path=%s  files=%d",
                    self.registry_path,
                    len(self.data.get("files", {})),
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Registry load failed, starting fresh: %s", e)
        else:
            logger.info("Registry not found, starting fresh: %s", self.registry_path)

    def save(self) -> None:
        """Atomically save registry to disk (tmp write → rename)."""
        self.data["generated_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)

        tmp_path = self.registry_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.registry_path)
        logger.info("Registry saved  path=%s  files=%d", self.registry_path, len(self.files))

    @property
    def files(self) -> Dict[str, Any]:
        """Get the files dict."""
        return self.data.get("files", {})

    def add_file(
        self,
        file_path: str,
        structure: List[dict],
        *,
        summary: str = "",
        tags: Optional[List[str]] = None,
        index_path: str = "",
        md_path: str = "",
        has_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Add or update a file entry in the registry.

        Args:
            file_path:      Relative path to the source .md file (used as key).
            structure:      The PageIndex structure tree for this document.
            summary:        One-line document summary.
            tags:           List of keyword tags.
            index_path:     Path to the *_structure.json file.
            md_path:        Path to the source .md file.
            has_embeddings: Whether embeddings have been generated.

        Returns:
            The created/updated registry entry dict.
        """
        sections = _extract_sections(structure)
        node_count = _count_nodes(structure)
        all_text = _collect_all_text(structure)
        size_tokens = _count_tokens(all_text)

        last_modified = ""
        if md_path and os.path.exists(md_path):
            mtime = os.path.getmtime(md_path)
            last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")

        entry = {
            "summary": _nfc(summary),
            "tags": tags or [],
            "sections": sections,
            "last_modified": last_modified,
            "size_tokens": size_tokens,
            "has_embeddings": has_embeddings,
            "node_count": node_count,
            "index_path": index_path,
            "md_path": md_path,
        }

        self.data["files"][_nfc(file_path)] = entry
        logger.info(
            "Registry.add_file  path=%s  sections=%d  nodes=%d  tokens=%d",
            file_path,
            len(sections),
            node_count,
            size_tokens,
        )
        return entry

    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the registry. Returns True if found and removed."""
        key = _nfc(file_path)
        if key in self.data["files"]:
            del self.data["files"][key]
            logger.info("Registry.remove_file  path=%s", file_path)
            return True
        return False

    def mark_embeddings(self, file_path: str, has_embeddings: bool = True) -> None:
        """Update the has_embeddings flag for a file."""
        key = _nfc(file_path)
        if key in self.data["files"]:
            self.data["files"][key]["has_embeddings"] = has_embeddings
            logger.debug(
                "Registry.mark_embeddings  path=%s  has_embeddings=%s",
                file_path,
                has_embeddings,
            )
        else:
            logger.warning("Registry.mark_embeddings  path=%s  not found in registry", file_path)

    def update_watcher_sync(self) -> None:
        """Update the last_watcher_sync timestamp."""
        ts = datetime.now(timezone.utc).isoformat()
        self.data["last_watcher_sync"] = ts
        logger.debug("Registry.update_watcher_sync  timestamp=%s", ts)

    def get_corpus_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire corpus.

        Returns:
            Dict with total_files, total_tokens, files_with_embeddings, etc.
        """
        files = self.files
        summary = {
            "total_files": len(files),
            "total_tokens": sum(f.get("size_tokens", 0) for f in files.values()),
            "total_nodes": sum(f.get("node_count", 0) for f in files.values()),
            "files_with_embeddings": sum(1 for f in files.values() if f.get("has_embeddings")),
            "file_names": list(files.keys()),
        }
        logger.info(
            "Registry.get_corpus_summary  files=%d  tokens=%d  nodes=%d  with_embeddings=%d",
            summary["total_files"],
            summary["total_tokens"],
            summary["total_nodes"],
            summary["files_with_embeddings"],
        )
        return summary

    def get_for_llm(self) -> Dict[str, Any]:
        """
        Get a compact version of the registry suitable for LLM context.

        Strips internal paths and keeps only what the Intent Classifier needs:
        summary, tags, sections, last_modified, size_tokens, has_embeddings.
        """
        compact = {}
        for path, entry in self.files.items():
            compact[path] = {
                "summary": entry.get("summary", ""),
                "tags": entry.get("tags", []),
                "sections": entry.get("sections", []),
                "last_modified": entry.get("last_modified", ""),
                "size_tokens": entry.get("size_tokens", 0),
                "has_embeddings": entry.get("has_embeddings", False),
            }
        result = {
            "registry_version": self.data.get("registry_version", "1.0"),
            "files": compact,
        }
        logger.debug(
            "Registry.get_for_llm  files=%d  total_sections=%d",
            len(compact),
            sum(len(v.get("sections", [])) for v in compact.values()),
        )
        return result
