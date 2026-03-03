"""
VFS — Stage 4: Virtual Filesystem Population.

Creates a per-session virtual filesystem with ranked sections, a manifest,
and working directories for the agent. The VFS is the agent's primary
interface to retrieved content.

Session structure::

    /sessions/{session_id}/
      manifest.json              <- agent reads this first
      retrieved/
        rank1__docname__section.md
        rank2__docname__section.md
      conversation/
        history_summary.md
        recent_turns.json
      db_context/
        relevant_records.json
      working/
        scratchpad.md

Usage::

    config = AgentConfig()
    vfs = VFS(config)
    session = vfs.create_session(ranked_sections, intent_result)
    manifest = vfs.read_manifest(session.session_id)
"""

import json
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.config import AgentConfig
from agent.fusion_ranker import RankedSection
from agent.intent_classifier import IntentResult
from agent.monitoring import UsageTracker

logger = logging.getLogger(__name__)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _safe_filename(text: str) -> str:
    """Convert text to a safe filename component."""
    # Lowercase, replace non-alphanumeric with underscore
    safe = re.sub(r"[^\w]+", "_", text.lower().strip())
    # Trim underscores and truncate
    safe = safe.strip("_")[:50]
    return safe or "untitled"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """A single entry in the VFS manifest.json."""

    vfs_path: str
    source_file: str
    section: str
    one_line_summary: str
    retrieval_method: str
    final_score: float
    score_breakdown: Dict[str, float]
    why_included: str
    last_modified: str = ""
    fetch_more_hint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vfs_path": self.vfs_path,
            "source_file": self.source_file,
            "section": self.section,
            "one_line_summary": self.one_line_summary,
            "retrieval_method": self.retrieval_method,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown,
            "why_included": self.why_included,
            "last_modified": self.last_modified,
            "fetch_more_hint": self.fetch_more_hint,
        }


@dataclass
class SessionInfo:
    """Information about a VFS session."""

    session_id: str
    session_dir: str
    manifest_path: str
    section_count: int
    created_at: str


# ---------------------------------------------------------------------------
# VFS
# ---------------------------------------------------------------------------


class VFS:
    """
    Stage 4: Virtual Filesystem Population.

    Creates session directories with ranked sections written as individual
    .md files and a manifest.json for agent navigation.

    Args:
        config: AgentConfig with sessions_dir setting.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def _session_dir(self, session_id: str) -> str:
        """Get the session directory path."""
        return os.path.join(os.path.abspath(self.config.sessions_dir), session_id)

    def create_session(
        self,
        ranked_sections: List[RankedSection],
        intent: IntentResult,
        conversation_summary: str = "",
        recent_turns: Optional[List[Dict[str, str]]] = None,
        db_context: Optional[Dict[str, Any]] = None,
        registry_files: Optional[Dict[str, Any]] = None,
        tracker: Optional[UsageTracker] = None,
    ) -> SessionInfo:
        """
        Create a new VFS session populated with ranked sections.

        Args:
            ranked_sections:      Ranked sections from Stage 3.
            intent:               IntentResult from Stage 1.
            conversation_summary: Optional conversation history summary.
            recent_turns:         Optional list of recent conversation turns.
            db_context:           Optional database query results.
            registry_files:       Optional registry for last_modified lookup.
            tracker:              Optional UsageTracker.

        Returns:
            SessionInfo with session ID and directory details.
        """
        if tracker:
            tracker.start_stage("vfs_population")

        session_id = str(uuid.uuid4())[:12]
        session_dir = self._session_dir(session_id)
        registry_files = registry_files or {}

        logger.info(
            "VFS.create_session  id=%s  sections=%d",
            session_id,
            len(ranked_sections),
        )

        # Create directory structure
        dirs = [
            session_dir,
            os.path.join(session_dir, "retrieved"),
            os.path.join(session_dir, "conversation"),
            os.path.join(session_dir, "db_context"),
            os.path.join(session_dir, "working"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # Write ranked sections and build manifest
        manifest_entries: List[ManifestEntry] = []

        for rank, section in enumerate(ranked_sections, 1):
            # Build filename: rank{N}__{doc_name}__{section_title}.md
            doc_safe = _safe_filename(section.doc_name)
            title_safe = _safe_filename(section.section_title)
            filename = f"rank{rank}__{doc_safe}__{title_safe}.md"
            vfs_path = f"retrieved/{filename}"
            abs_path = os.path.join(session_dir, vfs_path)

            # Write section content
            header = f"## {section.section_title}\n"
            header += f"<!-- source: {section.doc_name} | section_id: {section.section_id} -->\n\n"
            content = header + _nfc(section.content)

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug(
                "VFS.create_session  wrote section  rank=%d  doc=%s  title=%r  size=%d",
                rank,
                section.doc_name,
                section.section_title[:50],
                len(content),
            )

            # Look up last_modified from registry
            last_modified = ""
            for path, entry in registry_files.items():
                basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if basename == section.doc_name or path == section.doc_name:
                    last_modified = entry.get("last_modified", "")
                    break

            # Build summary from first ~100 chars of content
            summary_text = section.content.strip()[:150].replace("\n", " ").strip()
            if len(section.content.strip()) > 150:
                summary_text += "..."

            manifest_entries.append(
                ManifestEntry(
                    vfs_path=vfs_path,
                    source_file=f"{section.doc_name}.md",
                    section=section.section_title,
                    one_line_summary=summary_text,
                    retrieval_method=section.retrieval_method,
                    final_score=section.final_score,
                    score_breakdown={
                        "bm25": section.bm25_score,
                        "vector": section.vector_score,
                        "grep_density": section.grep_density,
                    },
                    why_included=section.why_included,
                    last_modified=last_modified,
                    fetch_more_hint=False,
                )
            )

        # Write manifest.json
        manifest = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "intent_type": intent.intent_type,
            "query_variants": intent.query_variants,
            "sections": [e.to_dict() for e in manifest_entries],
        }
        manifest_path = os.path.join(session_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Write conversation context
        if conversation_summary:
            history_path = os.path.join(session_dir, "conversation", "history_summary.md")
            with open(history_path, "w", encoding="utf-8") as f:
                f.write(conversation_summary)

        if recent_turns:
            turns_path = os.path.join(session_dir, "conversation", "recent_turns.json")
            with open(turns_path, "w", encoding="utf-8") as f:
                json.dump(recent_turns, f, indent=2, ensure_ascii=False)

        # Write db_context
        if db_context and intent.needs_db:
            db_path = os.path.join(session_dir, "db_context", "relevant_records.json")
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(db_context, f, indent=2, ensure_ascii=False)

        # Create empty scratchpad
        scratchpad_path = os.path.join(session_dir, "working", "scratchpad.md")
        with open(scratchpad_path, "w", encoding="utf-8") as f:
            f.write("")

        if tracker:
            tracker.end_stage("vfs_population")

        session_info = SessionInfo(
            session_id=session_id,
            session_dir=session_dir,
            manifest_path=manifest_path,
            section_count=len(manifest_entries),
            created_at=manifest["created_at"],
        )

        logger.info(
            "VFS.create_session  done  id=%s  sections=%d  dir=%s",
            session_id,
            len(manifest_entries),
            session_dir,
        )
        return session_info

    def read_manifest(self, session_id: str) -> Dict[str, Any]:
        """Read the manifest.json for a session."""
        manifest_path = os.path.join(self._session_dir(session_id), "manifest.json")
        logger.debug("VFS.read_manifest  session=%s  path=%s", session_id, manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(
            "VFS.read_manifest  sections=%d  intent=%s",
            len(data.get("sections", [])),
            data.get("intent_type", ""),
        )
        return data

    def read_section(self, session_id: str, vfs_path: str) -> str:
        """Read a section file from the VFS."""
        abs_path = os.path.join(self._session_dir(session_id), vfs_path)
        if not os.path.exists(abs_path):
            logger.warning("VFS.read_section  not found  session=%s  path=%s", session_id, vfs_path)
            raise FileNotFoundError(f"VFS section not found: {vfs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(
            "VFS.read_section  session=%s  path=%s  size=%d",
            session_id,
            vfs_path,
            len(content),
        )
        return content

    def write_scratchpad(self, session_id: str, content: str) -> None:
        """Write to the agent's scratchpad."""
        scratchpad_path = os.path.join(self._session_dir(session_id), "working", "scratchpad.md")
        with open(scratchpad_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug("VFS.write_scratchpad  session=%s  size=%d", session_id, len(content))

    def read_scratchpad(self, session_id: str) -> str:
        """Read the agent's scratchpad."""
        scratchpad_path = os.path.join(self._session_dir(session_id), "working", "scratchpad.md")
        if not os.path.exists(scratchpad_path):
            logger.debug("VFS.read_scratchpad  session=%s  not found", session_id)
            return ""
        with open(scratchpad_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug("VFS.read_scratchpad  session=%s  size=%d", session_id, len(content))
        return content

    def add_fetched_section(
        self,
        session_id: str,
        doc_name: str,
        section_title: str,
        content: str,
        source_file: str = "",
    ) -> str:
        """
        Add a dynamically fetched section to the VFS (via fetch_section tool).

        Appends the section to the retrieved/ directory and updates manifest.json.

        Args:
            session_id:    The session ID.
            doc_name:      Document name.
            section_title: Section title.
            content:       Section content.
            source_file:   Source file path.

        Returns:
            The vfs_path of the newly added section.
        """
        session_dir = self._session_dir(session_id)
        manifest_path = os.path.join(session_dir, "manifest.json")

        # Read current manifest to determine next rank
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        current_count = len(manifest.get("sections", []))
        rank = current_count + 1

        # Write section file
        doc_safe = _safe_filename(doc_name)
        title_safe = _safe_filename(section_title)
        filename = f"rank{rank}__{doc_safe}__{title_safe}.md"
        vfs_path = f"retrieved/{filename}"
        abs_path = os.path.join(session_dir, vfs_path)

        header = f"## {section_title}\n"
        header += f"<!-- source: {doc_name} | fetched on demand -->\n\n"
        full_content = header + _nfc(content)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        # Update manifest
        summary_text = content.strip()[:150].replace("\n", " ").strip()
        if len(content.strip()) > 150:
            summary_text += "..."

        new_entry = {
            "vfs_path": vfs_path,
            "source_file": source_file or f"{doc_name}.md",
            "section": section_title,
            "one_line_summary": summary_text,
            "retrieval_method": "fetch_on_demand",
            "final_score": 0.0,
            "score_breakdown": {"bm25": 0.0, "vector": 0.0, "grep_density": 0.0},
            "why_included": "Fetched on demand by agent",
            "last_modified": "",
            "fetch_more_hint": False,
        }
        manifest["sections"].append(new_entry)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(
            "VFS.add_fetched_section  session=%s  doc=%s  section=%s",
            session_id,
            doc_name,
            section_title,
        )
        return vfs_path

    def cleanup_session(self, session_id: str) -> None:
        """Remove a session directory and all its contents."""
        import shutil

        session_dir = self._session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info("VFS.cleanup_session  id=%s", session_id)

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        sessions_dir = os.path.abspath(self.config.sessions_dir)
        if not os.path.isdir(sessions_dir):
            return []
        sessions = [
            d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))
        ]
        logger.debug("VFS.list_sessions  count=%d", len(sessions))
        return sessions
