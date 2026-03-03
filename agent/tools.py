"""
Tools — Agent tools for the Deep Agent Loop (Stage 5).

Provides the tool functions that the agent can call during its reasoning loop:
  - read_section(vfs_path)         — read a retrieved section from VFS
  - fetch_section(source_file, section_header) — pull additional section on demand
  - search_conversation_history(query)         — search prior conversation
  - write_scratchpad(content)      — write to private working memory
  - read_scratchpad()              — read working memory

The ``query_db`` tool is deferred (skipped for now per user decision).

These tools are implemented as plain functions that the agent loop
wraps as LangChain-compatible tools.

Usage::

    tools = AgentTools(config, vfs=vfs, session_id=session_id)
    result = tools.read_section("retrieved/rank1__auth__oauth_flow.md")
"""

import json
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from agent.config import AgentConfig
from agent.vfs import VFS

logger = logging.getLogger(__name__)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class AgentTools:
    """
    Collection of tools available to the Deep Agent.

    Each method corresponds to a tool the agent can invoke during
    Stage 5 (Agent Loop). The tools operate on the VFS session created
    in Stage 4.

    Args:
        config:     AgentConfig instance.
        vfs:        VFS instance for session filesystem access.
        session_id: The current session ID.
    """

    def __init__(
        self,
        config: AgentConfig,
        vfs: VFS,
        session_id: str,
    ):
        self.config = config
        self.vfs = vfs
        self.session_id = session_id
        self._sections_read: List[str] = []  # Track which sections were read

    @property
    def sections_read(self) -> List[str]:
        """List of VFS paths the agent has read."""
        return list(self._sections_read)

    def read_section(self, vfs_path: str) -> str:
        """
        Read a section from the VFS.

        The agent calls this to read sections listed in the manifest.
        Only sections actually read via this tool may be cited.

        Args:
            vfs_path: Path within the session VFS (e.g., "retrieved/rank1__...md").

        Returns:
            The section content as a string.
        """
        logger.info(
            "AgentTools.read_section  session=%s  path=%s",
            self.session_id,
            vfs_path,
        )
        try:
            content = self.vfs.read_section(self.session_id, vfs_path)
            self._sections_read.append(vfs_path)
            logger.debug(
                "AgentTools.read_section  success  path=%s  content_len=%d",
                vfs_path,
                len(content),
            )
            return content
        except FileNotFoundError:
            return f"Error: Section not found at {vfs_path}"
        except Exception as e:
            logger.error("AgentTools.read_section  error=%s", e)
            return f"Error reading section: {e}"

    def fetch_section(
        self,
        source_file: str,
        section_header: str,
    ) -> str:
        """
        Fetch a specific section from a source document on demand.

        Searches the structure JSON for the matching section title,
        extracts its content, writes it to the VFS, and returns it.

        Args:
            source_file:    Source document name (e.g., "authentication.md").
            section_header: Section title to fetch (e.g., "Token Refresh").

        Returns:
            The section content, or an error message if not found.
        """
        logger.info(
            "AgentTools.fetch_section  source=%s  header=%s",
            source_file,
            section_header,
        )

        # Derive doc_name and look up structure JSON
        doc_name = os.path.splitext(os.path.basename(source_file))[0]
        index_path = os.path.join(
            os.path.abspath(self.config.results_dir),
            f"{doc_name}_structure.json",
        )

        if not os.path.exists(index_path):
            return f"Error: Structure file not found for {source_file}"

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                structure_data = json.load(f)

            structure = structure_data.get("structure", [])
            section = self._find_section(structure, section_header)

            if section is None:
                return (
                    f"Error: Section '{section_header}' not found in {source_file}. "
                    f"Available sections: {self._list_sections(structure)}"
                )

            content = section.get("text", "")
            title = section.get("title", section_header)
            logger.debug(
                "AgentTools.fetch_section  found  title=%r  content_len=%d",
                title,
                len(content),
            )

            # Add to VFS
            vfs_path = self.vfs.add_fetched_section(
                session_id=self.session_id,
                doc_name=doc_name,
                section_title=title,
                content=content,
                source_file=source_file,
            )

            self._sections_read.append(vfs_path)
            return content

        except Exception as e:
            logger.error("AgentTools.fetch_section  error=%s", e)
            return f"Error fetching section: {e}"

    def _find_section(self, nodes: List[dict], title: str) -> Optional[dict]:
        """Recursively search for a section by title (case-insensitive)."""
        title_lower = _nfc(title).lower()
        for node in nodes:
            node_title = _nfc(node.get("title", "")).lower()
            if node_title == title_lower or title_lower in node_title:
                logger.debug(
                    "AgentTools._find_section  matched  query=%r  found=%r  node_id=%s",
                    title,
                    node.get("title", ""),
                    node.get("node_id", ""),
                )
                return node
            children = node.get("nodes") or []
            if children:
                found = self._find_section(children, title)
                if found:
                    return found
        return None

    def _list_sections(self, nodes: List[dict], depth: int = 0) -> str:
        """List available section titles for error messages."""
        titles = []
        for node in nodes:
            title = node.get("title", "")
            if title:
                titles.append(title)
            children = node.get("nodes") or []
            if children and depth < 2:
                for child in children:
                    child_title = child.get("title", "")
                    if child_title:
                        titles.append(f"  {child_title}")
        return ", ".join(titles[:20])

    def search_conversation_history(self, query: str) -> str:
        """
        Search prior conversation history.

        Reads the conversation summary and recent turns from the VFS
        and returns relevant excerpts.

        Args:
            query: Search query for conversation history.

        Returns:
            Matching conversation context, or a message if none found.
        """
        logger.info(
            "AgentTools.search_conversation_history  query=%r",
            query[:50],
        )

        session_dir = os.path.join(os.path.abspath(self.config.sessions_dir), self.session_id)

        results = []

        # Check history summary
        summary_path = os.path.join(session_dir, "conversation", "history_summary.md")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read()
            if summary.strip():
                results.append(f"## Conversation Summary\n{summary}")

        # Check recent turns
        turns_path = os.path.join(session_dir, "conversation", "recent_turns.json")
        if os.path.exists(turns_path):
            with open(turns_path, "r", encoding="utf-8") as f:
                turns = json.load(f)
            if turns:
                # Simple keyword matching on turns
                query_lower = query.lower()
                matching_turns = []
                for turn in turns:
                    content = turn.get("content", "")
                    if query_lower in content.lower():
                        matching_turns.append(turn)
                if matching_turns:
                    results.append(f"## Matching Turns\n{json.dumps(matching_turns, indent=2)}")

        if not results:
            logger.debug("AgentTools.search_conversation_history  no matches found")
            return "No relevant conversation history found."

        logger.debug(
            "AgentTools.search_conversation_history  results=%d  total_len=%d",
            len(results),
            sum(len(r) for r in results),
        )
        return "\n\n".join(results)

    def write_scratchpad(self, content: str) -> str:
        """
        Write to the agent's private scratchpad.

        Content must be valid markdown following the required format:
          ## Reasoning
          ## Key Facts
          ## Open Questions
          ## Synthesis Plan

        Args:
            content: Markdown content to write.

        Returns:
            Confirmation message or error.
        """
        logger.info(
            "AgentTools.write_scratchpad  session=%s  length=%d",
            self.session_id,
            len(content),
        )

        # Basic validation — must be non-empty markdown
        if not content.strip():
            return "Error: Cannot write empty scratchpad."

        try:
            self.vfs.write_scratchpad(self.session_id, content)
            return "Scratchpad updated successfully."
        except Exception as e:
            logger.error("AgentTools.write_scratchpad  error=%s", e)
            return f"Error writing scratchpad: {e}"

    def read_scratchpad(self) -> str:
        """
        Read the agent's current scratchpad.

        Returns:
            The scratchpad content, or a message if empty.
        """
        logger.info(
            "AgentTools.read_scratchpad  session=%s",
            self.session_id,
        )
        content = self.vfs.read_scratchpad(self.session_id)
        return content if content else "Scratchpad is empty."

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in OpenAI function-calling format.

        Returns:
            List of tool definition dicts for the LLM.
        """
        logger.debug("AgentTools.get_tool_definitions  returning 5 tool definitions")
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_section",
                    "description": (
                        "Read a section from the VFS. Use for sections listed "
                        "in the manifest. Only sections read via this tool may be cited."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vfs_path": {
                                "type": "string",
                                "description": "Path within the VFS, e.g. 'retrieved/rank1__auth__oauth_flow.md'",
                            },
                        },
                        "required": ["vfs_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_section",
                    "description": (
                        "Fetch a specific section from a source document on demand. "
                        "Use when: manifest hints more content exists, following a "
                        "multihop reference, or self-sufficiency check reveals a gap."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_file": {
                                "type": "string",
                                "description": "Source document name, e.g. 'authentication.md'",
                            },
                            "section_header": {
                                "type": "string",
                                "description": "Section title to fetch, e.g. 'Token Refresh'",
                            },
                        },
                        "required": ["source_file", "section_header"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_conversation_history",
                    "description": (
                        "Search prior conversation turns. Use when the user refers "
                        "to something said earlier or the query is a follow-up."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for conversation history",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_scratchpad",
                    "description": (
                        "Write to your private working memory. Use required format: "
                        "## Reasoning, ## Key Facts, ## Open Questions, ## Synthesis Plan. "
                        "Every fact must have a source citation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Markdown content to write to scratchpad",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_scratchpad",
                    "description": (
                        "Read your current scratchpad. Use to review reasoning "
                        "mid-task or before final synthesis."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]
