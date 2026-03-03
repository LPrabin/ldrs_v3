"""
Watcher — File system watcher for automatic re-indexing.

Monitors configured directories for ``.md`` file changes and triggers
the indexing pipeline on create/modify/delete events.

Uses the ``watchdog`` library with configurable debounce to avoid
re-indexing on rapid successive saves.

Architecture (from AGENT_SYSTEM.md)::

    On CREATE / MODIFY:
      1. Re-chunk affected file into sections (PageIndex)
      2. Re-embed sections → upsert into postgres vector db
      3. Update registry.json atomically
      4. Log: timestamp, file, event_type, sections_updated

    On DELETE:
      - Remove embeddings from postgres vector db
      - Remove entry from registry.json
      - Remove structure JSON

    Debounce: configurable (default 2s)

Usage::

    config = AgentConfig()
    watcher = FileWatcher(config)
    await watcher.start()    # starts background thread
    # ... application runs ...
    await watcher.stop()     # stops watcher and cleans up
"""

import asyncio
import logging
import os
import threading
import time
from typing import Callable, Dict, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from agent.config import AgentConfig
from agent.indexer import Indexer

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """
    Watchdog event handler with debounce logic.

    Collects events for ``.md`` files and after the debounce interval
    triggers the callback with the set of affected paths.
    """

    def __init__(
        self,
        debounce_seconds: float,
        on_changes: Callable[[Dict[str, str]], None],
    ):
        super().__init__()
        self._debounce = debounce_seconds
        self._on_changes = on_changes
        self._pending: Dict[str, str] = {}  # path -> event_type
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _schedule_flush(self) -> None:
        """Schedule a flush after debounce interval."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()
        logger.debug(
            "_DebouncedHandler._schedule_flush  pending=%d  debounce=%.1fs",
            len(self._pending),
            self._debounce,
        )

    def _flush(self) -> None:
        """Flush pending events to the callback."""
        with self._lock:
            if not self._pending:
                logger.debug("_DebouncedHandler._flush  no pending events, skipping")
                return
            events = dict(self._pending)
            self._pending.clear()

        logger.info(
            "FileWatcher flush  events=%d  paths=%s",
            len(events),
            list(events.keys()),
        )
        self._on_changes(events)

    def _handle_event(self, event: FileSystemEvent, event_type: str) -> None:
        """Handle a single filesystem event."""
        if event.is_directory:
            return

        src_path = event.src_path
        if not src_path.endswith(".md"):
            return

        logger.debug("FileWatcher event  type=%s  path=%s", event_type, src_path)

        with self._lock:
            # For moves, treat the destination as a create
            if event_type == "moved" and hasattr(event, "dest_path"):
                dest_path = event.dest_path
                if dest_path.endswith(".md"):
                    self._pending[dest_path] = "created"
                self._pending[src_path] = "deleted"
            else:
                self._pending[src_path] = event_type

        self._schedule_flush()

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "moved")

    def cancel(self) -> None:
        """Cancel any pending timer."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class FileWatcher:
    """
    File system watcher that triggers re-indexing on .md file changes.

    Monitors directories specified in ``config.watch_dirs`` using watchdog.
    On changes, runs the Indexer pipeline to re-index affected files.

    Args:
        config:  AgentConfig instance.
        indexer: Optional pre-created Indexer (shared with pipeline).
    """

    def __init__(
        self,
        config: AgentConfig,
        indexer: Optional[Indexer] = None,
    ):
        self.config = config
        self._indexer = indexer
        self._owns_indexer = indexer is None
        self._observer: Optional[Observer] = None
        self._handler: Optional[_DebouncedHandler] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    @property
    def indexer(self) -> Indexer:
        """Lazy-init indexer if not injected."""
        if self._indexer is None:
            self._indexer = Indexer(self.config)
        return self._indexer

    def _on_changes(self, events: Dict[str, str]) -> None:
        """
        Callback from the debounced handler. Runs in watchdog's thread.

        Schedules async indexing tasks on the event loop.
        """
        if self._loop is None:
            logger.warning("FileWatcher._on_changes  no event loop set")
            return

        for path, event_type in events.items():
            logger.info("FileWatcher._on_changes  type=%s  path=%s", event_type, path)
            if event_type == "deleted":
                asyncio.run_coroutine_threadsafe(self._handle_delete(path), self._loop)
            else:  # created or modified
                asyncio.run_coroutine_threadsafe(self._handle_upsert(path), self._loop)

    async def _handle_upsert(self, md_path: str) -> None:
        """Re-index a created or modified file."""
        try:
            result = await self.indexer.index_file(md_path)
            if result.success:
                logger.info(
                    "FileWatcher  indexed  doc=%s  nodes=%d  embedded=%d",
                    result.doc_name,
                    result.node_count,
                    result.embedded_count,
                )
                # Update watcher sync timestamp
                self.indexer.registry.update_watcher_sync()
                self.indexer.registry.save()
            else:
                logger.error(
                    "FileWatcher  index failed  doc=%s  error=%s",
                    result.doc_name,
                    result.error,
                )
        except Exception as e:
            logger.error("FileWatcher._handle_upsert  error=%s  path=%s", e, md_path)

    async def _handle_delete(self, md_path: str) -> None:
        """Remove a deleted file from the index."""
        try:
            removed = await self.indexer.remove_file(md_path)
            logger.info("FileWatcher  removed  path=%s  found=%s", md_path, removed)
            # Update watcher sync timestamp
            self.indexer.registry.update_watcher_sync()
            self.indexer.registry.save()
        except Exception as e:
            logger.error("FileWatcher._handle_delete  error=%s  path=%s", e, md_path)

    async def start(self) -> None:
        """
        Start the file watcher.

        Creates watchdog observer threads for all configured watch directories.
        Must be called from an async context (event loop must be running).
        """
        if self._running:
            logger.warning("FileWatcher.start  already running")
            return

        self._loop = asyncio.get_running_loop()

        # Start the indexer's DB connection
        await self.indexer.startup()

        self._handler = _DebouncedHandler(
            debounce_seconds=self.config.watch_debounce,
            on_changes=self._on_changes,
        )

        self._observer = Observer()

        for watch_dir in self.config.watch_dirs:
            abs_dir = os.path.abspath(watch_dir)
            if os.path.isdir(abs_dir):
                self._observer.schedule(self._handler, abs_dir, recursive=True)
                logger.info("FileWatcher  watching  dir=%s", abs_dir)
            else:
                logger.warning("FileWatcher  dir not found, skipping: %s", abs_dir)

        self._observer.daemon = True
        self._observer.start()
        self._running = True

        logger.info(
            "FileWatcher started  dirs=%s  debounce=%.1fs",
            self.config.watch_dirs,
            self.config.watch_debounce,
        )

    async def stop(self) -> None:
        """Stop the file watcher and clean up resources."""
        if not self._running:
            return

        if self._handler:
            self._handler.cancel()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        if self._owns_indexer:
            await self.indexer.shutdown()

        self._running = False
        logger.info("FileWatcher stopped")

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently active."""
        return self._running
