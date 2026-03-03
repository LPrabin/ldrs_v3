"""
AgentConfig — Centralised configuration for the Hybrid RAG Deep Agent System.

All directories, model names, database credentials, and tuning parameters
are loaded from environment variables with sensible defaults.

Usage::

    config = AgentConfig()                    # reads from .env
    config = AgentConfig(default_model="gpt-4o")  # override
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Centralised configuration for the LDRS v3 pipeline.

    Environment variables are read at instantiation. Constructor args
    override env values.

    LLM calls are routed through **LiteLLM**, which selects the provider
    based on the model-name prefix:

        openai/gpt-4o          → OpenAI
        gemini/gemini-2.0-flash → Google Gemini
        ollama/llama3           → local Ollama
        qwen3-vl               → custom OpenAI-compatible endpoint (api_base)

    API keys are resolved in this order:
        1. Explicit ``api_key`` constructor arg / ``API_KEY`` env var
        2. Provider-specific env vars read by LiteLLM automatically:
           ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, etc.

    Attributes:
        # LLM
        api_key:            Fallback API key (used when provider-specific key is absent).
        api_base:           Base URL for local / custom OpenAI-compatible endpoints.
                            Ignored when using cloud providers (openai/*, gemini/*).
        default_model:      LiteLLM model string for chat completions.
        embedding_model:    LiteLLM model string for embeddings.

        # PostgreSQL
        postgres_host:      PostgreSQL host.
        postgres_port:      PostgreSQL port.
        postgres_db:        PostgreSQL database name.
        postgres_user:      PostgreSQL user.
        postgres_password:  PostgreSQL password.
        embedding_dim:      Embedding vector dimension.

        # LangSmith
        langsmith_tracing:  Enable LangSmith tracing.
        langsmith_api_key:  LangSmith API key.
        langsmith_project:  LangSmith project name.

        # File Watcher
        watch_dirs:         List of directories to monitor for .md changes.
        watch_debounce:     Debounce interval in seconds.

        # Retrieval
        max_vfs_sections:   Max sections to populate in VFS per query.
        max_context_chars:  Character budget for retrieved context.
        max_grep_results:   Max results per TreeGrep search.

        # Agent
        max_agent_iterations: Max agent loop iterations before forced synthesis.

        # Directories
        results_dir:        Directory for structure JSONs and registry.
        docs_dir:           Directory for source .md files.
        sessions_dir:       Directory for VFS session data.
    """

    # LLM settings — provider-agnostic via LiteLLM
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv("API_BASE", os.getenv("BASE_URL", "")))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "qwen3-vl"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    )

    # PostgreSQL
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "ldrs_v3"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "ldrs"))
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "ldrs_secret")
    )
    embedding_dim: int = 3072

    # LangSmith
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "ldrs-v3")
    )

    # File Watcher
    watch_dirs: List[str] = field(
        default_factory=lambda: [
            d.strip() for d in os.getenv("WATCH_DIRS", "./docs").split(",") if d.strip()
        ]
    )
    watch_debounce: float = field(default_factory=lambda: float(os.getenv("WATCH_DEBOUNCE", "2.0")))

    # Retrieval tuning
    max_vfs_sections: int = field(default_factory=lambda: int(os.getenv("MAX_VFS_SECTIONS", "15")))
    max_context_chars: int = 15_000
    max_grep_results: int = 50

    # Agent tuning
    max_agent_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
    )

    # BM25 Fusion weights (default balanced)
    bm25_weight: float = 0.4
    vector_weight: float = 0.4
    grep_weight: float = 0.2

    # Directories
    results_dir: str = "./results"
    docs_dir: str = "./docs"
    sessions_dir: str = "./sessions"
    registry_path: Optional[str] = None

    def __post_init__(self):
        """Set derived defaults and ensure directories exist."""
        if self.registry_path is None:
            self.registry_path = os.path.join(self.results_dir, "registry.json")

        logger.info(
            "AgentConfig loaded  model=%s  api_base=%s  embedding_model=%s  "
            "postgres=%s:%s/%s  results_dir=%s  docs_dir=%s  sessions_dir=%s",
            self.default_model,
            self.api_base or "(provider default)",
            self.embedding_model,
            self.postgres_host,
            self.postgres_port,
            self.postgres_db,
            self.results_dir,
            self.docs_dir,
            self.sessions_dir,
        )
        logger.debug(
            "AgentConfig details  embedding_dim=%d  max_vfs_sections=%d  "
            "max_context_chars=%d  max_grep_results=%d  max_agent_iterations=%d  "
            "bm25_weight=%.2f  vector_weight=%.2f  grep_weight=%.2f  "
            "langsmith_tracing=%s  watch_dirs=%s  watch_debounce=%.1fs",
            self.embedding_dim,
            self.max_vfs_sections,
            self.max_context_chars,
            self.max_grep_results,
            self.max_agent_iterations,
            self.bm25_weight,
            self.vector_weight,
            self.grep_weight,
            self.langsmith_tracing,
            self.watch_dirs,
            self.watch_debounce,
        )
        if not self.api_key and not self._has_provider_prefix(self.default_model):
            logger.warning(
                "AgentConfig  API_KEY is empty and model has no provider prefix — "
                "LLM calls may fail unless provider env vars are set"
            )
        if self.langsmith_tracing and not self.langsmith_api_key:
            logger.warning("AgentConfig  LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty")

    # ── Runtime config update ──────────────────────────────────────

    def update_llm_settings(
        self,
        *,
        default_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Update LLM-related settings at runtime.

        Only non-None arguments are applied.  Returns a dict of fields
        that actually changed (old → new) so callers can decide whether
        to invalidate caches.
        """
        changes: dict = {}

        if default_model is not None and default_model != self.default_model:
            changes["default_model"] = {"old": self.default_model, "new": default_model}
            self.default_model = default_model

        if embedding_model is not None and embedding_model != self.embedding_model:
            changes["embedding_model"] = {"old": self.embedding_model, "new": embedding_model}
            self.embedding_model = embedding_model

        if api_key is not None and api_key != self.api_key:
            changes["api_key"] = {"old": "***", "new": "***"}
            self.api_key = api_key

        if api_base is not None and api_base != self.api_base:
            changes["api_base"] = {"old": self.api_base, "new": api_base}
            self.api_base = api_base

        if changes:
            logger.info("AgentConfig.update_llm_settings  changed=%s", list(changes.keys()))
        else:
            logger.debug("AgentConfig.update_llm_settings  no changes")

        return changes

    def get_llm_settings(self) -> dict:
        """Return the current LLM/embedding provider settings (safe for API responses)."""
        return {
            "default_model": self.default_model,
            "embedding_model": self.embedding_model,
            "api_base": self.api_base or "",
            "api_key_set": bool(self.api_key),
        }

    # ── LiteLLM helpers ─────────────────────────────────────────────

    @staticmethod
    def _has_provider_prefix(model: str) -> bool:
        """Return True if model string has a LiteLLM provider prefix."""
        return "/" in model

    @property
    def litellm_chat_kwargs(self) -> dict:
        """Base kwargs for ``litellm.acompletion()`` calls.

        Includes ``model``, and conditionally ``api_key`` and ``api_base``
        so that LiteLLM routes to the correct provider.
        """
        kw: dict = {"model": self.default_model}
        
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if self.default_model.startswith("gemini/") and gemini_key:
            kw["api_key"] = gemini_key
        elif self.api_key:
            kw["api_key"] = self.api_key
            
        if self.api_base and not self._has_provider_prefix(self.default_model):
            kw["api_base"] = self.api_base
        logger.debug("AgentConfig.litellm_chat_kwargs  %s", kw)
        return kw

    @property
    def litellm_embedding_kwargs(self) -> dict:
        """Base kwargs for ``litellm.aembedding()`` calls.

        Includes ``model``, and conditionally ``api_key`` and ``api_base``.
        """
        kw: dict = {"model": self.embedding_model}
        
        # Log which API key is being used for debugging
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        logger.debug("Embedding API key check - GEMINI_API_KEY present: %s, value starts with: %s", 
                    bool(gemini_key), gemini_key[:10] if gemini_key else "None")
        
        if self.embedding_model.startswith("gemini/") and gemini_key:
            kw["api_key"] = gemini_key
        elif self.api_key:
            kw["api_key"] = self.api_key
            
        if self.api_base and (not self._has_provider_prefix(self.embedding_model) or self.embedding_model.startswith("openai/")):
            kw["api_base"] = self.api_base
        logger.debug("AgentConfig.litellm_embedding_kwargs  embedding_model=%s, will use provider: %s", 
                    self.embedding_model, self.embedding_model.split("/")[0] if "/" in self.embedding_model else "unknown")
        logger.debug("AgentConfig.litellm_embedding_kwargs  %s", kw)
        return kw

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL connection string."""
        dsn = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        logger.debug(
            "AgentConfig.postgres_dsn  host=%s  port=%s  db=%s",
            self.postgres_host,
            self.postgres_port,
            self.postgres_db,
        )
        return dsn

    @property
    def async_postgres_dsn(self) -> str:
        """Async PostgreSQL connection string (for asyncpg)."""
        dsn = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        logger.debug(
            "AgentConfig.async_postgres_dsn  host=%s  port=%s  db=%s",
            self.postgres_host,
            self.postgres_port,
            self.postgres_db,
        )
        return dsn
