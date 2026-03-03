"""
LDRS v3 — Streamlit Chat Interface.

A chat-based UI that talks to the FastAPI backend at /query.
Displays answers with citations, pipeline metadata, and usage stats.

Usage::

    streamlit run ui/streamlit_app.py
"""

import json
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8001")

st.set_page_config(
    page_title="LDRS v3 — RAG Agent",
    page_icon="🔍",  # noqa: RUF001
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_query(
    query: str, conversation_summary: str = "", recent_turns: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Send a query to the API and return the response."""
    payload = {
        "query": query,
        "conversation_summary": conversation_summary,
        "recent_turns": recent_turns or [],
        "cleanup_session": False,
    }
    try:
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Could not connect to API server. Is it running?",
            "success": False,
            "error": "Connection refused",
        }
    except requests.exceptions.Timeout:
        return {"answer": "Request timed out (120s).", "success": False, "error": "Timeout"}
    except Exception as e:
        return {"answer": f"API error: {e}", "success": False, "error": str(e)}


def api_health() -> Dict[str, Any]:
    """Check API health."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "pipeline_started": False}


def api_corpus() -> Dict[str, Any]:
    """Get corpus summary."""
    try:
        resp = requests.get(f"{API_BASE}/corpus", timeout=10)
        return resp.json()
    except Exception:
        return {}


def api_get_config() -> Dict[str, Any]:
    """Get current LLM/embedding provider config from API."""
    try:
        resp = requests.get(f"{API_BASE}/config", timeout=5)
        return resp.json()
    except Exception:
        return {}


def api_update_config(
    default_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Push LLM/embedding config changes to the API."""
    payload: Dict[str, Any] = {}
    if default_model is not None:
        payload["default_model"] = default_model
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
    if api_key is not None:
        payload["api_key"] = api_key
    if api_base is not None:
        payload["api_base"] = api_base

    if not payload:
        return {"changed": False}

    try:
        resp = requests.put(f"{API_BASE}/config", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("LDRS v3")
st.sidebar.caption("Hybrid RAG Deep Agent System")

# Health check
health = api_health()
if health.get("pipeline_started"):
    st.sidebar.success(f"API: Connected ({health.get('corpus_files', 0)} files)")
else:
    st.sidebar.error(f"API: {health.get('status', 'unreachable')}")

# LLM / Embedding provider configuration
with st.sidebar.expander("LLM / Embedding Config", expanded=False):
    # Fetch current config from API (or fall back to health response)
    if "llm_config" not in st.session_state:
        remote_cfg = api_get_config()
        if remote_cfg:
            st.session_state.llm_config = remote_cfg
        else:
            # Fall back to health data
            st.session_state.llm_config = {
                "default_model": health.get("default_model", ""),
                "embedding_model": health.get("embedding_model", ""),
                "api_base": health.get("api_base", ""),
                "api_key_set": health.get("api_key_set", False),
            }

    cfg = st.session_state.llm_config

    # Model presets for quick selection
    CHAT_PRESETS = [
        "(custom)",
        "qwen3-vl",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.0-flash",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "ollama/llama3",
    ]
    EMBED_PRESETS = [
        "(custom)",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "openai/text-embedding-3-small",
        "gemini/gemini-embedding-001",
        "ollama/nomic-embed-text",
    ]

    # Chat model selector
    current_chat = cfg.get("default_model", "")
    if current_chat in CHAT_PRESETS:
        chat_idx = CHAT_PRESETS.index(current_chat)
    else:
        chat_idx = 0  # (custom)

    chat_preset = st.selectbox("Chat Model", CHAT_PRESETS, index=chat_idx, key="chat_preset")
    if chat_preset == "(custom)":
        new_chat_model = st.text_input(
            "Custom chat model", value=current_chat, key="custom_chat_model"
        )
    else:
        new_chat_model = chat_preset

    # Embedding model selector
    current_embed = cfg.get("embedding_model", "")
    if current_embed in EMBED_PRESETS:
        embed_idx = EMBED_PRESETS.index(current_embed)
    else:
        embed_idx = 0  # (custom)

    embed_preset = st.selectbox(
        "Embedding Model", EMBED_PRESETS, index=embed_idx, key="embed_preset"
    )
    if embed_preset == "(custom)":
        new_embed_model = st.text_input(
            "Custom embedding model", value=current_embed, key="custom_embed_model"
        )
    else:
        new_embed_model = embed_preset

    # API base (for local / custom endpoints)
    new_api_base = st.text_input(
        "API Base URL",
        value=cfg.get("api_base", ""),
        help="Base URL for local / custom OpenAI-compatible endpoints. Leave empty for cloud providers.",
        key="llm_api_base",
    )

    # API key
    api_key_placeholder = "(set)" if cfg.get("api_key_set") else "(not set)"
    new_api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder=api_key_placeholder,
        help="Leave empty to keep current key. Enter a value to change it.",
        key="llm_api_key",
    )

    # Apply button
    if st.button("Apply Config", key="apply_llm_config"):
        # Only send fields that actually changed
        update_kwargs: Dict[str, Optional[str]] = {}
        if new_chat_model != current_chat:
            update_kwargs["default_model"] = new_chat_model
        if new_embed_model != current_embed:
            update_kwargs["embedding_model"] = new_embed_model
        if new_api_base != cfg.get("api_base", ""):
            update_kwargs["api_base"] = new_api_base
        if new_api_key:  # Only send if user typed something
            update_kwargs["api_key"] = new_api_key

        if update_kwargs:
            result = api_update_config(**update_kwargs)
            if result.get("error"):
                st.error(f"Config update failed: {result['error']}")
            elif result.get("changed"):
                st.success("Config updated. New queries will use the new settings.")
                # Refresh cached config
                st.session_state.llm_config = result.get("current", cfg)
                st.rerun()
            else:
                st.info("No changes detected.")
        else:
            st.info("No changes to apply.")

    # Show current active config
    st.caption(f"Active: {cfg.get('default_model', '?')} / {cfg.get('embedding_model', '?')}")

# Corpus info
with st.sidebar.expander("Corpus Info"):
    corpus = api_corpus()
    if corpus:
        st.write(f"**Files:** {corpus.get('total_files', 0)}")
        st.write(f"**Tokens:** {corpus.get('total_tokens', 0):,}")
        st.write(f"**Nodes:** {corpus.get('total_nodes', 0)}")
        st.write(f"**Embedded:** {corpus.get('files_with_embeddings', 0)}")
        file_names = corpus.get("file_names", [])
        if file_names:
            st.write("**Files:**")
            for f in file_names:
                st.write(f"  - {f}")
    else:
        st.write("Could not fetch corpus info.")

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.metadata = []
    if "llm_config" in st.session_state:
        del st.session_state["llm_config"]
    st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("LDRS v3 — Chat")

# Display existing messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and i < len(st.session_state.metadata):
            meta = st.session_state.metadata[i]
            if meta:
                with st.expander("Pipeline Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", meta.get("intent_type", "—"))
                        st.metric("Candidates", meta.get("candidates_count", 0))
                    with col2:
                        st.metric("Ranked", meta.get("ranked_count", 0))
                        st.metric("Claims Checked", meta.get("claims_checked", 0))
                    with col3:
                        st.metric("Time (ms)", f"{meta.get('total_time_ms', 0):.0f}")
                        st.metric("Flagged", meta.get("claims_flagged", 0))

                    # Selected files
                    selected = meta.get("selected_files", [])
                    if selected:
                        st.write("**Selected Files:**", ", ".join(selected))

                    # Citations
                    citations = meta.get("citations", [])
                    if citations:
                        st.write("**Citations:**")
                        for c in citations:
                            st.write(f"  - {c}")

                    # Usage
                    usage = meta.get("usage", {})
                    if usage:
                        with st.expander("Usage Stats"):
                            st.json(usage)

                    # Session
                    session_id = meta.get("session_id", "")
                    if session_id:
                        st.caption(f"Session: {session_id}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.metadata.append(None)  # placeholder for user messages

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation context
    conv_summary = ""
    recent_turns = []
    for msg in st.session_state.messages[-10:]:
        recent_turns.append({"role": msg["role"], "content": msg["content"][:500]})

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api_query(
                query=prompt,
                conversation_summary=conv_summary,
                recent_turns=recent_turns,
            )

        answer = result.get("answer", "No answer received.")
        st.markdown(answer)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Store metadata
        meta = {
            "intent_type": result.get("intent_type"),
            "selected_files": result.get("selected_files"),
            "candidates_count": result.get("candidates_count", 0),
            "ranked_count": result.get("ranked_count", 0),
            "claims_checked": result.get("claims_checked", 0),
            "claims_supported": result.get("claims_supported", 0),
            "claims_flagged": result.get("claims_flagged", 0),
            "re_grounded": result.get("re_grounded", False),
            "citations": result.get("citations"),
            "usage": result.get("usage"),
            "total_time_ms": result.get("total_time_ms", 0),
            "session_id": result.get("session_id", ""),
            "success": result.get("success", True),
            "error": result.get("error", ""),
        }
        st.session_state.metadata.append(meta)

        # Show error if any
        if not result.get("success", True):
            st.error(f"Error: {result.get('error', 'Unknown error')}")

        # Show pipeline details
        if result.get("success", True):
            with st.expander("Pipeline Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent", meta.get("intent_type", "—"))
                    st.metric("Candidates", meta.get("candidates_count", 0))
                with col2:
                    st.metric("Ranked", meta.get("ranked_count", 0))
                    st.metric("Claims Checked", meta.get("claims_checked", 0))
                with col3:
                    st.metric("Time (ms)", f"{meta.get('total_time_ms', 0):.0f}")
                    st.metric("Flagged", meta.get("claims_flagged", 0))

                selected = meta.get("selected_files", [])
                if selected:
                    st.write("**Selected Files:**", ", ".join(selected))

                citations = meta.get("citations", [])
                if citations:
                    st.write("**Citations:**")
                    for c in citations:
                        st.write(f"  - {c}")

                usage = meta.get("usage", {})
                if usage:
                    with st.expander("Usage Stats"):
                        st.json(usage)

                if meta.get("re_grounded"):
                    st.warning("This answer was re-grounded due to verification failures.")

                session_id = meta.get("session_id", "")
                if session_id:
                    st.caption(f"Session: {session_id}")
