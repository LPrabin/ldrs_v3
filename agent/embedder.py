"""
Embedder — Section-level embedding pipeline using pgvector.

Embeds document sections via LiteLLM's embedding API (provider-agnostic)
and stores them in PostgreSQL with pgvector.

Usage::

    config = AgentConfig()
    embedder = Embedder(config)
    await embedder.connect()

    # Embed and store sections for a document
    await embedder.embed_document("auth.md", sections)

    # Query similar sections
    results = await embedder.search("OAuth2 token refresh", top_k=10)

    await embedder.close()
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import litellm

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """A single embedding search result."""

    doc_name: str
    section_id: str
    section_title: str
    content: str
    similarity: float
    source_file: str
    line_num: int = 0


class Embedder:
    """
    Section-level embedding pipeline backed by PostgreSQL + pgvector.

    Uses LiteLLM for provider-agnostic embedding calls (OpenAI, Gemini,
    Ollama, or any custom endpoint). Stores embeddings in the ``sections``
    table and supports cosine similarity search scoped to specific documents.

    Args:
        config: AgentConfig with database and embedding settings.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._embedding_kwargs = config.litellm_embedding_kwargs
        self._model = config.embedding_model
        self._dim = config.embedding_dim

    async def connect(self) -> None:
        """Create a connection pool to PostgreSQL."""
        logger.debug(
            "Embedder.connect  host=%s  min_size=2  max_size=10",
            self.config.postgres_host,
        )
        self._pool = await asyncpg.create_pool(
            self.config.async_postgres_dsn,
            min_size=2,
            max_size=10,
        )
        logger.info("Embedder connected to PostgreSQL  dsn=%s", self.config.postgres_host)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Embedder disconnected from PostgreSQL")

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text string.

        Args:
            text: Text to embed.

        Returns:
            List of floats (embedding vector).
        """
        logger.debug("Embedder._get_embedding  text_len=%d  model=%s", len(text), self._model)
        # Log what kwargs are being passed for debugging
        logger.info("Embedder._get_embedding  kwargs keys: %s", list(self._embedding_kwargs.keys()))
        if "api_key" in self._embedding_kwargs:
            logger.info("Embedder._get_embedding  api_key present, first 10 chars: %s", self._embedding_kwargs["api_key"][:10])
        else:
            logger.info("Embedder._get_embedding  NO api_key in kwargs - will use env var")
        response = await litellm.aembedding(
            **self._embedding_kwargs,
            input=[text],
        )
        logger.debug("Embedder._get_embedding  dim=%d", len(response.data[0]["embedding"]))
        return response.data[0]["embedding"]

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(
            "Embedder._get_embeddings_batch  texts=%d  model=%s",
            len(texts),
            self._model,
        )
        
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = await litellm.aembedding(
                **self._embedding_kwargs,
                input=batch_texts,
            )
            # Sort by index to maintain order within the batch
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])

        logger.debug(
            "Embedder._get_embeddings_batch  done  returned=%d  dim=%d",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    async def embed_document(
        self,
        doc_name: str,
        sections: List[Dict[str, Any]],
        source_file: str = "",
    ) -> int:
        """
        Embed and store all sections for a document.

        Existing sections for this doc_name are replaced (upsert).

        Args:
            doc_name:    Document name (used as key).
            sections:    List of section dicts with keys:
                         node_id, title, text, line_num (optional).
            source_file: Path to the source .md file.

        Returns:
            Number of sections embedded.
        """
        if not self._pool:
            raise RuntimeError("Embedder not connected. Call connect() first.")

        if not sections:
            return 0

        # Prepare texts for embedding
        texts = []
        section_data = []
        for s in sections:
            text = s.get("text", "")
            if not text.strip():
                logger.debug(
                    "Embedder.embed_document  skipping empty section  node_id=%s",
                    s.get("node_id", ""),
                )
                continue
            title = s.get("title", "")
            # Prepend title for better embedding quality
            embed_text = f"{title}\n\n{text}" if title else text
            texts.append(embed_text)
            section_data.append(
                {
                    "section_id": s.get("node_id", ""),
                    "section_title": title,
                    "content": text,
                    "line_num": s.get("line_num", 0),
                }
            )
            logger.debug(
                "Embedder.embed_document  section  id=%s  title=%r  text_len=%d",
                s.get("node_id", ""),
                title[:60],
                len(text),
            )

        if not texts:
            return 0

        # Get embeddings in batch
        logger.info(
            "Embedder.embed_document  doc=%s  sections=%d",
            doc_name,
            len(texts),
        )
        embeddings = await self._get_embeddings_batch(texts)

        # Upsert into database
        async with self._pool.acquire() as conn:
            # Delete existing sections for this doc
            await conn.execute(
                "DELETE FROM sections WHERE doc_name = $1",
                doc_name,
            )

            # Insert new sections
            for i, (data, embedding) in enumerate(zip(section_data, embeddings)):
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                await conn.execute(
                    """
                    INSERT INTO sections
                        (doc_name, section_id, section_title, source_file,
                         content, line_num, embedding, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                    ON CONFLICT (doc_name, section_id)
                    DO UPDATE SET
                        section_title = EXCLUDED.section_title,
                        source_file = EXCLUDED.source_file,
                        content = EXCLUDED.content,
                        line_num = EXCLUDED.line_num,
                        embedding = EXCLUDED.embedding,
                        token_count = EXCLUDED.token_count,
                        updated_at = NOW()
                    """,
                    doc_name,
                    data["section_id"],
                    data["section_title"],
                    source_file,
                    data["content"],
                    data["line_num"],
                    embedding_str,
                    len(texts[i].split()),  # approximate token count
                )

        logger.info(
            "Embedder.embed_document  done  doc=%s  stored=%d",
            doc_name,
            len(section_data),
        )
        return len(section_data)

    async def remove_document(self, doc_name: str) -> int:
        """Remove all embeddings for a document. Returns count removed."""
        if not self._pool:
            raise RuntimeError("Embedder not connected. Call connect() first.")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sections WHERE doc_name = $1",
                doc_name,
            )
            count = int(result.split()[-1])
            logger.info("Embedder.remove_document  doc=%s  removed=%d", doc_name, count)
            return count

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        doc_names: Optional[List[str]] = None,
    ) -> List[EmbeddingResult]:
        """
        Search for similar sections using cosine similarity.

        Args:
            query:     Natural language query to embed and search.
            top_k:     Maximum number of results.
            doc_names: Optional list of doc_names to scope the search.

        Returns:
            List of EmbeddingResult sorted by similarity (highest first).
        """
        if not self._pool:
            raise RuntimeError("Embedder not connected. Call connect() first.")

        query_embedding = await self._get_embedding(query)
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        async with self._pool.acquire() as conn:
            if doc_names:
                rows = await conn.fetch(
                    """
                    SELECT doc_name, section_id, section_title, content,
                           source_file, line_num,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM sections
                    WHERE doc_name = ANY($2)
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    embedding_str,
                    doc_names,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT doc_name, section_id, section_title, content,
                           source_file, line_num,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM sections
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_str,
                    top_k,
                )

        results = [
            EmbeddingResult(
                doc_name=row["doc_name"],
                section_id=row["section_id"],
                section_title=row["section_title"],
                content=row["content"],
                similarity=float(row["similarity"]),
                source_file=row["source_file"],
                line_num=row["line_num"] or 0,
            )
            for row in rows
        ]

        for r in results:
            logger.debug(
                "Embedder.search  result  doc=%s  section=%s  sim=%.4f  title=%r",
                r.doc_name,
                r.section_id,
                r.similarity,
                r.section_title[:50],
            )

        logger.info(
            "Embedder.search  query=%r  top_k=%d  results=%d",
            query[:50],
            top_k,
            len(results),
        )
        return results

    async def search_multi(
        self,
        queries: List[str],
        *,
        top_k_per_query: int = 5,
        doc_names: Optional[List[str]] = None,
    ) -> List[EmbeddingResult]:
        """
        Search for multiple queries and merge results, deduplicating by section_id.

        Args:
            queries:         List of query strings.
            top_k_per_query: Max results per query.
            doc_names:       Optional scope filter.

        Returns:
            Deduplicated, merged list of EmbeddingResult.
        """
        tasks = [self.search(q, top_k=top_k_per_query, doc_names=doc_names) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # Deduplicate by (doc_name, section_id), keeping highest similarity
        best: Dict[Tuple[str, str], EmbeddingResult] = {}
        for result_list in all_results:
            for r in result_list:
                key = (r.doc_name, r.section_id)
                if key not in best or r.similarity > best[key].similarity:
                    best[key] = r

        merged = sorted(best.values(), key=lambda r: -r.similarity)
        logger.info(
            "Embedder.search_multi  queries=%d  merged=%d",
            len(queries),
            len(merged),
        )
        return merged
