"""
Intent Classifier — Stage 1 of the Hybrid RAG Deep Agent pipeline.

Single LLM call that takes the user query + registry JSON and outputs:
  - intent_type: exact|conceptual|comparative|multihop|db_query|hybrid
  - selected_files: [{path, confidence}]
  - query_variants: rephrased queries for parallel retrieval
  - pattern_hints: {literals, phrases, prefix_wildcards} for TreeGrep
  - needs_db: boolean
  - likely_multihop: boolean

Usage::

    config = AgentConfig()
    classifier = IntentClassifier(config)
    result = await classifier.classify(
        query="How does OAuth2 token refresh work?",
        registry=registry.get_for_llm(),
    )
    print(result.intent_type)
    print(result.selected_files)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm

from agent.config import AgentConfig
from agent.monitoring import UsageTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the Intent Classifier
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a retrieval routing engine. Your job is to analyze the user's query
and the document registry, then output a strict JSON object that tells the
retrieval pipeline:

1. What type of query this is (intent_type)
2. Which files are most likely to contain the answer (selected_files)
3. Rephrased variants of the query for diverse retrieval (query_variants)
4. Pattern hints for literal/phrase grep matching (pattern_hints)
5. Whether database access is needed (needs_db)
6. Whether this is a multi-hop reasoning question (likely_multihop)

## Rules

- Output ONLY valid JSON. No markdown fences, no explanation.
- intent_type must be one of: exact, conceptual, comparative, multihop, db_query, hybrid
- selected_files: include only files likely relevant. Assign confidence 0.0-1.0.
  If the registry is empty, return an empty list.
- query_variants: 2-4 rephrased queries that capture different angles of the question.
  Include the original query as the first variant.
- pattern_hints:
  - literals: exact strings to grep for (names, codes, identifiers)
  - phrases: multi-word phrases to match as substrings
  - prefix_wildcards: word prefixes to match (e.g., "auth*" matches "authentication")
- If no files seem relevant at all, still classify the intent and return empty selected_files.
- Do NOT write raw regex. Only provide literals, phrases, and prefix_wildcards.

## Output Schema

{
  "intent_type": "exact|conceptual|comparative|multihop|db_query|hybrid",
  "selected_files": [
    {"path": "filename.md", "confidence": 0.92}
  ],
  "query_variants": [
    "original query",
    "rephrased variant 1",
    "rephrased variant 2"
  ],
  "pattern_hints": {
    "literals": ["exact_string_1", "exact_string_2"],
    "phrases": ["multi word phrase"],
    "prefix_wildcards": ["prefix*"]
  },
  "needs_db": false,
  "likely_multihop": false
}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SelectedFile:
    """A file selected by the Intent Classifier."""

    path: str
    confidence: float = 0.0


@dataclass
class PatternHints:
    """Pattern hints for TreeGrep from the Intent Classifier."""

    literals: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    prefix_wildcards: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "literals": self.literals,
            "phrases": self.phrases,
            "prefix_wildcards": self.prefix_wildcards,
        }


@dataclass
class IntentResult:
    """Output of the Intent Classifier (Stage 1)."""

    intent_type: str = "conceptual"
    selected_files: List[SelectedFile] = field(default_factory=list)
    query_variants: List[str] = field(default_factory=list)
    pattern_hints: PatternHints = field(default_factory=PatternHints)
    needs_db: bool = False
    likely_multihop: bool = False
    raw_response: str = ""

    @property
    def selected_paths(self) -> List[str]:
        """Get just the file paths from selected_files."""
        return [f.path for f in self.selected_files]


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """
    Stage 1: Intent Classification + Routing.

    Makes a single LLM call with the user query and registry JSON to
    classify intent and select relevant files for retrieval.

    Args:
        config: AgentConfig with LLM settings.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._chat_kwargs = config.litellm_chat_kwargs
        self._model = config.default_model

    async def classify(
        self,
        query: str,
        registry: Dict[str, Any],
        conversation_summary: str = "",
        tracker: Optional[UsageTracker] = None,
    ) -> IntentResult:
        """
        Classify a user query and select relevant files.

        Args:
            query:                The user's natural language query.
            registry:             Compact registry from ``Registry.get_for_llm()``.
            conversation_summary: Optional summary of recent conversation turns.
            tracker:              Optional UsageTracker for monitoring.

        Returns:
            IntentResult with classified intent and routing decisions.
        """
        if tracker:
            tracker.start_stage("intent_classifier")

        logger.info("IntentClassifier.classify  query=%r", query[:100])

        # Handle empty registry
        files = registry.get("files", {})
        if not files:
            logger.info("IntentClassifier.classify  empty registry, returning defaults")
            result = IntentResult(
                intent_type="conceptual",
                query_variants=[query],
                raw_response="{}",
            )
            if tracker:
                tracker.end_stage("intent_classifier")
            return result

        # Build user message
        user_parts = []
        user_parts.append(f"## User Query\n{query}")

        if conversation_summary:
            user_parts.append(f"## Conversation History\n{conversation_summary}")

        user_parts.append(
            f"## Document Registry\n```json\n{json.dumps(registry, indent=2, ensure_ascii=False)}\n```"
        )

        user_message = "\n\n".join(user_parts)

        # Make LLM call
        start_time = time.time()
        try:
            response = await litellm.acompletion(
                **self._chat_kwargs,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )

            raw = response.choices[0].message.content or "{}"
            latency_ms = (time.time() - start_time) * 1000

            # Record usage
            usage = response.usage
            if tracker and usage:
                tracker.record_llm_call(
                    stage="intent_classifier",
                    model=self._model,
                    input_tokens=usage.prompt_tokens or 0,
                    output_tokens=usage.completion_tokens or 0,
                    latency_ms=latency_ms,
                )
                logger.debug(
                    "IntentClassifier  llm_call  model=%s  input_tokens=%d  "
                    "output_tokens=%d  latency_ms=%.0f",
                    self._model,
                    usage.prompt_tokens or 0,
                    usage.completion_tokens or 0,
                    latency_ms,
                )

            logger.debug("IntentClassifier  raw_response=%s", raw[:500])

            # Parse response
            result = self._parse_response(raw, query)

            if tracker:
                tracker.end_stage("intent_classifier")

            return result

        except Exception as e:
            logger.error("IntentClassifier.classify  error=%s", e)
            if tracker:
                tracker.end_stage("intent_classifier")
            # Return safe defaults on failure
            return IntentResult(
                intent_type="conceptual",
                query_variants=[query],
                raw_response=str(e),
            )

    def _parse_response(self, raw: str, original_query: str) -> IntentResult:
        """
        Parse the LLM's JSON response into an IntentResult.

        Handles edge cases: malformed JSON, missing fields, invalid values.
        Falls back to safe defaults.

        Args:
            raw:            The raw LLM response string.
            original_query: The original user query (for defaults).

        Returns:
            IntentResult with validated fields.
        """
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last line (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("IntentClassifier._parse_response  JSON parse failed: %s", e)
            return IntentResult(
                intent_type="conceptual",
                query_variants=[original_query],
                raw_response=raw,
            )

        # Validate intent_type
        valid_intents = {"exact", "conceptual", "comparative", "multihop", "db_query", "hybrid"}
        intent_type = data.get("intent_type", "conceptual")
        if intent_type not in valid_intents:
            intent_type = "conceptual"

        # Parse selected_files
        selected_files = []
        for f in data.get("selected_files", []):
            if isinstance(f, dict) and "path" in f:
                sf = SelectedFile(
                    path=f["path"],
                    confidence=float(f.get("confidence", 0.5)),
                )
                selected_files.append(sf)
                logger.debug(
                    "IntentClassifier._parse_response  selected file  path=%s  confidence=%.2f",
                    sf.path,
                    sf.confidence,
                )

        # Parse query_variants
        query_variants = data.get("query_variants", [])
        if not query_variants:
            query_variants = [original_query]
        # Ensure original query is included
        if original_query not in query_variants:
            query_variants.insert(0, original_query)

        # Parse pattern_hints
        hints_data = data.get("pattern_hints", {})
        pattern_hints = PatternHints(
            literals=hints_data.get("literals", []),
            phrases=hints_data.get("phrases", []),
            prefix_wildcards=hints_data.get("prefix_wildcards", []),
        )

        result = IntentResult(
            intent_type=intent_type,
            selected_files=selected_files,
            query_variants=query_variants,
            pattern_hints=pattern_hints,
            needs_db=bool(data.get("needs_db", False)),
            likely_multihop=bool(data.get("likely_multihop", False)),
            raw_response=raw,
        )

        logger.info(
            "IntentClassifier  parsed  intent=%s  files=%d  variants=%d  "
            "hints=%d/%d/%d  needs_db=%s  multihop=%s",
            result.intent_type,
            len(result.selected_files),
            len(result.query_variants),
            len(pattern_hints.literals),
            len(pattern_hints.phrases),
            len(pattern_hints.prefix_wildcards),
            result.needs_db,
            result.likely_multihop,
        )
        return result
