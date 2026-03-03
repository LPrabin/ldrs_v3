"""
Monitoring — LangSmith integration and usage tracking.

Sets up LangSmith tracing for all LLM calls made through LangChain/deepagents.
Also provides a UsageTracker for aggregating per-query metrics:
  - Token usage (input/output/total)
  - Latency per stage
  - Estimated cost
  - Model name

Usage::

    from agent.config import AgentConfig
    from agent.monitoring import setup_monitoring, UsageTracker

    config = AgentConfig()
    setup_monitoring(config)

    tracker = UsageTracker()
    tracker.record_llm_call(
        stage="intent_classifier",
        model="qwen3-vl",
        input_tokens=150,
        output_tokens=200,
        latency_ms=340.5,
    )
    print(tracker.summary())
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_monitoring(config: "AgentConfig") -> None:
    """
    Configure LangSmith tracing from AgentConfig.

    Sets environment variables that LangSmith/LangChain read automatically.
    Must be called before any LangChain model is instantiated.

    Args:
        config: AgentConfig instance with LangSmith settings.
    """
    if config.langsmith_tracing and config.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
        logger.info("LangSmith tracing enabled  project=%s", config.langsmith_project)
    else:
        os.environ.pop("LANGSMITH_TRACING", None)
        logger.info("LangSmith tracing disabled")


@dataclass
class LLMCallRecord:
    """A single LLM API call record."""

    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTimer:
    """Timer for a pipeline stage."""

    stage: str
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class UsageTracker:
    """
    Aggregates usage metrics across a single query's lifecycle.

    Thread-safe accumulator for token counts, latencies, and costs
    across all pipeline stages. Produces a summary dict suitable for
    logging, API responses, or LangSmith metadata.
    """

    def __init__(self):
        self.calls: List[LLMCallRecord] = []
        self.stage_timers: Dict[str, StageTimer] = {}
        self._query_start: float = 0.0
        self._query_end: float = 0.0

    def start_query(self) -> None:
        """Mark the start of a query."""
        self._query_start = time.time()
        logger.info("UsageTracker.start_query  timestamp=%.3f", self._query_start)

    def end_query(self) -> None:
        """Mark the end of a query."""
        self._query_end = time.time()
        elapsed_ms = (self._query_end - self._query_start) * 1000 if self._query_start > 0 else 0.0
        logger.info(
            "UsageTracker.end_query  elapsed=%.1fms  llm_calls=%d",
            elapsed_ms,
            len(self.calls),
        )

    def start_stage(self, stage: str) -> None:
        """Start timing a pipeline stage."""
        self.stage_timers[stage] = StageTimer(stage=stage, start_time=time.time())
        logger.debug("UsageTracker.start_stage  stage=%s", stage)

    def end_stage(self, stage: str) -> None:
        """Stop timing a pipeline stage."""
        if stage in self.stage_timers:
            self.stage_timers[stage].end_time = time.time()
            logger.debug(
                "UsageTracker.end_stage  stage=%s  elapsed=%.1fms",
                stage,
                self.stage_timers[stage].elapsed_ms,
            )

    def record_llm_call(
        self,
        stage: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        **metadata,
    ) -> None:
        """Record a single LLM API call."""
        record = LLMCallRecord(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata=metadata,
        )
        self.calls.append(record)
        logger.debug(
            "UsageTracker.record  stage=%s  model=%s  in=%d  out=%d  latency=%.1fms",
            stage,
            model,
            input_tokens,
            output_tokens,
            latency_ms,
        )

    def summary(self) -> Dict[str, Any]:
        """
        Produce an aggregate summary of all recorded usage.

        Returns:
            Dict with total_tokens, total_cost, per_stage breakdowns,
            total_query_time, and individual call records.
        """
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost_usd for c in self.calls)

        stage_breakdown: Dict[str, Dict[str, Any]] = {}
        for call in self.calls:
            if call.stage not in stage_breakdown:
                stage_breakdown[call.stage] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "llm_calls": 0,
                    "total_latency_ms": 0.0,
                }
            sb = stage_breakdown[call.stage]
            sb["input_tokens"] += call.input_tokens
            sb["output_tokens"] += call.output_tokens
            sb["total_tokens"] += call.input_tokens + call.output_tokens
            sb["cost_usd"] += call.cost_usd
            sb["llm_calls"] += 1
            sb["total_latency_ms"] += call.latency_ms

        stage_timings = {
            name: round(timer.elapsed_ms, 2) for name, timer in self.stage_timers.items()
        }

        query_time_ms = 0.0
        if self._query_start > 0 and self._query_end > 0:
            query_time_ms = (self._query_end - self._query_start) * 1000

        result = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "total_llm_calls": len(self.calls),
            "total_query_time_ms": round(query_time_ms, 2),
            "stage_breakdown": stage_breakdown,
            "stage_timings_ms": stage_timings,
        }
        logger.info(
            "UsageTracker.summary  total_tokens=%d  total_cost=$%.6f  "
            "llm_calls=%d  query_time=%.1fms  stages=%s",
            total_input + total_output,
            total_cost,
            len(self.calls),
            query_time_ms,
            list(stage_timings.keys()),
        )
        return result
