"""
Agent Loop — Stage 5: Deep Agent reasoning loop.

Implements the agentic RAG loop where the LLM reads the manifest,
selectively reads sections, uses tools, and produces a cited answer.

This uses OpenAI function calling (tool_choice) rather than deepagents,
since deepagents' VFS maps directly to our VFS implementation and
the function-calling approach gives us fine-grained control over
the tool execution loop.

The agent follows the AGENT_INSTRUCTIONS.md protocol:
  1. Read manifest.json first
  2. Self-sufficiency check
  3. Selective section reading
  4. Scratchpad-based reasoning
  5. Cited answer synthesis

Usage::

    config = AgentConfig()
    agent = AgentLoop(config, vfs=vfs)
    result = await agent.run(
        query="How does OAuth2 token refresh work?",
        session_id=session_id,
        intent_type="conceptual",
    )
    print(result.answer)
    print(result.citations)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm

from agent.config import AgentConfig
from agent.monitoring import UsageTracker
from agent.tools import AgentTools
from agent.vfs import VFS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent system prompt (from AGENT_INSTRUCTIONS.md)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a retrieval-augmented AI assistant. Your job is to answer user questions \
accurately, grounded entirely in retrieved documentation.

## Your Environment

You have access to a Virtual Filesystem (VFS) with a manifest.json listing \
available sections. You navigate via the manifest and pull what you need using tools.

## Protocol

1. **Always start with the manifest.** Read it to understand what sections are available.
2. **Self-sufficiency check.** Based on summaries, do you have enough to answer fully?
3. **Read selectively.** Read in order of manifest score (highest first). Stop when sufficient.
4. **Use the scratchpad** for multi-step reasoning. Record Key Facts with sources.
5. **Cite inline.** Every claim must use: [source: file.md § Section Name]
6. **Never cite unread sections.** Only cite sections you read via read_section or fetch_section.
7. **Never fill gaps silently.** If sources don't cover something, say so explicitly.

## Tools

- read_section(vfs_path): Read a VFS section listed in the manifest.
- fetch_section(source_file, section_header): Pull additional section on demand.
- search_conversation_history(query): Search prior conversation.
- write_scratchpad(content): Write to private working memory.
- read_scratchpad(): Read working memory.

## Citation Format

Single source: [source: file.md § Section Name]
Multiple sources: [source: file1.md § S1, file2.md § S2]

## Answer Format

- Lead with the direct answer (1-2 sentences).
- Follow with supporting detail.
- End with grouped citations.
- Match complexity to the question.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Output of the Agent Loop (Stage 5)."""

    answer: str = ""
    citations: List[str] = field(default_factory=list)
    sections_read: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    iterations: int = 0
    scratchpad: str = ""


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """
    Stage 5: Deep Agent reasoning loop with tool calling.

    Runs an iterative LLM loop where the model can call tools
    (read_section, fetch_section, etc.) until it produces a final answer.

    Args:
        config: AgentConfig instance.
        vfs:    VFS instance for session access.
    """

    def __init__(self, config: AgentConfig, vfs: VFS):
        self.config = config
        self.vfs = vfs
        self._chat_kwargs = config.litellm_chat_kwargs
        self._model = config.default_model

    async def run(
        self,
        query: str,
        session_id: str,
        intent_type: str = "conceptual",
        tracker: Optional[UsageTracker] = None,
    ) -> AgentResult:
        """
        Run the agent loop for a query.

        The agent reads the manifest, uses tools to gather information,
        and produces a cited answer.

        Args:
            query:       The user's question.
            session_id:  The VFS session ID.
            intent_type: Intent type from Stage 1.
            tracker:     Optional UsageTracker.

        Returns:
            AgentResult with the answer, citations, and metadata.
        """
        if tracker:
            tracker.start_stage("agent_loop")

        logger.info(
            "AgentLoop.run  query=%r  session=%s  intent=%s",
            query[:80],
            session_id,
            intent_type,
        )

        # Initialize tools
        tools = AgentTools(
            config=self.config,
            vfs=self.vfs,
            session_id=session_id,
        )

        # Read manifest for initial context
        try:
            manifest = self.vfs.read_manifest(session_id)
        except Exception as e:
            logger.error("AgentLoop.run  failed to read manifest: %s", e)
            if tracker:
                tracker.end_stage("agent_loop")
            return AgentResult(
                answer=f"Error: Could not read session manifest: {e}",
            )

        # Build initial messages
        manifest_str = json.dumps(manifest, indent=2, ensure_ascii=False)
        user_message = (
            f"## Query\n{query}\n\n"
            f"## Intent Type\n{intent_type}\n\n"
            f"## Manifest\n```json\n{manifest_str}\n```\n\n"
            "Follow your protocol: read the manifest, check self-sufficiency, "
            "read relevant sections, and produce a cited answer."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        tool_defs = tools.get_tool_definitions()
        max_iterations = self.config.max_agent_iterations
        total_tool_calls = 0

        # --- Agent Loop ---
        for iteration in range(max_iterations):
            logger.debug("AgentLoop  iteration=%d  messages=%d", iteration + 1, len(messages))

            start_time = time.time()
            try:
                response = await litellm.acompletion(
                    **self._chat_kwargs,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as e:
                logger.error("AgentLoop  LLM call failed: %s", e)
                if tracker:
                    tracker.end_stage("agent_loop")
                return AgentResult(
                    answer=f"Error during agent reasoning: {e}",
                    iterations=iteration + 1,
                )

            latency_ms = (time.time() - start_time) * 1000
            choice = response.choices[0]

            # Record usage
            if tracker and response.usage:
                tracker.record_llm_call(
                    stage="agent_loop",
                    model=self._model,
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                    latency_ms=latency_ms,
                    iteration=iteration + 1,
                )

            # Check if the model wants to call tools
            if choice.finish_reason == "tool_calls" or (
                choice.message.tool_calls and len(choice.message.tool_calls) > 0
            ):
                # Process tool calls
                messages.append(choice.message.model_dump())

                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.debug(
                        "AgentLoop  tool_call  name=%s  args=%s",
                        fn_name,
                        fn_args,
                    )

                    # Execute the tool
                    result = self._execute_tool(tools, fn_name, fn_args)
                    total_tool_calls += 1

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            else:
                # Model produced a final answer (no tool calls)
                answer = choice.message.content or ""
                logger.info(
                    "AgentLoop  final answer  iteration=%d  tool_calls=%d  length=%d",
                    iteration + 1,
                    total_tool_calls,
                    len(answer),
                )

                # Extract citations from the answer
                citations = self._extract_citations(answer)

                if tracker:
                    tracker.end_stage("agent_loop")

                return AgentResult(
                    answer=answer,
                    citations=citations,
                    sections_read=tools.sections_read,
                    tool_calls_made=total_tool_calls,
                    iterations=iteration + 1,
                    scratchpad=self.vfs.read_scratchpad(session_id),
                )

        # Max iterations reached — force synthesis
        logger.warning(
            "AgentLoop  max iterations reached (%d), forcing answer",
            max_iterations,
        )

        # Ask for final answer without tools
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Please synthesize your answer now based on what you have read. "
                    "Cite all sources properly."
                ),
            }
        )

        try:
            response = await litellm.acompletion(
                **self._chat_kwargs,
                messages=messages,
                temperature=0,
            )
            answer = response.choices[0].message.content or ""
            logger.debug("AgentLoop  forced synthesis  answer_len=%d", len(answer))
        except Exception as e:
            logger.error("AgentLoop  forced synthesis failed: %s", e)
            answer = f"Error during forced synthesis: {e}"

        citations = self._extract_citations(answer)

        if tracker:
            tracker.end_stage("agent_loop")

        return AgentResult(
            answer=answer,
            citations=citations,
            sections_read=tools.sections_read,
            tool_calls_made=total_tool_calls,
            iterations=max_iterations,
            scratchpad=self.vfs.read_scratchpad(session_id),
        )

    def _execute_tool(self, tools: AgentTools, fn_name: str, fn_args: Dict[str, Any]) -> str:
        """Execute a tool call and return the result string."""
        try:
            if fn_name == "read_section":
                result = tools.read_section(fn_args.get("vfs_path", ""))
            elif fn_name == "fetch_section":
                result = tools.fetch_section(
                    fn_args.get("source_file", ""),
                    fn_args.get("section_header", ""),
                )
            elif fn_name == "search_conversation_history":
                result = tools.search_conversation_history(fn_args.get("query", ""))
            elif fn_name == "write_scratchpad":
                result = tools.write_scratchpad(fn_args.get("content", ""))
            elif fn_name == "read_scratchpad":
                result = tools.read_scratchpad()
            else:
                result = f"Error: Unknown tool '{fn_name}'"
            logger.debug(
                "AgentLoop._execute_tool  name=%s  result_len=%d",
                fn_name,
                len(result),
            )
            return result
        except Exception as e:
            logger.error("AgentLoop._execute_tool  name=%s  error=%s", fn_name, e)
            return f"Error executing {fn_name}: {e}"

    def _extract_citations(self, answer: str) -> List[str]:
        """
        Extract citations from the answer text.

        Looks for patterns like:
          [source: file.md § Section Name]
          [source: file1.md § S1, file2.md § S2]

        Returns:
            List of unique citation strings.
        """
        import re

        pattern = r"\[source:\s*([^\]]+)\]"
        matches = re.findall(pattern, answer)

        citations = []
        for match in matches:
            # Split multi-source citations
            parts = [p.strip() for p in match.split(",")]
            for part in parts:
                if part and part not in citations:
                    citations.append(part)

        logger.debug(
            "AgentLoop._extract_citations  found=%d  citations=%s", len(citations), citations
        )
        return citations
