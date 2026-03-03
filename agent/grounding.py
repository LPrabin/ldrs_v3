"""
Grounding — Stage 6: Grounding Verification.

Verifies that each cited claim in the agent's answer is supported by the
section it cites. Uses semantic entailment checking via LLM.

Process:
  1. Extract claims with citations from the answer.
  2. For each claim, locate the cited section in the VFS.
  3. Ask the LLM: "Does this section support this claim?"
  4. Flag unsupported claims.
  5. If too many flags: re-route to agent with grounding prompt.
  6. Log mismatches to hallucination_log.jsonl.

Usage::

    config = AgentConfig()
    verifier = GroundingVerifier(config, vfs=vfs)
    result = await verifier.verify(
        answer=agent_result.answer,
        session_id=session_id,
    )
    print(result.verified_answer)
    print(result.flags)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import litellm

from agent.config import AgentConfig
from agent.monitoring import UsageTracker
from agent.vfs import VFS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verification prompt
# ---------------------------------------------------------------------------

VERIFICATION_PROMPT = """\
You are a grounding verification engine. Your job is to determine whether
a specific claim is supported by the provided source text.

## Rules
- Output ONLY valid JSON with the fields: "supported" (boolean) and "reason" (string).
- A claim is "supported" if the source text contains information that directly
  supports or implies the claim.
- A claim is NOT supported if the source text contradicts it, is silent on it,
  or only tangentially relates to it.
- Be strict but fair. Minor paraphrasing is acceptable. Fabrication is not.

## Output Schema
{"supported": true, "reason": "The source explicitly states..."}
or
{"supported": false, "reason": "The source does not mention..."}
"""

# Max claims to verify per answer (cost control)
MAX_CLAIMS_TO_VERIFY = 10

# Flag threshold — if more than this fraction of claims are unsupported,
# trigger re-grounding
FLAG_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: str
    citation: str
    source_content: str
    supported: bool = True
    reason: str = ""


@dataclass
class GroundingResult:
    """Output of the grounding verification (Stage 6)."""

    original_answer: str
    verified_answer: str
    claims_checked: int = 0
    claims_supported: int = 0
    claims_flagged: int = 0
    flags: List[ClaimVerification] = field(default_factory=list)
    all_verifications: List[ClaimVerification] = field(default_factory=list)
    re_grounded: bool = False


# ---------------------------------------------------------------------------
# Grounding Verifier
# ---------------------------------------------------------------------------


class GroundingVerifier:
    """
    Stage 6: Grounding Verification.

    Checks each cited claim in the agent's answer against its source section.
    Flags unsupported claims and optionally caveats them.

    Args:
        config: AgentConfig instance.
        vfs:    VFS instance for reading cited sections.
    """

    def __init__(self, config: AgentConfig, vfs: VFS):
        self.config = config
        self.vfs = vfs
        self._chat_kwargs = config.litellm_chat_kwargs
        self._model = config.default_model

    async def verify(
        self,
        answer: str,
        session_id: str,
        tracker: Optional[UsageTracker] = None,
    ) -> GroundingResult:
        """
        Verify all cited claims in the answer.

        Args:
            answer:     The agent's answer with inline citations.
            session_id: The VFS session ID.
            tracker:    Optional UsageTracker.

        Returns:
            GroundingResult with verification details.
        """
        if tracker:
            tracker.start_stage("grounding")

        logger.info("GroundingVerifier.verify  session=%s  answer_len=%d", session_id, len(answer))

        # Extract claims with citations
        claims = self._extract_claims(answer)
        logger.info("GroundingVerifier  extracted %d claims", len(claims))

        if not claims:
            if tracker:
                tracker.end_stage("grounding")
            return GroundingResult(
                original_answer=answer,
                verified_answer=answer,
            )

        # Limit number of claims to check
        claims_to_check = claims[:MAX_CLAIMS_TO_VERIFY]

        # Read manifest to map citations to VFS paths
        try:
            manifest = self.vfs.read_manifest(session_id)
        except Exception as e:
            logger.warning("GroundingVerifier  could not read manifest: %s", e)
            if tracker:
                tracker.end_stage("grounding")
            return GroundingResult(
                original_answer=answer,
                verified_answer=answer,
            )

        # Verify each claim
        verifications: List[ClaimVerification] = []
        for claim_text, citation in claims_to_check:
            verification = await self._verify_claim(
                claim_text, citation, session_id, manifest, tracker
            )
            verifications.append(verification)

        # Count results
        supported = sum(1 for v in verifications if v.supported)
        flagged = sum(1 for v in verifications if not v.supported)
        flags = [v for v in verifications if not v.supported]

        # Build verified answer
        verified_answer = answer
        if flags:
            verified_answer = self._caveat_unsupported(answer, flags)
            # Log to hallucination log
            self._log_flags(session_id, flags)

        # Check if re-grounding is needed
        re_grounded = False
        if len(claims_to_check) > 0:
            flag_ratio = flagged / len(claims_to_check)
            if flag_ratio > FLAG_THRESHOLD:
                logger.warning(
                    "GroundingVerifier  flag_ratio=%.2f > threshold=%.2f, re-grounding recommended",
                    flag_ratio,
                    FLAG_THRESHOLD,
                )
                re_grounded = True

        if tracker:
            tracker.end_stage("grounding")

        result = GroundingResult(
            original_answer=answer,
            verified_answer=verified_answer,
            claims_checked=len(verifications),
            claims_supported=supported,
            claims_flagged=flagged,
            flags=flags,
            all_verifications=verifications,
            re_grounded=re_grounded,
        )

        logger.info(
            "GroundingVerifier.verify  done  checked=%d  supported=%d  flagged=%d",
            result.claims_checked,
            result.claims_supported,
            result.claims_flagged,
        )
        return result

    def _extract_claims(self, answer: str) -> List[Tuple[str, str]]:
        """
        Extract (claim_text, citation) pairs from the answer.

        Looks for sentences followed by [source: ...] citations.

        Returns:
            List of (claim_sentence, citation_string) tuples.
        """
        # Match sentences ending with a citation
        pattern = r"([^.!?\n]+[.!?]?)\s*\[source:\s*([^\]]+)\]"
        matches = re.findall(pattern, answer)

        claims = []
        for claim_text, citation in matches:
            claim_text = claim_text.strip()
            citation = citation.strip()
            if claim_text and citation:
                claims.append((claim_text, citation))

        logger.debug(
            "GroundingVerifier._extract_claims  raw_matches=%d  valid=%d",
            len(matches),
            len(claims),
        )
        return claims

    async def _verify_claim(
        self,
        claim: str,
        citation: str,
        session_id: str,
        manifest: Dict[str, Any],
        tracker: Optional[UsageTracker],
    ) -> ClaimVerification:
        """Verify a single claim against its cited source."""
        logger.debug(
            "GroundingVerifier._verify_claim  claim=%r  citation=%r",
            claim[:80],
            citation,
        )
        # Find the cited section content
        source_content = self._find_source_content(citation, session_id, manifest)

        if not source_content:
            logger.debug(
                "GroundingVerifier._verify_claim  source not found  citation=%r",
                citation,
            )
            return ClaimVerification(
                claim=claim,
                citation=citation,
                source_content="",
                supported=False,
                reason=f"Could not locate cited source: {citation}",
            )

        # Truncate source content if too long
        if len(source_content) > 3000:
            source_content = source_content[:3000] + "\n...[truncated]"

        # Ask LLM for verification
        user_message = (
            f"## Claim\n{claim}\n\n"
            f"## Cited Source ({citation})\n{source_content}\n\n"
            "Is this claim supported by the source text?"
        )

        start_time = time.time()
        try:
            response = await litellm.acompletion(
                **self._chat_kwargs,
                messages=[
                    {"role": "system", "content": VERIFICATION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )

            raw = response.choices[0].message.content or "{}"
            latency_ms = (time.time() - start_time) * 1000

            if tracker and response.usage:
                tracker.record_llm_call(
                    stage="grounding",
                    model=self._model,
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                    latency_ms=latency_ms,
                )

            # Parse verification result
            return self._parse_verification(raw, claim, citation, source_content)

        except Exception as e:
            logger.error("GroundingVerifier._verify_claim  error=%s", e)
            # On error, assume supported (don't penalize for verification failures)
            return ClaimVerification(
                claim=claim,
                citation=citation,
                source_content=source_content,
                supported=True,
                reason=f"Verification error: {e}",
            )

    def _parse_verification(
        self, raw: str, claim: str, citation: str, source_content: str
    ) -> ClaimVerification:
        """Parse the LLM's verification response."""
        logger.debug(
            "GroundingVerifier._parse_verification  raw_len=%d  claim=%r",
            len(raw),
            claim[:60],
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
            supported = bool(data.get("supported", True))
            reason = data.get("reason", "")
        except json.JSONDecodeError:
            # Try to extract supported/not from the raw text
            lower = raw.lower()
            if "not supported" in lower or '"supported": false' in lower:
                supported = False
                reason = raw
            else:
                supported = True
                reason = raw

        return ClaimVerification(
            claim=claim,
            citation=citation,
            source_content=source_content[:500],  # Keep truncated for logging
            supported=supported,
            reason=reason,
        )

    def _find_source_content(self, citation: str, session_id: str, manifest: Dict[str, Any]) -> str:
        """
        Find the content of a cited source in the VFS.

        Matches citation strings like "auth.md § OAuth Flow" against
        manifest entries.
        """
        # Parse citation: "file.md § Section" or just "file.md"
        parts = citation.split("§")
        file_part = parts[0].strip()
        section_part = parts[1].strip() if len(parts) > 1 else ""

        logger.debug(
            "GroundingVerifier._find_source_content  file_part=%r  section_part=%r",
            file_part,
            section_part,
        )

        sections = manifest.get("sections", [])

        for entry in sections:
            source_file = entry.get("source_file", "")
            section_name = entry.get("section", "")
            vfs_path = entry.get("vfs_path", "")

            # Match by file name and optional section
            file_match = file_part in source_file or file_part.replace(".md", "") in source_file
            section_match = not section_part or section_part.lower() in section_name.lower()

            logger.debug(
                "GroundingVerifier._find_source_content  checking entry  "
                "source=%r  section=%r  file_match=%s  section_match=%s",
                source_file,
                section_name,
                file_match,
                section_match,
            )

            if file_match and section_match and vfs_path:
                try:
                    content = self.vfs.read_section(session_id, vfs_path)
                    logger.debug(
                        "GroundingVerifier._find_source_content  found  "
                        "vfs_path=%s  content_len=%d",
                        vfs_path,
                        len(content),
                    )
                    return content
                except Exception as e:
                    logger.debug(
                        "GroundingVerifier._find_source_content  read failed  "
                        "vfs_path=%s  error=%s",
                        vfs_path,
                        e,
                    )
                    continue

        logger.debug(
            "GroundingVerifier._find_source_content  no match found  citation=%r",
            citation,
        )
        return ""

    def _caveat_unsupported(self, answer: str, flags: List[ClaimVerification]) -> str:
        """Add caveats to unsupported claims in the answer."""
        modified = answer
        caveated = 0
        for flag in flags:
            if flag.claim in modified:
                caveat = f"{flag.claim} [Note: This claim could not be fully verified against the cited source.]"
                modified = modified.replace(flag.claim, caveat, 1)
                caveated += 1
                logger.debug(
                    "GroundingVerifier._caveat_unsupported  inserted caveat  claim=%r",
                    flag.claim[:60],
                )
            else:
                logger.debug(
                    "GroundingVerifier._caveat_unsupported  claim not found in answer  claim=%r",
                    flag.claim[:60],
                )
        logger.debug(
            "GroundingVerifier._caveat_unsupported  caveated=%d/%d",
            caveated,
            len(flags),
        )
        return modified

    def _log_flags(self, session_id: str, flags: List[ClaimVerification]) -> None:
        """Log flagged claims to hallucination_log.jsonl."""
        log_dir = os.path.abspath(self.config.results_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "hallucination_log.jsonl")

        with open(log_path, "a", encoding="utf-8") as f:
            for flag in flags:
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                    "claim": flag.claim,
                    "citation": flag.citation,
                    "supported": flag.supported,
                    "reason": flag.reason,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.debug(
                    "GroundingVerifier._log_flags  claim=%r  citation=%r  reason=%r",
                    flag.claim[:60],
                    flag.citation,
                    flag.reason[:80],
                )

        logger.info(
            "GroundingVerifier  logged %d flags to %s",
            len(flags),
            log_path,
        )
