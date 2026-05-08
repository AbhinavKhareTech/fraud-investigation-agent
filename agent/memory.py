"""
Structured Investigation Memory.

This is NOT a chat history buffer. It is an evidence ledger that tracks:
- Entities examined and their risk profiles
- Findings from each tool call with confidence scores
- Hypotheses formed and updated as evidence accumulates
- Evidence gaps from tool failures (critical for graceful degradation eval)
- Investigation phase transitions

Design choice: dict-based state over an ORM or message log.
An investigation is a case file, not a conversation.
See ARCHITECTURE.md for the full rationale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InvestigationPhase(Enum):
    """
    Explicit investigation phases.

    TRIAGE:     Initial risk assessment. Agent calls assess_payment_risk.
    DEEP_DIVE:  Follow-up on flagged signals. Agent calls detect_merchant_ring
                or re-assesses connected entities.
    SYNTHESIS:  Agent has enough evidence (or has exhausted tools). Produces
                structured summary with risk verdict and evidence citations.
    """
    TRIAGE = "triage"
    DEEP_DIVE = "deep_dive"
    SYNTHESIS = "synthesis"


@dataclass
class Finding:
    """A single piece of evidence from a tool call."""
    source_tool: str
    entity_id: str
    data: dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source_tool": self.source_tool,
            "entity_id": self.entity_id,
            "data": self.data,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class EvidenceGap:
    """A gap in the investigation caused by a tool failure or timeout."""
    tool_name: str
    reason: str
    args: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "reason": self.reason,
            "args": self.args,
            "timestamp": self.timestamp,
        }


class InvestigationMemory:
    """
    Structured state for a single fraud investigation session.

    Usage:
        memory = InvestigationMemory(session_id="inv-001")
        memory.add_finding(Finding(...))
        memory.mark_entity_examined("mrc_00005")
        context = memory.get_state_summary()  # Feed to LLM
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.phase = InvestigationPhase.TRIAGE
        self.turn_count = 0
        self.created_at = time.time()

        # Evidence ledger
        self.findings: list[Finding] = []
        self.evidence_gaps: list[EvidenceGap] = []

        # Entity tracking
        self.entities_examined: set[str] = set()
        self.entity_risk_scores: dict[str, float] = {}

        # Tool call log (prevents duplicate calls)
        self.tool_calls: list[dict] = []

        # Conversation messages (for LLM context)
        self._messages: list[dict[str, str]] = []

    def add_finding(self, finding: Finding) -> None:
        """Add validated finding and update phase if needed."""
        self.findings.append(finding)

        # Update entity risk score - take the max, a single high-risk signal dominates
        if finding.entity_id != "unknown":
            current = self.entity_risk_scores.get(finding.entity_id, 0.0)
            self.entity_risk_scores[finding.entity_id] = max(
                current, finding.confidence
            )

        self._maybe_advance_phase()

    def add_evidence_gap(self, tool_name: str, reason: str, args: dict) -> None:
        """Record a gap from tool failure. Agent will acknowledge this."""
        self.evidence_gaps.append(
            EvidenceGap(tool_name=tool_name, reason=reason, args=args)
        )

    def mark_entity_examined(self, entity_id: str) -> None:
        self.entities_examined.add(entity_id)

    def record_tool_call(self, tool_name: str, args: dict) -> None:
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "timestamp": time.time(),
        })

    def has_called(self, tool_name: str, args: dict) -> bool:
        """Check if this exact tool call was already made."""
        return any(
            tc["tool"] == tool_name and tc["args"] == args
            for tc in self.tool_calls
        )

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def increment_turn(self) -> None:
        self.turn_count += 1

    def _maybe_advance_phase(self) -> None:
        """Auto-advance phase based on investigation depth."""
        if self.phase == InvestigationPhase.TRIAGE and len(self.findings) >= 1:
            self.phase = InvestigationPhase.DEEP_DIVE
        elif self.phase == InvestigationPhase.DEEP_DIVE and len(self.findings) >= 3:
            self.phase = InvestigationPhase.SYNTHESIS

    def get_state_summary(self) -> str:
        """
        Produce a structured text summary of investigation state.
        This is what the LLM sees - not raw JSON, but a readable briefing.
        """
        lines = [
            f"## Investigation State (Session: {self.session_id})",
            f"Phase: {self.phase.value}",
            f"Turn: {self.turn_count}",
            f"Entities examined: {', '.join(sorted(self.entities_examined)) or 'none yet'}",
            "",
        ]

        if self.entity_risk_scores:
            lines.append("### Risk Scores")
            for eid, score in sorted(self.entity_risk_scores.items(), key=lambda x: -x[1]):
                level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
                lines.append(f"  {eid}: {score:.3f} ({level})")
            lines.append("")

        if self.findings:
            lines.append(f"### Findings ({len(self.findings)} total)")
            for i, f in enumerate(self.findings, 1):
                lines.append(
                    f"  [{i}] {f.source_tool} on {f.entity_id} "
                    f"(confidence: {f.confidence:.2f})"
                )
                decision = f.data.get("decision", "")
                score = f.data.get("ensemble_score", "")
                if decision:
                    lines.append(f"      Decision: {decision}, Score: {score}")
                signals = f.data.get("graph_signals", [])
                for sig in signals[:5]:
                    lines.append(f"      Signal: {sig}")
                rings = f.data.get("rings_detected", [])
                if rings:
                    if isinstance(rings[0], dict):
                        lines.append(f"      Rings: {len(rings)} detected")
                    else:
                        lines.append(f"      Rings: {rings}")
            lines.append("")

        if self.evidence_gaps:
            lines.append(f"### Evidence Gaps ({len(self.evidence_gaps)})")
            for gap in self.evidence_gaps:
                lines.append(f"  [GAP] {gap.tool_name}: {gap.reason}")
            lines.append("")

        if self.tool_calls:
            lines.append(f"### Tools Called ({len(self.tool_calls)})")
            for tc in self.tool_calls:
                lines.append(f"  - {tc['tool']}({tc['args']})")
            lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Compact summary for API responses."""
        highest = max(
            self.entity_risk_scores.items(), key=lambda x: x[1], default=("none", 0.0)
        )
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "turn_count": self.turn_count,
            "findings_count": len(self.findings),
            "entities_examined": sorted(self.entities_examined),
            "evidence_gaps_count": len(self.evidence_gaps),
            "highest_risk_entity": {"id": highest[0], "score": highest[1]},
        }
