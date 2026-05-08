"""Tests for InvestigationMemory."""

import pytest
from agent.memory import InvestigationMemory, Finding, InvestigationPhase


class TestInvestigationMemory:

    def test_initial_state(self):
        mem = InvestigationMemory(session_id="test-001")
        assert mem.phase == InvestigationPhase.TRIAGE
        assert mem.turn_count == 0
        assert len(mem.findings) == 0
        assert len(mem.evidence_gaps) == 0
        assert len(mem.entities_examined) == 0

    def test_phase_transitions(self):
        mem = InvestigationMemory(session_id="test-002")
        assert mem.phase == InvestigationPhase.TRIAGE

        mem.add_finding(Finding(
            source_tool="assess_payment_risk", entity_id="mrc_00005",
            data={"decision": "BLOCK", "ensemble_score": 0.588}, confidence=0.6,
        ))
        assert mem.phase == InvestigationPhase.DEEP_DIVE

        mem.add_finding(Finding(
            source_tool="detect_merchant_ring", entity_id="mrc_00005",
            data={"rings_detected": [{"ring_id": "ring_b"}]}, confidence=0.7,
        ))
        mem.add_finding(Finding(
            source_tool="assess_payment_risk", entity_id="mrc_00006",
            data={"decision": "BLOCK", "ensemble_score": 0.52}, confidence=0.55,
        ))
        assert mem.phase == InvestigationPhase.SYNTHESIS

    def test_entity_risk_score_takes_max(self):
        mem = InvestigationMemory(session_id="test-003")
        mem.add_finding(Finding(
            source_tool="assess_payment_risk", entity_id="mrc_00005",
            data={"decision": "REVIEW", "ensemble_score": 0.45}, confidence=0.45,
        ))
        mem.add_finding(Finding(
            source_tool="detect_merchant_ring", entity_id="mrc_00005",
            data={"rings_detected": [{"ring_id": "ring_b"}]}, confidence=0.75,
        ))
        assert mem.entity_risk_scores["mrc_00005"] == 0.75

    def test_evidence_gap_recording(self):
        mem = InvestigationMemory(session_id="test-004")
        mem.add_evidence_gap(
            tool_name="detect_merchant_ring", reason="timeout",
            args={"merchant_id": "mrc_00005"},
        )
        assert len(mem.evidence_gaps) == 1
        assert mem.evidence_gaps[0].tool_name == "detect_merchant_ring"

    def test_duplicate_tool_call_detection(self):
        mem = InvestigationMemory(session_id="test-005")
        args = {"merchant_id": "mrc_00005"}
        mem.record_tool_call("assess_payment_risk", args)
        assert mem.has_called("assess_payment_risk", args) is True
        assert mem.has_called("detect_merchant_ring", args) is False

    def test_state_summary_contains_key_sections(self):
        mem = InvestigationMemory(session_id="test-006")
        mem.mark_entity_examined("mrc_00005")
        mem.add_finding(Finding(
            source_tool="assess_payment_risk", entity_id="mrc_00005",
            data={"decision": "BLOCK", "ensemble_score": 0.588,
                  "graph_signals": ["[G] SHARED_BANK_ACCOUNT"]},
            confidence=0.6,
        ))
        summary = mem.get_state_summary()
        assert "mrc_00005" in summary
        assert "BLOCK" in summary
        assert "Findings" in summary

    def test_message_tracking(self):
        mem = InvestigationMemory(session_id="test-007")
        mem.add_message("user", "Investigate mrc_00005")
        mem.add_message("assistant", "Starting investigation...")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_get_summary_compact(self):
        mem = InvestigationMemory(session_id="test-008")
        mem.add_finding(Finding(
            source_tool="assess_payment_risk", entity_id="mrc_00005",
            data={"decision": "BLOCK", "ensemble_score": 0.588}, confidence=0.6,
        ))
        mem.mark_entity_examined("mrc_00005")
        summary = mem.get_summary()
        assert summary["session_id"] == "test-008"
        assert summary["findings_count"] == 1
        assert "mrc_00005" in summary["entities_examined"]
