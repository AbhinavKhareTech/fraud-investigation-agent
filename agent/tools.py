"""
Tool Registry - wraps BGI Trident engine functions as agent-callable tools.

Each tool has:
- A schema (name, description, parameters) for LLM tool-calling
- An async execute method that calls the underlying engine
- Input validation before calling the engine
- Timeout handling (configurable, default 10s)

The tools here are thin wrappers. The heavy lifting (graph construction,
XGBoost scoring, ring detection) lives in trident-payment-fraud.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0

TOOL_SCHEMAS = [
    {
        "name": "assess_payment_risk",
        "description": (
            "Assess fraud risk for a payment or merchant. Runs three-prong "
            "scoring: XGBoost velocity/amount features, graph-native signals "
            "(shared bank accounts, device mules, ring patterns), and ensemble "
            "decision. Returns ALLOW/REVIEW/BLOCK with score breakdown and "
            "graph signals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "merchant_id": {
                    "type": "string",
                    "description": "Merchant ID to assess (e.g., mrc_00005)",
                },
                "payment_id": {
                    "type": "string",
                    "description": "Optional specific payment ID",
                },
                "payer_id": {
                    "type": "string",
                    "description": "Optional payer ID for transaction-level assessment",
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount in INR",
                },
            },
            "required": ["merchant_id"],
        },
    },
    {
        "name": "detect_merchant_ring",
        "description": (
            "Deep ring analysis for a merchant. Detects shared settlement "
            "bank accounts, coordinated payer pools, and refund cycling "
            "patterns. Returns ring members, shared infrastructure, and "
            "strength scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "merchant_id": {
                    "type": "string",
                    "description": "Merchant ID to analyze for ring membership",
                },
                "depth": {
                    "type": "integer",
                    "description": "Graph traversal depth (1-3). Default 2.",
                    "default": 2,
                },
            },
            "required": ["merchant_id"],
        },
    },
]


class ToolRegistry:
    """
    Registry of callable tools for the investigation agent.

    Supports two modes:
    - Live mode: wraps the actual BGI Trident PaymentRiskEngine
    - Mock mode: returns synthetic results for eval/testing
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._timeout = DEFAULT_TIMEOUT

    @classmethod
    def from_engine(cls, engine) -> ToolRegistry:
        """Create registry backed by a live BGI Trident PaymentRiskEngine."""
        registry = cls()
        registry._tools["assess_payment_risk"] = cls._wrap_risk_assess(engine)
        registry._tools["detect_merchant_ring"] = cls._wrap_ring_detect(engine)
        return registry

    @classmethod
    def from_mock(cls, mock_data: dict[str, dict] | None = None) -> ToolRegistry:
        """Create registry with mock responses for eval/testing."""
        registry = cls()
        data = mock_data or DEFAULT_MOCK_DATA
        registry._tools["assess_payment_risk"] = cls._make_mock("assess_payment_risk", data)
        registry._tools["detect_merchant_ring"] = cls._make_mock("detect_merchant_ring", data)
        return registry

    def get_schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    async def call(self, tool_name: str, args: dict) -> dict:
        """Call a tool with timeout and error handling."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        fn = self._tools[tool_name]
        try:
            result = await asyncio.wait_for(
                self._run_tool(fn, args), timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool {tool_name} exceeded {self._timeout}s timeout")

    async def _run_tool(self, fn: Callable, args: dict) -> dict:
        """Run tool function, handling both sync and async callables."""
        result = fn(**args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _wrap_risk_assess(engine):
        def _call(merchant_id: str, payment_id: str = "", payer_id: str = "",
                  amount: float = 0.0, **kwargs) -> dict:
            return engine.assess_payment_risk(
                payment_id=payment_id or f"pay_inv_{merchant_id}",
                merchant_id=merchant_id,
                payer_id=payer_id or "investigator",
                amount=amount or 10000.0,
            )
        return _call

    @staticmethod
    def _wrap_ring_detect(engine):
        def _call(merchant_id: str, depth: int = 2, **kwargs) -> dict:
            return engine.detect_merchant_ring(merchant_id=merchant_id, depth=depth)
        return _call

    @staticmethod
    def _make_mock(tool_name: str, data: dict):
        def _call(**kwargs) -> dict:
            merchant_id = kwargs.get("merchant_id", "unknown")
            key = f"{tool_name}:{merchant_id}"
            if key in data:
                return data[key]
            if tool_name == "assess_payment_risk":
                return {"decision": "ALLOW", "ensemble_score": 0.15,
                        "graph_signals": [], "rings_detected": []}
            return {"rings_detected": [], "ring_members": []}
        return _call


# Mock data for eval scenarios - maps to synthetic fraud rings in trident-payment-fraud
DEFAULT_MOCK_DATA = {
    # Ring B merchant - BLOCK (shared settlement bank)
    "assess_payment_risk:mrc_00005": {
        "decision": "BLOCK", "ensemble_score": 0.588,
        "graph_signals": [
            "[G] SHARED_BANK_ACCOUNT: merchant mrc_00005 shares bank with 4 other merchants",
            "[G] MERCHANT_RING_HIGH: 3 HIGH-strength ring partners detected",
        ],
        "rings_detected": ["ring_b"],
    },
    "detect_merchant_ring:mrc_00005": {
        "rings_detected": [{
            "ring_id": "ring_b", "pattern": "shared_settlement_bank",
            "members": ["mrc_00005", "mrc_00006", "mrc_00008", "mrc_00009"],
            "shared_bank_account": "HDFC_XXXX4872", "strength": "HIGH",
        }],
        "ring_members": [
            {"id": "mrc_00006", "shared_payers": 7, "shared_bank": True},
            {"id": "mrc_00008", "shared_payers": 5, "shared_bank": True},
            {"id": "mrc_00009", "shared_payers": 6, "shared_bank": True},
        ],
    },
    # Clean merchant - ALLOW
    "assess_payment_risk:mrc_00001": {
        "decision": "ALLOW", "ensemble_score": 0.12,
        "graph_signals": [], "rings_detected": [],
    },
    "detect_merchant_ring:mrc_00001": {
        "rings_detected": [], "ring_members": [],
    },
    # Ring A - refund cycling (REVIEW)
    "assess_payment_risk:mrc_00002": {
        "decision": "REVIEW", "ensemble_score": 0.45,
        "graph_signals": [
            "[G] REFUND_CYCLING: 40 purchase-refund cycles in 30 days",
            "[X] HIGH_REFUND_RATE: 68% refund rate vs 5% baseline",
        ],
        "rings_detected": ["ring_a"],
    },
    "detect_merchant_ring:mrc_00002": {
        "rings_detected": [{
            "ring_id": "ring_a", "pattern": "refund_cycling",
            "members": ["mrc_00002", "mrc_00003", "mrc_00004"],
            "shared_payers": 15, "strength": "MEDIUM",
        }],
        "ring_members": [
            {"id": "mrc_00003", "shared_payers": 12, "shared_bank": False},
            {"id": "mrc_00004", "shared_payers": 10, "shared_bank": False},
        ],
    },
    # Ring C - card testing burst (BLOCK)
    "assess_payment_risk:mrc_00010": {
        "decision": "BLOCK", "ensemble_score": 0.82,
        "graph_signals": [
            "[X] VELOCITY_SPIKE: 35 transactions in 36 seconds",
            "[G] DEVICE_MULE: single device across 35 micro-transactions",
            "[X] AMOUNT_ANOMALY: all transactions under INR 50 (testing pattern)",
        ],
        "rings_detected": ["ring_c"],
    },
    # Ring B peripherals
    "assess_payment_risk:mrc_00006": {
        "decision": "BLOCK", "ensemble_score": 0.52,
        "graph_signals": [
            "[G] SHARED_BANK_ACCOUNT: shares bank with mrc_00005",
            "[G] MERCHANT_RING_HIGH: part of ring_b",
        ],
        "rings_detected": ["ring_b"],
    },
    "detect_merchant_ring:mrc_00006": {
        "rings_detected": [{
            "ring_id": "ring_b", "pattern": "shared_settlement_bank",
            "members": ["mrc_00005", "mrc_00006", "mrc_00008", "mrc_00009"],
            "shared_bank_account": "HDFC_XXXX4872", "strength": "HIGH",
        }],
        "ring_members": [
            {"id": "mrc_00005", "shared_payers": 7, "shared_bank": True},
            {"id": "mrc_00008", "shared_payers": 5, "shared_bank": True},
            {"id": "mrc_00009", "shared_payers": 6, "shared_bank": True},
        ],
    },
    "assess_payment_risk:mrc_00008": {
        "decision": "BLOCK", "ensemble_score": 0.55,
        "graph_signals": ["[G] SHARED_BANK_ACCOUNT: shares bank with mrc_00005"],
        "rings_detected": ["ring_b"],
    },
    "detect_merchant_ring:mrc_00008": {
        "rings_detected": [{
            "ring_id": "ring_b", "pattern": "shared_settlement_bank",
            "members": ["mrc_00005", "mrc_00006", "mrc_00008", "mrc_00009"],
            "shared_bank_account": "HDFC_XXXX4872", "strength": "HIGH",
        }],
        "ring_members": [
            {"id": "mrc_00005", "shared_payers": 7, "shared_bank": True},
            {"id": "mrc_00006", "shared_payers": 5, "shared_bank": True},
            {"id": "mrc_00009", "shared_payers": 6, "shared_bank": True},
        ],
    },
    "assess_payment_risk:mrc_00009": {
        "decision": "REVIEW", "ensemble_score": 0.48,
        "graph_signals": ["[G] SHARED_BANK_ACCOUNT: shares bank with mrc_00005"],
        "rings_detected": ["ring_b"],
    },
    # Ring A peripherals
    "assess_payment_risk:mrc_00003": {
        "decision": "REVIEW", "ensemble_score": 0.38,
        "graph_signals": ["[G] REFUND_CYCLING: connected to mrc_00002 refund ring"],
        "rings_detected": ["ring_a"],
    },
    "detect_merchant_ring:mrc_00003": {
        "rings_detected": [{
            "ring_id": "ring_a", "pattern": "refund_cycling",
            "members": ["mrc_00002", "mrc_00003", "mrc_00004"],
            "shared_payers": 12, "strength": "MEDIUM",
        }],
        "ring_members": [
            {"id": "mrc_00002", "shared_payers": 12, "shared_bank": False},
            {"id": "mrc_00004", "shared_payers": 10, "shared_bank": False},
        ],
    },
    "assess_payment_risk:mrc_00004": {
        "decision": "REVIEW", "ensemble_score": 0.35,
        "graph_signals": ["[G] REFUND_CYCLING: connected to mrc_00002 refund ring"],
        "rings_detected": ["ring_a"],
    },
}
