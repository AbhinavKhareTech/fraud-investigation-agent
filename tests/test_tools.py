"""Tests for ToolRegistry."""

import pytest
from agent.tools import ToolRegistry


class TestToolRegistry:

    def test_mock_registry_creation(self):
        registry = ToolRegistry.from_mock()
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        tool_names = {s["name"] for s in schemas}
        assert "assess_payment_risk" in tool_names
        assert "detect_merchant_ring" in tool_names

    @pytest.mark.asyncio
    async def test_mock_risk_assess_known_merchant(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("assess_payment_risk", {"merchant_id": "mrc_00005"})
        assert result["decision"] == "BLOCK"
        assert result["ensemble_score"] > 0.5

    @pytest.mark.asyncio
    async def test_mock_risk_assess_clean_merchant(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("assess_payment_risk", {"merchant_id": "mrc_00001"})
        assert result["decision"] == "ALLOW"
        assert result["ensemble_score"] < 0.2

    @pytest.mark.asyncio
    async def test_mock_ring_detect(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("detect_merchant_ring", {"merchant_id": "mrc_00005"})
        assert len(result["rings_detected"]) > 0
        assert result["rings_detected"][0]["ring_id"] == "ring_b"

    @pytest.mark.asyncio
    async def test_mock_unknown_merchant_defaults_to_allow(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("assess_payment_risk", {"merchant_id": "mrc_99999"})
        assert result["decision"] == "ALLOW"

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        registry = ToolRegistry.from_mock()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.call("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_ring_a_refund_cycling(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("assess_payment_risk", {"merchant_id": "mrc_00002"})
        assert result["decision"] == "REVIEW"
        assert any("REFUND" in s for s in result["graph_signals"])

    @pytest.mark.asyncio
    async def test_ring_c_card_testing(self):
        registry = ToolRegistry.from_mock()
        result = await registry.call("assess_payment_risk", {"merchant_id": "mrc_00010"})
        assert result["decision"] == "BLOCK"
        assert result["ensemble_score"] > 0.7
