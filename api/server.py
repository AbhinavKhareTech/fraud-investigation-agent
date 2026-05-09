"""
FastAPI server for the fraud investigation agent.
Run: uvicorn api.server:app --reload
"""

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.orchestrator import InvestigationOrchestrator
from agent.tools import ToolRegistry
from agent.llm import LLMClient

app = FastAPI(title="Fraud Investigation Agent", version="0.1.0")
orchestrator = None


def detect_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        return "openai"
    elif os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "openai"


@app.on_event("startup")
async def startup():
    global orchestrator
    try:
        from bgi_trident.mcp.bgi_risk_engine import PaymentRiskEngine
        engine = PaymentRiskEngine(data_dir="src/data")
        engine.load()
        tools = ToolRegistry.from_engine(engine)
    except (ImportError, Exception):
        tools = ToolRegistry.from_mock()
    llm = LLMClient(provider=detect_provider())
    orchestrator = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)


class InvestigateRequest(BaseModel):
    session_id: str = "default"
    message: str


@app.post("/investigate")
async def investigate(req: InvestigateRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return await orchestrator.investigate(req.session_id, req.message)


@app.get("/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    memory = orchestrator.sessions.get(session_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return memory.get_summary()


@app.post("/sessions")
async def create_session():
    sid = f"inv-{int(time.time())}"
    orchestrator.get_or_create_session(sid)
    return {"session_id": sid}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in orchestrator.sessions:
        del orchestrator.sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
